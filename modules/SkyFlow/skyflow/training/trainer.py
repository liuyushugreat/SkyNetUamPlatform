"""SkyFlow training loop with cosine annealing and multi-seed evaluation."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from skyflow.config import SkyFlowConfig
from skyflow.models.tr_gat import TRGAT
from skyflow.models.conflict_head import ConflictScoringHead
from skyflow.data.tkg_builder import TKGSnapshot
from skyflow.training.losses import FocalLoss
from skyflow.training.metrics import ConflictMetrics, LatencyTimer

logger = logging.getLogger(__name__)


class SkyFlowTrainer:
    """End-to-end trainer for TR-GAT conflict detection."""

    def __init__(self, cfg: SkyFlowConfig, device: Optional[torch.device] = None):
        self.cfg = cfg

        if device is not None:
            self.device = device
        elif cfg.training.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(cfg.training.device)

        self.model: Optional[TRGAT] = None
        self.head: Optional[ConflictScoringHead] = None

    def build_model(self) -> Tuple[TRGAT, ConflictScoringHead]:
        mc = self.cfg.model
        self.model = TRGAT(
            node_feature_dim=self.cfg.data.uav_feature_dim,
            embed_dim=mc.embed_dim,
            num_layers=mc.num_layers,
            num_heads=mc.num_heads,
            num_relations=mc.num_relation_types,
            temporal_dim=mc.temporal_dim,
            recurrent_dim=mc.recurrent_dim,
            dropout=mc.dropout,
        ).to(self.device)

        self.head = ConflictScoringHead(
            embed_dim=mc.embed_dim,
            recurrent_dim=mc.recurrent_dim,
            dropout=mc.dropout,
        ).to(self.device)

        total_params = self.model.count_parameters() + sum(
            p.numel() for p in self.head.parameters() if p.requires_grad
        )
        logger.info(f"Model parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        return self.model, self.head

    def train(
        self,
        train_data: List[Tuple[TKGSnapshot, torch.Tensor]],
        val_data: List[Tuple[TKGSnapshot, torch.Tensor]],
        seed: int = 42,
    ) -> Dict:
        """Train for one seed, return best metrics."""
        torch.manual_seed(seed)
        np.random.seed(seed)

        if self.model is None:
            self.build_model()

        tc = self.cfg.training
        params = list(self.model.parameters()) + list(self.head.parameters())
        optimizer = Adam(params, lr=tc.learning_rate, weight_decay=tc.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=tc.epochs)
        criterion = FocalLoss(gamma=tc.focal_gamma)

        best_f1 = -1.0
        best_metrics = {}
        output_dir = Path(self.cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        def _save_checkpoint(metrics_dict, epoch):
            torch.save({
                "model": self.model.state_dict(),
                "head": self.head.state_dict(),
                "epoch": epoch,
                "metrics": metrics_dict,
                "config": self.cfg,
            }, output_dir / "best_model.pt")

        for epoch in range(tc.epochs):
            self.model.train()
            self.head.train()
            epoch_loss = 0.0
            n_batches = 0

            np.random.shuffle(train_data)

            for snapshot, labels in train_data:
                snapshot = self._to_device(snapshot)
                labels = labels.to(self.device)

                optimizer.zero_grad()

                node_emb, rec_state = self.model(
                    snapshot.node_features,
                    snapshot.edge_indices,
                    snapshot.edge_deltas,
                )

                pairs = snapshot.conflict_pairs
                if pairs is None or pairs.size(1) == 0:
                    continue

                h_i = node_emb[pairs[0]]
                h_j = node_emb[pairs[1]]
                s_i = rec_state[pairs[0]]
                s_j = rec_state[pairs[1]]

                preds = self.head(h_i, h_j, s_i, s_j)
                loss = criterion(preds, labels)

                loss.backward()
                nn.utils.clip_grad_norm_(params, tc.gradient_clip_norm)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                val_metrics = self.evaluate(val_data)
                logger.info(
                    f"Epoch {epoch+1}/{tc.epochs} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Val CDR: {val_metrics.cdr:.4f} | "
                    f"Val F1: {val_metrics.f1:.4f} | "
                    f"Val FAR: {val_metrics.far:.4f}"
                )

                if val_metrics.f1 > best_f1:
                    best_f1 = val_metrics.f1
                    best_metrics = {
                        "cdr": val_metrics.cdr,
                        "far": val_metrics.far,
                        "f1": val_metrics.f1,
                        "precision": val_metrics.precision,
                        "latency_ms": val_metrics.latency_ms,
                        "epoch": epoch + 1,
                        "seed": seed,
                    }
                    _save_checkpoint(best_metrics, epoch + 1)

        if not (output_dir / "best_model.pt").exists():
            best_metrics = {"cdr": 0, "far": 1, "f1": 0, "precision": 0,
                            "latency_ms": 0, "epoch": tc.epochs, "seed": seed}
            _save_checkpoint(best_metrics, tc.epochs)

        return best_metrics

    @torch.no_grad()
    def evaluate(
        self,
        data: List[Tuple[TKGSnapshot, torch.Tensor]],
    ) -> "ConflictMetrics":
        from skyflow.training.metrics import MetricResult

        self.model.eval()
        self.head.eval()
        metrics = ConflictMetrics(threshold=self.cfg.training.conflict_threshold)

        for snapshot, labels in data:
            snapshot = self._to_device(snapshot)
            labels = labels.to(self.device)

            timer = LatencyTimer()
            with timer:
                node_emb, rec_state = self.model(
                    snapshot.node_features,
                    snapshot.edge_indices,
                    snapshot.edge_deltas,
                )

                pairs = snapshot.conflict_pairs
                if pairs is None or pairs.size(1) == 0:
                    continue

                h_i = node_emb[pairs[0]]
                h_j = node_emb[pairs[1]]
                s_i = rec_state[pairs[0]]
                s_j = rec_state[pairs[1]]

                preds = self.head(h_i, h_j, s_i, s_j)

            metrics.update(preds, labels, latency_ms=timer.elapsed_ms)

        return metrics.compute()

    def train_multi_seed(
        self,
        train_data: List[Tuple[TKGSnapshot, torch.Tensor]],
        val_data: List[Tuple[TKGSnapshot, torch.Tensor]],
        test_data: List[Tuple[TKGSnapshot, torch.Tensor]],
    ) -> Dict:
        """Train across multiple seeds and report mean ± std."""
        all_results = []
        base_seed = self.cfg.training.seed

        for seed_idx in range(self.cfg.training.num_seeds):
            seed = base_seed + seed_idx
            logger.info(f"\n{'='*60}\nSeed {seed_idx+1}/{self.cfg.training.num_seeds} (seed={seed})\n{'='*60}")

            self.model = None
            self.head = None
            self.build_model()

            best = self.train(train_data, val_data, seed=seed)

            ckpt = torch.load(
                Path(self.cfg.output_dir) / "best_model.pt",
                map_location=self.device,
                weights_only=False,
            )
            self.model.load_state_dict(ckpt["model"])
            self.head.load_state_dict(ckpt["head"])

            test_metrics = self.evaluate(test_data)
            result = {
                "seed": seed,
                "cdr": test_metrics.cdr,
                "far": test_metrics.far,
                "f1": test_metrics.f1,
                "precision": test_metrics.precision,
                "latency_ms": test_metrics.latency_ms,
            }
            all_results.append(result)
            logger.info(f"Seed {seed} test: CDR={result['cdr']:.4f}, F1={result['f1']:.4f}")

        summary = self._summarize_seeds(all_results)
        output_dir = Path(self.cfg.output_dir)
        with open(output_dir / "multi_seed_results.json", "w") as f:
            json.dump({"seeds": all_results, "summary": summary}, f, indent=2)

        return summary

    def _summarize_seeds(self, results: List[Dict]) -> Dict:
        metrics_keys = ["cdr", "far", "f1", "precision", "latency_ms"]
        summary = {}
        for key in metrics_keys:
            values = [r[key] for r in results]
            summary[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        return summary

    def _to_device(self, snapshot: TKGSnapshot) -> TKGSnapshot:
        snapshot.node_features = snapshot.node_features.to(self.device)
        snapshot.edge_indices = {
            r: e.to(self.device) for r, e in snapshot.edge_indices.items()
        }
        snapshot.edge_deltas = {
            r: d.to(self.device) for r, d in snapshot.edge_deltas.items()
        }
        if snapshot.conflict_pairs is not None:
            snapshot.conflict_pairs = snapshot.conflict_pairs.to(self.device)
        return snapshot
