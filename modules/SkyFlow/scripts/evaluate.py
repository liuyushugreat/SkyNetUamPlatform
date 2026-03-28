#!/usr/bin/env python3
"""Evaluate a trained TR-GAT checkpoint on UrbanAir-500 test set.

Usage:
    python scripts/evaluate.py --checkpoint outputs/best_model.pt
    python scripts/evaluate.py --checkpoint outputs/best_model.pt --device cpu
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from skyflow.config import SkyFlowConfig
from skyflow.data.urbanair500 import UrbanAir500
from skyflow.models.tr_gat import TRGAT
from skyflow.models.conflict_head import ConflictScoringHead
from skyflow.training.metrics import ConflictMetrics, LatencyTimer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate SkyFlow TR-GAT")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-scenarios", type=int, default=5)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config", SkyFlowConfig.from_yaml(args.config))

    model = TRGAT(
        node_feature_dim=cfg.data.uav_feature_dim,
        embed_dim=cfg.model.embed_dim,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        num_relations=cfg.model.num_relation_types,
        temporal_dim=cfg.model.temporal_dim,
        recurrent_dim=cfg.model.recurrent_dim,
        dropout=0.0,
    ).to(device)

    head = ConflictScoringHead(
        embed_dim=cfg.model.embed_dim,
        recurrent_dim=cfg.model.recurrent_dim,
        dropout=0.0,
    ).to(device)

    model.load_state_dict(ckpt["model"])
    head.load_state_dict(ckpt["head"])
    model.eval()
    head.eval()

    logger.info(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")
    logger.info(f"Parameters: {model.count_parameters():,}")

    n_uavs = 50 if args.quick else cfg.data.num_uavs
    sim = UrbanAir500(num_uavs=n_uavs, seed=12345)
    test_data = sim.generate_dataset(
        "test", args.num_scenarios,
        10.0 if args.quick else 60.0,
        device,
    )

    metrics = ConflictMetrics(threshold=cfg.training.conflict_threshold)

    with torch.no_grad():
        for snapshot, labels in test_data:
            snapshot.node_features = snapshot.node_features.to(device)
            snapshot.edge_indices = {r: e.to(device) for r, e in snapshot.edge_indices.items()}
            snapshot.edge_deltas = {r: d.to(device) for r, d in snapshot.edge_deltas.items()}
            if snapshot.conflict_pairs is not None:
                snapshot.conflict_pairs = snapshot.conflict_pairs.to(device)
            labels = labels.to(device)

            timer = LatencyTimer()
            with timer:
                node_emb, rec_state = model(
                    snapshot.node_features,
                    snapshot.edge_indices,
                    snapshot.edge_deltas,
                )
                pairs = snapshot.conflict_pairs
                if pairs is None or pairs.size(1) == 0:
                    continue
                preds = head(
                    node_emb[pairs[0]], node_emb[pairs[1]],
                    rec_state[pairs[0]], rec_state[pairs[1]],
                )

            metrics.update(preds, labels, latency_ms=timer.elapsed_ms)

    result = metrics.compute()
    logger.info("\n" + "=" * 50)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 50)
    logger.info(f"  CDR (Recall):     {result.cdr:.4f}")
    logger.info(f"  FAR:              {result.far:.4f}")
    logger.info(f"  Precision:        {result.precision:.4f}")
    logger.info(f"  F1:               {result.f1:.4f}")
    logger.info(f"  Latency (95pct):  {result.latency_ms:.1f} ms")
    logger.info(f"  Latency (mean):   {result.latency_mean_ms:.1f} ms")
    logger.info(f"  Total pairs:      {result.num_pairs}")
    logger.info(f"  Positive pairs:   {result.num_positives}")

    output_path = Path(cfg.output_dir) / "eval_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "cdr": result.cdr, "far": result.far,
            "f1": result.f1, "precision": result.precision,
            "latency_ms": result.latency_ms,
            "num_pairs": result.num_pairs,
        }, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
