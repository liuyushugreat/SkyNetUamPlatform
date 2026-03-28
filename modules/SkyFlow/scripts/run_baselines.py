#!/usr/bin/env python3
"""Run all baselines on UrbanAir-500 and compare with TR-GAT.

Usage:
    python scripts/run_baselines.py
    python scripts/run_baselines.py --quick --device cpu
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np

from skyflow.config import SkyFlowConfig
from skyflow.data.urbanair500 import UrbanAir500
from skyflow.baselines.velocity_obstacle import VelocityObstacle
from skyflow.baselines.lstm_pair import LSTMPair
from skyflow.baselines.transformer_pair import TransformerPair
from skyflow.baselines.stgcn import STGCN
from skyflow.baselines.gat_static import GATStatic
from skyflow.training.metrics import ConflictMetrics, LatencyTimer
from skyflow.training.losses import FocalLoss
from skyflow.utils.visualization import generate_all_figures

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_baseline(model, train_data, val_data, cfg, device, epochs=None):
    """Simple training loop for learning-based baselines."""
    if epochs is None:
        epochs = cfg.training.epochs

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    criterion = FocalLoss(gamma=cfg.training.focal_gamma)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n = 0
        for snapshot, labels in train_data:
            optimizer.zero_grad()
            preds = model(snapshot)
            if preds.numel() == 0:
                continue
            loss = criterion(preds, labels.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip_norm)
            optimizer.step()
            total_loss += loss.item()
            n += 1

        if (epoch + 1) % max(epochs // 5, 1) == 0:
            logger.info(f"  Epoch {epoch+1}/{epochs} loss: {total_loss/max(n,1):.4f}")


def evaluate_model(model, test_data, cfg, device, is_deterministic=False):
    """Evaluate a model and return metrics."""
    metrics = ConflictMetrics(threshold=cfg.training.conflict_threshold)

    if not is_deterministic and hasattr(model, "eval"):
        model.eval()

    with torch.no_grad():
        for snapshot, labels in test_data:
            labels = labels.to(device)
            timer = LatencyTimer()
            with timer:
                if is_deterministic:
                    preds = model.predict(snapshot)
                else:
                    preds = model(snapshot)
            if preds.numel() == 0:
                continue
            metrics.update(preds.to(device), labels, latency_ms=timer.elapsed_ms)

    return metrics.compute()


def main():
    parser = argparse.ArgumentParser(description="Run baselines")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    cfg = SkyFlowConfig.from_yaml(args.config)
    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    if args.quick:
        cfg.data.num_uavs = 50
        cfg.training.epochs = 10

    logger.info(f"Device: {device}, UAVs: {cfg.data.num_uavs}")

    logger.info("Generating data...")
    sim = UrbanAir500(num_uavs=cfg.data.num_uavs, seed=42)
    n_sc = 3 if args.quick else 10
    dur = 10.0 if args.quick else 60.0

    train_data = sim.generate_dataset("train", n_sc, dur, device)
    test_data = sim.generate_dataset("test", max(n_sc // 2, 1), dur, device)

    logger.info(f"Train: {len(train_data)}, Test: {len(test_data)}")

    epochs = cfg.training.epochs
    all_results = {}

    # 1. Velocity Obstacle (deterministic)
    logger.info("\n--- Velocity Obstacle ---")
    vo = VelocityObstacle()
    result = evaluate_model(vo, test_data, cfg, device, is_deterministic=True)
    all_results["VO"] = _to_dict(result)
    logger.info(f"  CDR: {result.cdr:.4f}, FAR: {result.far:.4f}, F1: {result.f1:.4f}, Latency: {result.latency_ms:.1f}ms")

    # 2. LSTM-Pair
    logger.info("\n--- LSTM-Pair ---")
    lstm = LSTMPair(input_dim=cfg.data.uav_feature_dim).to(device)
    logger.info(f"  Parameters: {lstm.count_parameters():,}")
    train_baseline(lstm, train_data, test_data, cfg, device, epochs)
    result = evaluate_model(lstm, test_data, cfg, device)
    all_results["LSTM-P"] = _to_dict(result)
    logger.info(f"  CDR: {result.cdr:.4f}, FAR: {result.far:.4f}, F1: {result.f1:.4f}, Latency: {result.latency_ms:.1f}ms")

    # 3. Transformer-Pair
    logger.info("\n--- Transformer-Pair ---")
    tfm = TransformerPair(input_dim=cfg.data.uav_feature_dim).to(device)
    logger.info(f"  Parameters: {tfm.count_parameters():,}")
    train_baseline(tfm, train_data, test_data, cfg, device, epochs)
    result = evaluate_model(tfm, test_data, cfg, device)
    all_results["Tfm-P"] = _to_dict(result)
    logger.info(f"  CDR: {result.cdr:.4f}, FAR: {result.far:.4f}, F1: {result.f1:.4f}, Latency: {result.latency_ms:.1f}ms")

    # 4. STGCN
    logger.info("\n--- STGCN ---")
    stgcn = STGCN(input_dim=cfg.data.uav_feature_dim).to(device)
    logger.info(f"  Parameters: {stgcn.count_parameters():,}")
    train_baseline(stgcn, train_data, test_data, cfg, device, epochs)
    result = evaluate_model(stgcn, test_data, cfg, device)
    all_results["STGCN"] = _to_dict(result)
    logger.info(f"  CDR: {result.cdr:.4f}, FAR: {result.far:.4f}, F1: {result.f1:.4f}, Latency: {result.latency_ms:.1f}ms")

    # 5. GAT-Static
    logger.info("\n--- GAT-Static ---")
    gat = GATStatic(input_dim=cfg.data.uav_feature_dim).to(device)
    logger.info(f"  Parameters: {gat.count_parameters():,}")
    train_baseline(gat, train_data, test_data, cfg, device, epochs)
    result = evaluate_model(gat, test_data, cfg, device)
    all_results["GAT-S"] = _to_dict(result)
    logger.info(f"  CDR: {result.cdr:.4f}, FAR: {result.far:.4f}, F1: {result.f1:.4f}, Latency: {result.latency_ms:.1f}ms")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "baseline_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\nResults saved to {output_dir / 'baseline_results.json'}")

    chart_results = {
        name: {
            "cdr_mean": r["cdr"], "cdr_std": 0,
            "far_mean": r["far"], "far_std": 0,
            "f1_mean": r["f1"], "f1_std": 0,
            "precision_mean": r["precision"], "precision_std": 0,
            "recall_mean": r["cdr"], "recall_std": 0,
            "latency_mean": r["latency_ms"], "latency_std": 0,
        }
        for name, r in all_results.items()
    }
    generate_all_figures(chart_results, output_dir / "charts")
    logger.info(f"Figures saved to {output_dir / 'charts'}/")


def _to_dict(result):
    return {
        "cdr": result.cdr,
        "far": result.far,
        "f1": result.f1,
        "precision": result.precision,
        "latency_ms": result.latency_ms,
    }


if __name__ == "__main__":
    main()
