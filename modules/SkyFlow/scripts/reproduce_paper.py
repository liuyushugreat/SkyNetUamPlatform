#!/usr/bin/env python3
"""One-click reproduction of all paper experiments and figures.

Runs the full experimental pipeline:
  1. Generate UrbanAir-500 benchmark data
  2. Train TR-GAT across 5 seeds
  3. Train and evaluate all baselines
  4. Run ablation study (TR-GAT vs TR-GAT-NoTemp)
  5. Generate all paper figures and tables

Usage:
    python scripts/reproduce_paper.py                    # Full reproduction
    python scripts/reproduce_paper.py --quick --device cpu  # Quick verification
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from skyflow.config import SkyFlowConfig
from skyflow.data.urbanair500 import UrbanAir500
from skyflow.models.tr_gat import TRGAT
from skyflow.models.conflict_head import ConflictScoringHead
from skyflow.baselines.velocity_obstacle import VelocityObstacle
from skyflow.baselines.lstm_pair import LSTMPair
from skyflow.baselines.transformer_pair import TransformerPair
from skyflow.baselines.stgcn import STGCN
from skyflow.baselines.gat_static import GATStatic
from skyflow.training.trainer import SkyFlowTrainer
from skyflow.training.losses import FocalLoss
from skyflow.training.metrics import ConflictMetrics, LatencyTimer
from skyflow.utils.visualization import generate_all_figures

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Reproduce SkyFlow paper results")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--quick", action="store_true", help="Quick verification (~5 min on CPU)")
    parser.add_argument("--skip-baselines", action="store_true")
    args = parser.parse_args()

    cfg = SkyFlowConfig.from_yaml(args.config)

    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    if args.quick:
        cfg.data.num_uavs = 50
        cfg.training.epochs = 10
        cfg.training.num_seeds = 2
        n_train_scenarios = 3
        n_val_scenarios = 2
        n_test_scenarios = 2
        scenario_duration = 10.0
    else:
        n_train_scenarios = 20
        n_val_scenarios = 5
        n_test_scenarios = 5
        scenario_duration = 60.0

    start_time = time.time()
    logger.info("=" * 70)
    logger.info("SkyFlow Paper Reproduction Pipeline")
    logger.info(f"Device: {device} | UAVs: {cfg.data.num_uavs} | Quick: {args.quick}")
    logger.info("=" * 70)

    # ── Step 1: Generate Data ──
    logger.info("\n[Step 1/5] Generating UrbanAir-500 benchmark data...")
    sim = UrbanAir500(num_uavs=cfg.data.num_uavs, seed=cfg.training.seed)
    train_data = sim.generate_dataset("train", n_train_scenarios, scenario_duration, device)
    val_data = sim.generate_dataset("val", n_val_scenarios, scenario_duration, device)
    test_data = sim.generate_dataset("test", n_test_scenarios, scenario_duration, device)
    logger.info(f"  Train: {len(train_data)} snapshots | Val: {len(val_data)} | Test: {len(test_data)}")

    all_results = {}
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 2: Train TR-GAT (multi-seed) ──
    logger.info(f"\n[Step 2/5] Training TR-GAT ({cfg.training.num_seeds} seeds, {cfg.training.epochs} epochs)...")
    trainer = SkyFlowTrainer(cfg, device=device)
    trgat_summary = trainer.train_multi_seed(train_data, val_data, test_data)
    all_results["TR-GAT"] = {k: v["mean"] for k, v in trgat_summary.items()}
    all_results["TR-GAT"]["cdr_std"] = trgat_summary["cdr"]["std"]
    all_results["TR-GAT"]["far_std"] = trgat_summary["far"]["std"]
    all_results["TR-GAT"]["f1_std"] = trgat_summary["f1"]["std"]
    logger.info(f"  TR-GAT: CDR={trgat_summary['cdr']['mean']:.4f}±{trgat_summary['cdr']['std']:.4f}")

    # ── Step 3: TR-GAT-NoTemp Ablation ──
    logger.info("\n[Step 3/5] TR-GAT-NoTemp ablation...")
    notp_cfg = SkyFlowConfig.from_yaml(args.config)
    if args.quick:
        notp_cfg.data.num_uavs = cfg.data.num_uavs
        notp_cfg.training.epochs = cfg.training.epochs
        notp_cfg.training.num_seeds = cfg.training.num_seeds
    notp_cfg.model.temporal_dim = 2
    notp_cfg.output_dir = str(output_dir / "ablation_notemp")

    notp_trainer = SkyFlowTrainer(notp_cfg, device=device)

    notp_seeds_results = []
    for si in range(cfg.training.num_seeds):
        seed = cfg.training.seed + si
        notp_trainer.model = None
        notp_trainer.head = None
        notp_trainer.build_model()
        notp_trainer.train(train_data, val_data, seed=seed)
        r = notp_trainer.evaluate(test_data)
        notp_seeds_results.append({"cdr": r.cdr, "far": r.far, "f1": r.f1, "latency_ms": r.latency_ms, "precision": r.precision})

    notp_cdr = [r["cdr"] for r in notp_seeds_results]
    notp_far = [r["far"] for r in notp_seeds_results]
    notp_f1 = [r["f1"] for r in notp_seeds_results]
    all_results["TR-GAT-NT"] = {
        "cdr": np.mean(notp_cdr), "cdr_std": np.std(notp_cdr),
        "far": np.mean(notp_far), "far_std": np.std(notp_far),
        "f1": np.mean(notp_f1), "f1_std": np.std(notp_f1),
        "precision": np.mean([r["precision"] for r in notp_seeds_results]),
        "latency_ms": np.mean([r["latency_ms"] for r in notp_seeds_results]),
    }
    logger.info(f"  TR-GAT-NT: CDR={np.mean(notp_cdr):.4f}±{np.std(notp_cdr):.4f}")

    # ── Step 4: Baselines ──
    if not args.skip_baselines:
        logger.info("\n[Step 4/5] Training and evaluating baselines...")
        baseline_epochs = cfg.training.epochs

        # VO
        logger.info("  [4a] Velocity Obstacle (deterministic)...")
        vo = VelocityObstacle()
        vo_metrics = _eval_deterministic(vo, test_data, cfg, device)
        all_results["VO"] = vo_metrics
        logger.info(f"    CDR={vo_metrics['cdr']:.4f}")

        # LSTM-Pair
        logger.info("  [4b] LSTM-Pair...")
        lstm = LSTMPair(input_dim=cfg.data.uav_feature_dim).to(device)
        _train_simple(lstm, train_data, cfg, device, baseline_epochs)
        lstm_metrics = _eval_nn(lstm, test_data, cfg, device)
        all_results["LSTM-P"] = lstm_metrics
        logger.info(f"    CDR={lstm_metrics['cdr']:.4f}")

        # Transformer-Pair
        logger.info("  [4c] Transformer-Pair...")
        tfm = TransformerPair(input_dim=cfg.data.uav_feature_dim).to(device)
        _train_simple(tfm, train_data, cfg, device, baseline_epochs)
        tfm_metrics = _eval_nn(tfm, test_data, cfg, device)
        all_results["Tfm-P"] = tfm_metrics
        logger.info(f"    CDR={tfm_metrics['cdr']:.4f}")

        # STGCN
        logger.info("  [4d] STGCN...")
        stgcn = STGCN(input_dim=cfg.data.uav_feature_dim).to(device)
        _train_simple(stgcn, train_data, cfg, device, baseline_epochs)
        stgcn_metrics = _eval_nn(stgcn, test_data, cfg, device)
        all_results["STGCN"] = stgcn_metrics
        logger.info(f"    CDR={stgcn_metrics['cdr']:.4f}")

        # GAT-Static
        logger.info("  [4e] GAT-Static...")
        gat = GATStatic(input_dim=cfg.data.uav_feature_dim).to(device)
        _train_simple(gat, train_data, cfg, device, baseline_epochs)
        gat_metrics = _eval_nn(gat, test_data, cfg, device)
        all_results["GAT-S"] = gat_metrics
        logger.info(f"    CDR={gat_metrics['cdr']:.4f}")
    else:
        logger.info("\n[Step 4/5] Skipping baselines (--skip-baselines)")

    # ── Step 5: Generate Figures ──
    logger.info("\n[Step 5/5] Generating paper figures...")
    chart_data = {}
    for name, r in all_results.items():
        chart_data[name] = {
            "cdr_mean": r.get("cdr", 0), "cdr_std": r.get("cdr_std", 0),
            "far_mean": r.get("far", 0), "far_std": r.get("far_std", 0),
            "f1_mean": r.get("f1", 0), "f1_std": r.get("f1_std", 0),
            "precision_mean": r.get("precision", 0), "precision_std": 0,
            "recall_mean": r.get("cdr", 0), "recall_std": r.get("cdr_std", 0),
            "latency_mean": r.get("latency_ms", 0), "latency_std": 0,
        }
    generate_all_figures(chart_data, output_dir / "charts")

    with open(output_dir / "all_results.json", "w") as f:
        json.dump({k: {kk: float(vv) for kk, vv in v.items()} for k, v in all_results.items()}, f, indent=2)

    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 70)
    logger.info("REPRODUCTION COMPLETE")
    logger.info(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    logger.info("=" * 70)

    logger.info("\nFinal Results:")
    logger.info(f"{'Method':<12} {'CDR':>8} {'FAR':>8} {'F1':>8} {'Latency':>10}")
    logger.info("-" * 50)
    method_order = ["VO", "LSTM-P", "Tfm-P", "STGCN", "GAT-S", "TR-GAT-NT", "TR-GAT"]
    for m in method_order:
        if m in all_results:
            r = all_results[m]
            logger.info(f"{m:<12} {r['cdr']:>8.4f} {r['far']:>8.4f} {r['f1']:>8.4f} {r.get('latency_ms', 0):>8.1f}ms")

    logger.info(f"\nAll outputs saved to: {output_dir.absolute()}/")


def _train_simple(model, data, cfg, device, epochs):
    criterion = FocalLoss(gamma=cfg.training.focal_gamma)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    model.train()
    for epoch in range(epochs):
        for snapshot, labels in data:
            optimizer.zero_grad()
            preds = model(snapshot)
            if preds.numel() == 0:
                continue
            loss = criterion(preds, labels.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip_norm)
            optimizer.step()


def _eval_deterministic(model, data, cfg, device):
    metrics = ConflictMetrics(threshold=cfg.training.conflict_threshold)
    for snapshot, labels in data:
        timer = LatencyTimer()
        with timer:
            preds = model.predict(snapshot)
        if preds.numel() == 0:
            continue
        metrics.update(preds.to(device), labels.to(device), latency_ms=timer.elapsed_ms)
    r = metrics.compute()
    return {"cdr": r.cdr, "far": r.far, "f1": r.f1, "precision": r.precision, "latency_ms": r.latency_ms}


@torch.no_grad()
def _eval_nn(model, data, cfg, device):
    model.eval()
    metrics = ConflictMetrics(threshold=cfg.training.conflict_threshold)
    for snapshot, labels in data:
        timer = LatencyTimer()
        with timer:
            preds = model(snapshot)
        if preds.numel() == 0:
            continue
        metrics.update(preds.to(device), labels.to(device), latency_ms=timer.elapsed_ms)
    r = metrics.compute()
    return {"cdr": r.cdr, "far": r.far, "f1": r.f1, "precision": r.precision, "latency_ms": r.latency_ms}


if __name__ == "__main__":
    main()
