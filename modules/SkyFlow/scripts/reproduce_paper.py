#!/usr/bin/env python3
"""One-click reproduction of all paper experiments and figures.

Runs the full experimental pipeline:
  1. Generate UrbanAir-500 benchmark data
  2. Train TR-GAT across 5 seeds
  3. Train and evaluate all baselines
  4. Run ablation study (TR-GAT vs TR-GAT-NoTemp)
  5. Scalability sweep (100–500 UAVs)
  6. Statistical significance tests (Bonferroni-corrected)
  7. Generate all paper figures and tables

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
from skyflow.data.tkg_builder import TKGBuilder
from skyflow.models.tr_gat import TRGAT
from skyflow.models.conflict_head import ConflictScoringHead
from skyflow.baselines.velocity_obstacle import VelocityObstacle
from skyflow.baselines.lstm_pair import LSTMPair
from skyflow.baselines.transformer_pair import TransformerPair
from skyflow.baselines.stgcn import STGCN
from skyflow.baselines.gat_static import GATStatic
from skyflow.training.trainer import SkyFlowTrainer
from skyflow.training.losses import FocalLoss
from skyflow.training.metrics import ConflictMetrics, LatencyTimer, bonferroni_ttest
from skyflow.utils.visualization import generate_all_figures

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Reproduce SkyFlow paper results")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--quick", action="store_true", help="Quick verification (~5 min on CPU)")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-scalability", action="store_true")
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
    logger.info(f"Seeds: {cfg.training.seeds[:cfg.training.num_seeds]}")
    logger.info("=" * 70)

    # ── Step 1: Generate Data ──
    logger.info("\n[Step 1/7] Generating UrbanAir-500 benchmark data...")
    sim = UrbanAir500(num_uavs=cfg.data.num_uavs, seed=cfg.training.seed)
    train_data = sim.generate_dataset("train", n_train_scenarios, scenario_duration, device)
    val_data = sim.generate_dataset("val", n_val_scenarios, scenario_duration, device)
    test_data = sim.generate_dataset("test", n_test_scenarios, scenario_duration, device)
    logger.info(f"  Train: {len(train_data)} snapshots | Val: {len(val_data)} | Test: {len(test_data)}")

    all_results = {}
    per_seed_cdrs = {}
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 2: Train TR-GAT (multi-seed) ──
    logger.info(f"\n[Step 2/7] Training TR-GAT ({cfg.training.num_seeds} seeds, {cfg.training.epochs} epochs)...")
    trainer = SkyFlowTrainer(cfg, device=device)
    trgat_summary = trainer.train_multi_seed(train_data, val_data, test_data)
    all_results["TR-GAT"] = {k: v["mean"] for k, v in trgat_summary.items()}
    all_results["TR-GAT"]["cdr_std"] = trgat_summary["cdr"]["std"]
    all_results["TR-GAT"]["far_std"] = trgat_summary["far"]["std"]
    all_results["TR-GAT"]["f1_std"] = trgat_summary["f1"]["std"]

    seed_results_path = output_dir / "multi_seed_results.json"
    if seed_results_path.exists():
        with open(seed_results_path) as f:
            seed_data = json.load(f)
        per_seed_cdrs["TR-GAT"] = [r["cdr"] for r in seed_data["seeds"]]
    logger.info(f"  TR-GAT: CDR={trgat_summary['cdr']['mean']:.4f}±{trgat_summary['cdr']['std']:.4f}")

    # ── Step 3: TR-GAT-NoTemp Ablation ──
    logger.info("\n[Step 3/7] TR-GAT-NoTemp ablation...")
    notp_cfg = SkyFlowConfig.from_yaml(args.config)
    if args.quick:
        notp_cfg.data.num_uavs = cfg.data.num_uavs
        notp_cfg.training.epochs = cfg.training.epochs
        notp_cfg.training.num_seeds = cfg.training.num_seeds
    notp_cfg.model.temporal_dim = 2
    notp_cfg.output_dir = str(output_dir / "ablation_notemp")

    notp_trainer = SkyFlowTrainer(notp_cfg, device=device)
    seeds = cfg.training.seeds[:cfg.training.num_seeds]

    notp_seeds_results = []
    for si, seed in enumerate(seeds):
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
    per_seed_cdrs["TR-GAT-NT"] = notp_cdr
    logger.info(f"  TR-GAT-NT: CDR={np.mean(notp_cdr):.4f}±{np.std(notp_cdr):.4f}")

    # ── Step 4: Baselines ──
    if not args.skip_baselines:
        logger.info("\n[Step 4/7] Training and evaluating baselines...")
        baseline_epochs = cfg.training.epochs

        # VO
        logger.info("  [4a] Velocity Obstacle (deterministic)...")
        vo = VelocityObstacle()
        vo_metrics = _eval_deterministic(vo, test_data, cfg, device)
        all_results["VO"] = vo_metrics
        per_seed_cdrs["VO"] = [vo_metrics["cdr"]] * cfg.training.num_seeds
        logger.info(f"    CDR={vo_metrics['cdr']:.4f}")

        baseline_models = [
            ("LSTM-P", LSTMPair(input_dim=cfg.data.uav_feature_dim)),
            ("Tfm-P", TransformerPair(input_dim=cfg.data.uav_feature_dim)),
            ("STGCN", STGCN(input_dim=cfg.data.uav_feature_dim)),
            ("GAT-S", GATStatic(input_dim=cfg.data.uav_feature_dim)),
        ]

        for bl_name, bl_model in baseline_models:
            logger.info(f"  [{bl_name}]...")
            bl_seed_cdrs = []
            bl_seed_results = []
            for si, seed in enumerate(seeds):
                torch.manual_seed(seed)
                np.random.seed(seed)
                bl_instance = type(bl_model)(input_dim=cfg.data.uav_feature_dim).to(device)
                _train_simple(bl_instance, train_data, cfg, device, baseline_epochs)
                m = _eval_nn(bl_instance, test_data, cfg, device)
                bl_seed_cdrs.append(m["cdr"])
                bl_seed_results.append(m)
            all_results[bl_name] = {
                "cdr": np.mean(bl_seed_cdrs),
                "cdr_std": np.std(bl_seed_cdrs),
                "far": np.mean([r["far"] for r in bl_seed_results]),
                "far_std": np.std([r["far"] for r in bl_seed_results]),
                "f1": np.mean([r["f1"] for r in bl_seed_results]),
                "f1_std": np.std([r["f1"] for r in bl_seed_results]),
                "precision": np.mean([r["precision"] for r in bl_seed_results]),
                "latency_ms": np.mean([r["latency_ms"] for r in bl_seed_results]),
            }
            per_seed_cdrs[bl_name] = bl_seed_cdrs
            logger.info(f"    CDR={np.mean(bl_seed_cdrs):.4f}±{np.std(bl_seed_cdrs):.4f}")
    else:
        logger.info("\n[Step 4/7] Skipping baselines (--skip-baselines)")

    # ── Step 5: Scalability Sweep ──
    if not args.skip_scalability:
        logger.info("\n[Step 5/7] Scalability analysis...")
        fleet_sizes = [100, 200, 300, 400, 500] if not args.quick else [50, 100]
        scalability = {}
        for n_uav in fleet_sizes:
            logger.info(f"  Fleet size: {n_uav}")
            scale_sim = UrbanAir500(num_uavs=n_uav, seed=cfg.training.seed)
            builder = TKGBuilder()
            graph_lats, fwd_lats, total_lats = [], [], []

            plans = scale_sim.generate_flight_plans(n_uav)
            for epoch_idx, (state, _) in enumerate(
                scale_sim.simulate_scenario(30.0, plans)
            ):
                if epoch_idx % 10 != 0:
                    continue
                t0 = time.perf_counter()
                snapshot = builder.build(state, device=device)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                graph_lats.append((t1 - t0) * 1000)

                with torch.no_grad():
                    t2 = time.perf_counter()
                    trainer.model.eval()
                    node_emb, _ = trainer.model(
                        snapshot.node_features.to(device),
                        {r: e.to(device) for r, e in snapshot.edge_indices.items()},
                        {r: d.to(device) for r, d in snapshot.edge_deltas.items()},
                    )
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t3 = time.perf_counter()
                    fwd_lats.append((t3 - t2) * 1000)
                total_lats.append((t3 - t0) * 1000)

            scalability[n_uav] = {
                "graph_ms": float(np.percentile(graph_lats, 95)) if graph_lats else 0,
                "fwd_ms": float(np.percentile(fwd_lats, 95)) if fwd_lats else 0,
                "total_ms": float(np.percentile(total_lats, 95)) if total_lats else 0,
            }
            logger.info(f"    Total p95: {scalability[n_uav]['total_ms']:.1f} ms")

        with open(output_dir / "scalability_results.json", "w") as f:
            json.dump(scalability, f, indent=2)
    else:
        logger.info("\n[Step 5/7] Skipping scalability (--skip-scalability)")

    # ── Step 6: Statistical Significance ──
    logger.info("\n[Step 6/7] Statistical significance tests...")
    if "TR-GAT" in per_seed_cdrs and len(per_seed_cdrs) > 1:
        baseline_cdrs = {k: v for k, v in per_seed_cdrs.items() if k != "TR-GAT"}
        sig_results = bonferroni_ttest(per_seed_cdrs["TR-GAT"], baseline_cdrs)
        with open(output_dir / "significance_tests.json", "w") as f:
            json.dump(sig_results, f, indent=2)
        logger.info(f"  {'Baseline':<12} {'ΔCDR':>8} {'t-stat':>8} {'p-value':>10} {'Sig?':>5}")
        logger.info("  " + "-" * 48)
        for name, sr in sig_results.items():
            logger.info(
                f"  {name:<12} {sr['delta_cdr']:>+8.4f} {sr['t_stat']:>8.2f} "
                f"{sr['p_value']:>10.4f} {'Yes' if sr['significant'] else 'No':>5}"
            )
    else:
        logger.info("  Insufficient data for significance tests")

    # ── Step 7: Generate Figures ──
    logger.info("\n[Step 7/7] Generating paper figures...")
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

    logger.info("\n" + "=" * 78)
    logger.info("  TABLE 3: Overall Detection Performance on UrbanAir-500 (mean ± std)")
    logger.info("=" * 78)
    logger.info(f"  {'Method':<12} {'CDR ↑':>14} {'FAR ↓':>14} {'F1 ↑':>14} {'Latency ↓':>12}")
    logger.info("  " + "-" * 72)
    method_order = ["VO", "LSTM-P", "Tfm-P", "STGCN", "GAT-S", "TR-GAT-NT", "TR-GAT"]
    for m in method_order:
        if m in all_results:
            r = all_results[m]
            cdr_s = f"{r['cdr']:.4f}"
            far_s = f"{r['far']:.4f}"
            f1_s = f"{r['f1']:.4f}"
            if r.get('cdr_std', 0) > 0:
                cdr_s += f"±{r['cdr_std']:.4f}"
                far_s += f"±{r['far_std']:.4f}"
                f1_s += f"±{r['f1_std']:.4f}"
            lat_s = f"{r.get('latency_ms', 0):.1f} ms"
            logger.info(f"  {m:<12} {cdr_s:>14} {far_s:>14} {f1_s:>14} {lat_s:>12}")
    logger.info("=" * 78)

    logger.info(f"\nAll outputs saved to: {output_dir.absolute()}/")
    logger.info("  all_results.json         — Table 3 metrics")
    logger.info("  multi_seed_results.json  — Per-seed breakdown")
    if not args.skip_scalability:
        logger.info("  scalability_results.json — Table 7 latency")
    logger.info("  significance_tests.json  — Table 5 Bonferroni t-tests")
    logger.info("  charts/                  — Publication-quality figures")


def _train_simple(model, data, cfg, device, epochs):
    criterion = FocalLoss(gamma=cfg.training.focal_gamma)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate,
                                   weight_decay=cfg.training.weight_decay)
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
