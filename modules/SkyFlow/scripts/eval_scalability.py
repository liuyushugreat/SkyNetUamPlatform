#!/usr/bin/env python3
"""Scalability analysis: latency vs. fleet size (Table 7 in the paper).

Sweeps fleet sizes [100, 200, 300, 400, 500] and reports 95th-pctl
latency breakdown: graph construction vs. TR-GAT forward pass.
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
from skyflow.training.metrics import LatencyTimer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FLEET_SIZES = [100, 200, 300, 400, 500]


def main():
    parser = argparse.ArgumentParser(description="SkyFlow scalability evaluation")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/best_model.pt")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--n-epochs", type=int, default=1000,
                        help="Number of test epochs per fleet size")
    parser.add_argument("--scenario-duration", type=float, default=60.0)
    args = parser.parse_args()

    cfg = SkyFlowConfig.from_yaml(args.config)
    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    ckpt_path = Path(args.checkpoint)
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
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
    else:
        logger.warning(f"No checkpoint at {ckpt_path}, using random weights")
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

    model.eval()
    head.eval()

    results = {}
    for n_uav in FLEET_SIZES:
        logger.info(f"\nEvaluating fleet size: {n_uav} UAVs")
        sim = UrbanAir500(num_uavs=n_uav, seed=cfg.training.seed)
        builder = TKGBuilder()

        graph_latencies = []
        fwd_latencies = []
        total_latencies = []
        n_measured = 0

        for scenario_idx in range(max(1, args.n_epochs // 50)):
            builder.reset()
            plans = sim.generate_flight_plans(n_uav)
            for epoch_idx, (state, conflicts) in enumerate(
                sim.simulate_scenario(args.scenario_duration, plans)
            ):
                if epoch_idx % 10 != 0:
                    continue

                t_total_start = time.perf_counter()

                t_graph_start = time.perf_counter()
                snapshot = builder.build(state, device=device)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_graph_end = time.perf_counter()
                graph_ms = (t_graph_end - t_graph_start) * 1000.0

                with torch.no_grad():
                    t_fwd_start = time.perf_counter()
                    node_emb, rec_state = model(
                        snapshot.node_features.to(device),
                        {r: e.to(device) for r, e in snapshot.edge_indices.items()},
                        {r: d.to(device) for r, d in snapshot.edge_deltas.items()},
                    )
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t_fwd_end = time.perf_counter()
                    fwd_ms = (t_fwd_end - t_fwd_start) * 1000.0

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                total_ms = (time.perf_counter() - t_total_start) * 1000.0

                graph_latencies.append(graph_ms)
                fwd_latencies.append(fwd_ms)
                total_latencies.append(total_ms)
                n_measured += 1

                if n_measured >= args.n_epochs:
                    break
            if n_measured >= args.n_epochs:
                break

        graph_p95 = float(np.percentile(graph_latencies, 95))
        fwd_p95 = float(np.percentile(fwd_latencies, 95))
        total_p95 = float(np.percentile(total_latencies, 95))

        results[n_uav] = {
            "graph_const_ms": graph_p95,
            "trgat_fwd_ms": fwd_p95,
            "total_ms": total_p95,
            "n_samples": n_measured,
        }
        logger.info(f"  Graph: {graph_p95:.1f} ms | TR-GAT: {fwd_p95:.1f} ms | Total: {total_p95:.1f} ms")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "scalability_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info(f"{'UAVs':<8} {'Graph (ms)':>12} {'TR-GAT (ms)':>12} {'Total (ms)':>12}")
    logger.info("-" * 48)
    for n in FLEET_SIZES:
        if n in results:
            r = results[n]
            logger.info(f"{n:<8} {r['graph_const_ms']:>12.1f} {r['trgat_fwd_ms']:>12.1f} {r['total_ms']:>12.1f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
