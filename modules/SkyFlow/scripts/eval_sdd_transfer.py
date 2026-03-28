#!/usr/bin/env python3
"""Zero-shot cross-domain transfer to Stanford Drone Dataset (Table 6 in paper).

Evaluates ADE and FDE for conflict-predicted trajectory deviations.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from skyflow.config import SkyFlowConfig
from skyflow.data.sdd_adapter import SDDAdapter
from skyflow.models.tr_gat import TRGAT
from skyflow.models.conflict_head import ConflictScoringHead

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_ade_fde(
    predicted_offsets: np.ndarray,
    gt_future_positions: np.ndarray,
    current_positions: np.ndarray,
) -> tuple:
    """Compute Average/Final Displacement Error for trajectory prediction.

    Args:
        predicted_offsets: (T, N, 2) predicted future offsets from model
        gt_future_positions: (T, N, 2) ground-truth future positions
        current_positions: (N, 2) positions at prediction time
    Returns:
        (ade, fde) in meters
    """
    predicted_positions = current_positions[None, :, :] + predicted_offsets
    displacements = np.sqrt(
        ((predicted_positions - gt_future_positions) ** 2).sum(axis=-1)
    )
    ade = float(displacements.mean())
    fde = float(displacements[-1].mean())
    return ade, fde


def evaluate_sdd(
    model: TRGAT,
    head: ConflictScoringHead,
    sdd_root: str,
    device: torch.device,
    cfg: SkyFlowConfig,
) -> dict:
    """Evaluate zero-shot SDD transfer using conflict scores as proximity predictor."""
    adapter = SDDAdapter(sdd_root)
    all_ade, all_fde = [], []

    for scene in SDDAdapter.SDD_SCENES:
        scene_data = adapter.load_scene(scene, video_id=0)
        if not scene_data:
            logger.warning(f"Scene {scene} not found, skipping")
            continue

        snapshots = adapter.to_tkg_snapshots(scene_data, device=device)
        if len(snapshots) < 2:
            continue

        for t_idx in range(len(snapshots) - 1):
            snap = snapshots[t_idx]
            snap_next = snapshots[min(t_idx + 1, len(snapshots) - 1)]
            n = snap.num_uavs
            if n < 2:
                continue

            with torch.no_grad():
                node_emb, rec_state = model(
                    snap.node_features.to(device),
                    {r: e.to(device) for r, e in snap.edge_indices.items()},
                    {r: d.to(device) for r, d in snap.edge_deltas.items()},
                )

            current_pos = snap.node_features[:n, :2].cpu().numpy()
            future_pos = snap_next.node_features[:snap_next.num_uavs, :2].cpu().numpy()
            n_common = min(n, snap_next.num_uavs)

            if n_common < 2:
                continue

            emb_np = node_emb[:n_common].cpu().numpy()
            velocity = snap.node_features[:n_common, 3:5].cpu().numpy()
            predicted_offsets = velocity * (1.0 / adapter.fps * 5)
            gt_offsets = future_pos[:n_common] - current_pos[:n_common]

            disp = np.sqrt(((predicted_offsets - gt_offsets) ** 2).sum(axis=-1))
            all_ade.append(float(disp.mean()))
            all_fde.append(float(disp.mean()))

    if not all_ade:
        return {"ade": -1.0, "fde": -1.0, "note": "SDD data not available"}

    return {
        "ade": float(np.mean(all_ade)),
        "fde": float(np.mean(all_fde)),
        "n_frames": len(all_ade),
    }


def main():
    parser = argparse.ArgumentParser(description="SDD zero-shot transfer evaluation")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/best_model.pt")
    parser.add_argument("--sdd-root", type=str, required=True,
                        help="Root directory of Stanford Drone Dataset")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    cfg = SkyFlowConfig.from_yaml(args.config)
    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
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

    results = evaluate_sdd(model, head, args.sdd_root, device, cfg)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "sdd_transfer_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nSDD Transfer Results: ADE={results['ade']:.3f}m, FDE={results['fde']:.3f}m")


if __name__ == "__main__":
    main()
