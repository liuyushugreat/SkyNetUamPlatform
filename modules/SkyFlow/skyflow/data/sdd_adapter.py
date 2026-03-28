"""Stanford Drone Dataset (SDD) adapter for cross-domain transfer evaluation.

Treats each pedestrian/cyclist as an entity and applies SkyFlow zero-shot,
zeroing out UAV-specific features to assess relational temporal reasoning
generalization across embodied interaction domains.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from skyflow.data.tkg_builder import AirspaceState, TKGBuilder, TKGSnapshot


class SDDAdapter:
    """Load and convert SDD annotations into TKG snapshots."""

    SDD_SCENES = [
        "bookstore", "coupa", "deathCircle", "gates",
        "hyang", "little", "nexus", "quad",
    ]

    def __init__(
        self,
        sdd_root: str | Path,
        feature_dim: int = 23,
        proximity_threshold: float = 3.0,
        fps: float = 30.0,
    ):
        self.sdd_root = Path(sdd_root)
        self.feature_dim = feature_dim
        self.proximity_threshold = proximity_threshold
        self.fps = fps

    def load_scene(
        self,
        scene_name: str,
        video_id: int = 0,
    ) -> List[Dict]:
        """Load annotations for a specific scene and video."""
        ann_path = self.sdd_root / scene_name / f"video{video_id}" / "annotations.txt"
        if not ann_path.exists():
            return []

        data = np.loadtxt(str(ann_path), delimiter=" ")
        frames = {}
        for row in data:
            track_id = int(row[0])
            x_min, y_min, x_max, y_max = row[1], row[2], row[3], row[4]
            frame = int(row[5])
            cx = (x_min + x_max) / 2.0
            cy = (y_min + y_max) / 2.0

            if frame not in frames:
                frames[frame] = []
            frames[frame].append({
                "track_id": track_id,
                "x": cx,
                "y": cy,
                "frame": frame,
            })

        return [
            {"frame": f, "agents": agents}
            for f, agents in sorted(frames.items())
        ]

    def to_tkg_snapshots(
        self,
        scene_data: List[Dict],
        device: torch.device = torch.device("cpu"),
        sample_rate: int = 5,
    ) -> List[TKGSnapshot]:
        """Convert SDD scene frames into TKG snapshots for zero-shot eval."""
        builder = TKGBuilder(
            approach_cpa_h=self.proximity_threshold * 10,
            approach_cpa_v=100.0,
        )
        snapshots = []

        prev_positions = {}

        for idx, frame_data in enumerate(scene_data):
            if idx % sample_rate != 0:
                continue

            agents = frame_data["agents"]
            n = len(agents)
            if n < 2:
                continue

            positions = np.zeros((n, 3), dtype=np.float32)
            velocities = np.zeros((n, 3), dtype=np.float32)

            for i, agent in enumerate(agents):
                positions[i] = [agent["x"], agent["y"], 0.0]
                tid = agent["track_id"]
                if tid in prev_positions:
                    dt = sample_rate / self.fps
                    velocities[i, :2] = (positions[i, :2] - prev_positions[tid]) / dt
                prev_positions[agent["track_id"]] = positions[i, :2].copy()

            t = frame_data["frame"] / self.fps

            state = AirspaceState(
                uav_positions=positions,
                uav_velocities=velocities,
                uav_headings=np.arctan2(velocities[:, 1], velocities[:, 0]),
                uav_battery=np.ones(n, dtype=np.float32),
                uav_priority=np.ones(n, dtype=np.int32),
                uav_avoiding=np.zeros(n, dtype=bool),
                sector_occupancy=np.zeros((1, 8), dtype=np.float32),
                weather_cells=np.zeros((0, 12), dtype=np.float32),
                restricted_zones=np.zeros((0, 8), dtype=np.float32),
                corridor_reservations=[],
                epoch_time=t,
            )

            snapshot = builder.build(state, device=device)
            snapshots.append(snapshot)

        return snapshots
