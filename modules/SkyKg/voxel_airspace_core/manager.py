"""
Voxel Airspace Manager
======================

This module provides a stateful manager around:
- `SparseOctree` occupancy index (global, resettable)
- `VoxelBuilder` for ingesting GeoJSON city/building footprints
- `VoxelAStar` for 3D voxel path planning using `octree.query`

It is designed to be used by `api.py` (FastAPI router) but is also usable directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .indexer import SparseOctree
from .builder import VoxelBuilder
from .pathfinder import VoxelAStar


Point3 = Tuple[float, float, float]


@dataclass
class ManagerConfig:
    """Configuration for initializing/resetting the global octree and planner."""

    # Octree configuration
    root_size: float = 100_000.0
    max_depth: int = 10
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Builder configuration
    default_height: float = 50.0
    default_base_z: float = 0.0
    height_key: str = "height"
    base_z_key: str = "base_z"

    # Planner configuration
    resolution: float = 10.0
    heuristic: str = "euclidean"  # "euclidean" | "manhattan"
    safety_margin: float = 0.0
    safety_weight: float = 5.0
    max_expansions: int = 2_000_000


class VoxelAirspaceManager:
    """
    A stateful manager that holds a global `SparseOctree` and exposes high-level operations.

    Public Methods
    --------------
    - load_city_model(geojson): reset and build octree occupancy from GeoJSON
    - find_path(start, end): run A* planning on the current octree
    - export_occupied_voxels_geojson(): export occupied leaf voxels as GeoJSON point cloud
    """

    def __init__(self, config: Optional[ManagerConfig] = None) -> None:
        self.config = config or ManagerConfig()

        self.octree: SparseOctree = self._new_octree()
        self.builder: VoxelBuilder = self._new_builder()
        self._built: bool = False

    # -----------------
    # Construction
    # -----------------
    def _new_octree(self) -> SparseOctree:
        return SparseOctree(
            root_size=float(self.config.root_size),
            max_depth=int(self.config.max_depth),
            origin=tuple(self.config.origin),
        )

    def _new_builder(self) -> VoxelBuilder:
        return VoxelBuilder(
            default_height=float(self.config.default_height),
            default_base_z=float(self.config.default_base_z),
            height_key=str(self.config.height_key),
            base_z_key=str(self.config.base_z_key),
        )

    def reset(self) -> None:
        """Reset the global octree to an empty state."""
        self.octree = self._new_octree()
        self.builder = self._new_builder()
        self._built = False

    # -----------------
    # City model build
    # -----------------
    def load_city_model(self, geojson: Dict[str, Any]) -> Dict[str, int]:
        """
        Reset and build the octree occupancy from GeoJSON.

        Returns statistics for paper/debug:
        - total_nodes, leaf_nodes, occupied_nodes
        """
        self.reset()

        inserted = self.builder.build_from_geojson(geojson, self.octree)
        stats = self.octree.get_statistics()
        self._built = True

        # "体素总数" is often interpreted as leaf voxels, so we return both.
        return {
            "inserted_buildings": int(inserted),
            "total_nodes": int(stats.get("total_nodes", 0)),
            "leaf_nodes": int(stats.get("leaf_nodes", 0)),
            "occupied_nodes": int(stats.get("occupied_nodes", 0)),
        }

    # -----------------
    # Planning
    # -----------------
    def find_path(self, start: Point3, end: Point3) -> List[Point3]:
        """
        Plan a path from start to end using voxel A* with octree occupancy queries.

        Raises
        ------
        ValueError
            If start/end out of bounds or inside obstacles, or if model not built.
        """
        if not self._built:
            # In many APIs, planning without build is allowed (empty world). Here we
            # enforce a build step to prevent silent mistakes.
            raise ValueError("City model has not been built. Call load_city_model() first.")

        planner = VoxelAStar(
            octree=self.octree,
            resolution=float(self.config.resolution),
            heuristic=self.config.heuristic,  # type: ignore[arg-type]
            safety_margin=float(self.config.safety_margin),
            safety_weight=float(self.config.safety_weight),
            max_expansions=int(self.config.max_expansions),
        )
        return planner.find_path(start_point=start, end_point=end)

    # -----------------
    # Debug export
    # -----------------
    def export_occupied_voxels_geojson(self) -> Dict[str, Any]:
        """
        Export occupied leaf voxels as a GeoJSON FeatureCollection of Points.

        Each point represents the **center** of an occupied leaf voxel.
        `properties.size` records voxel edge length for visualization sizing.
        """
        if not self._built:
            raise ValueError("City model has not been built. Call load_city_model() first.")

        occupied = self.octree.get_occupied_voxels()
        features: List[Dict[str, Any]] = []

        for (x, y, z, size) in occupied:
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [float(x), float(y), float(z)]},
                    "properties": {"size": float(size)},
                }
            )

        return {"type": "FeatureCollection", "features": features}


# A single global manager instance for API usage.
GLOBAL_MANAGER = VoxelAirspaceManager()


