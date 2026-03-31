"""
Voxel Airspace Core Module.

This module provides sparse octree indexing for efficient 3D airspace occupancy management.
"""

from .indexer import VoxelNode, SparseOctree
from .pathfinder import VoxelAStar

try:
    from .builder import VoxelBuilder, build_from_geojson
except ImportError:  # pragma: no cover
    # Keep the core octree/pathfinding package importable when optional geometry
    # dependencies such as `shapely` are unavailable in lightweight environments.
    VoxelBuilder = None  # type: ignore[assignment]
    build_from_geojson = None  # type: ignore[assignment]

__all__ = ['VoxelNode', 'SparseOctree', 'VoxelBuilder', 'build_from_geojson', 'VoxelAStar']

