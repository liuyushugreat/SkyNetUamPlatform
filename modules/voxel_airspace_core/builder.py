"""
Voxel Builder: Convert GeoJSON building footprints into 3D occupancy for SparseOctree.

This module focuses on **fast, paper-friendly** occupancy construction:
- Use `shapely` to parse 2D building geometries (Polygon/MultiPolygon).
- Avoid per-point checks: compute a tight **2D bounding box** from the polygon and
  extrude it with a scalar height into a **3D axis-aligned bounding box (AABB)**.
- Insert only AABBs into the octree (`octree_instance.insert(bbox)`), which triggers
  recursive subdivision only where necessary.

Why Bounding-Box-first?
-----------------------
For large city-scale maps, per-point rasterization is expensive. AABB insertion provides:
- O(1) geometry preprocessing per building (bounds extraction)
- octree-side pruning via AABB intersection tests
- a clean theoretical story for papers: "coarse-to-fine sparse occupancy marking"

Important Note (Accuracy vs. Speed):
------------------------------------
This builder marks occupancy using **extruded AABBs**, not exact polygon prisms.
That is conservative (may over-mark corners). If you later need tighter occupancy,
you can add an optional refinement stage (e.g., subdivide to max depth and test voxel
centers against the polygon) — but this file intentionally implements the fast path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import numpy as np

try:
    from shapely.geometry import shape
    from shapely.geometry.base import BaseGeometry
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "shapely is required for modules.voxel_airspace_core.builder. "
        "Install it via: pip install shapely"
    ) from e


GeoJSON = Dict[str, Any]
BBox = Tuple[float, float, float, float, float, float]


@dataclass
class VoxelBuilder:
    """
    Build octree occupancy from vector map data (GeoJSON building footprints).

    Parameters
    ----------
    default_height:
        Height (meters) used when a feature lacks `properties['height']`.
    default_base_z:
        Base altitude (meters) used when a feature lacks `properties['base_z']`
        (or equivalent). Many GeoJSON building datasets assume ground at z=0.
    height_key:
        Property key to read building height from. Default: "height".
    base_z_key:
        Property key to read base altitude from. Default: "base_z".
    """

    default_height: float = 50.0
    default_base_z: float = 0.0
    height_key: str = "height"
    base_z_key: str = "base_z"

    def build_from_geojson(self, geojson_data: GeoJSON, octree_instance: Any) -> int:
        """
        Iterate GeoJSON Polygon features and mark the corresponding 3D space as occupied.

        This method performs a **Bounding-Box First** strategy:
        1) Convert feature geometry to a shapely object
        2) Compute 2D bounds: (min_x, min_y, max_x, max_y)
        3) Extrude with (base_z, base_z + height) to form 3D AABB
        4) Call `octree_instance.insert((min_x, min_y, min_z, max_x, max_y, max_z))`

        Parameters
        ----------
        geojson_data:
            A GeoJSON-like dict. Supported:
            - FeatureCollection: {"type": "FeatureCollection", "features": [...]}
            - Single Feature: {"type": "Feature", "geometry": {...}, "properties": {...}}
        octree_instance:
            An instance compatible with `SparseOctree`, i.e., provides `insert(bbox)`.

        Returns
        -------
        int
            Number of building features successfully inserted.
        """
        features = _iter_geojson_features(geojson_data)
        inserted = 0

        for feat in features:
            geom_obj = _parse_geometry(feat.get("geometry"))
            if geom_obj is None:
                continue

            # Accept Polygon / MultiPolygon; other types are ignored.
            if geom_obj.geom_type not in ("Polygon", "MultiPolygon"):
                continue

            props = feat.get("properties") or {}
            height = _safe_float(props.get(self.height_key), default=self.default_height)
            base_z = _safe_float(props.get(self.base_z_key), default=self.default_base_z)
            if height < 0:
                # Defensive: negative height makes no physical sense; fall back to default.
                height = float(self.default_height)

            min_x, min_y, max_x, max_y = geom_obj.bounds  # shapely bounds are fast
            min_z = base_z
            max_z = base_z + height

            bbox: BBox = (min_x, min_y, min_z, max_x, max_y, max_z)

            # Fast-path insertion: the octree handles pruning and subdivision.
            octree_instance.insert(np.asarray(bbox, dtype=np.float64))
            inserted += 1

        return inserted


def build_from_geojson(geojson_data: GeoJSON, octree_instance: Any, *, default_height: float = 50.0) -> int:
    """
    Convenience function (as requested) for building occupancy without instantiating VoxelBuilder.

    Parameters
    ----------
    geojson_data:
        GeoJSON dict.
    octree_instance:
        SparseOctree-like instance (must provide `.insert(bbox)`).
    default_height:
        Default height if feature lacks height attribute.
    """
    return VoxelBuilder(default_height=default_height).build_from_geojson(geojson_data, octree_instance)


def _iter_geojson_features(geojson_data: GeoJSON) -> Iterable[Dict[str, Any]]:
    """Yield GeoJSON Feature dicts from FeatureCollection or single Feature."""
    if not isinstance(geojson_data, dict):
        return []

    t = geojson_data.get("type")
    if t == "FeatureCollection":
        return geojson_data.get("features") or []
    if t == "Feature":
        return [geojson_data]

    # Sometimes users pass raw geometry; wrap it as a single feature.
    if "coordinates" in geojson_data and "type" in geojson_data:
        return [{"type": "Feature", "geometry": geojson_data, "properties": {}}]

    return []


def _parse_geometry(geometry: Optional[Dict[str, Any]]) -> Optional["BaseGeometry"]:
    """
    Parse GeoJSON geometry dict to shapely geometry.

    Includes a common robustness trick: `buffer(0)` to fix minor self-intersections.
    """
    if not geometry or not isinstance(geometry, dict):
        return None

    try:
        g = shape(geometry)
    except Exception:
        return None

    # Fix invalid geometries if possible (optional but helpful for real-world GeoJSON).
    try:
        if not g.is_valid:
            g = g.buffer(0)
    except Exception:
        pass

    return g


def _safe_float(value: Any, default: float) -> float:
    """Convert value to float safely; return default on failure."""
    if value is None:
        return float(default)
    try:
        return float(value)
    except Exception:
        return float(default)


