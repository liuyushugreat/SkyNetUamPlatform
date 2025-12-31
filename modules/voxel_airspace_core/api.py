"""
FastAPI Router for Voxel Airspace Core
======================================

Endpoints
---------
1) POST /build
   - Body: GeoJSON (FeatureCollection/Feature/Geometry)
   - Action: reset and rebuild octree occupancy
   - Returns: voxel statistics (including total_nodes / leaf_nodes)

2) POST /plan
   - Body: {"start": [x,y,z], "end": [x,y,z]}
   - Action: run voxel A* with octree occupancy checks
   - Returns: {"path": [[x,y,z], ...]}

3) GET /debug/voxels
   - Action: export all occupied leaf voxels as GeoJSON Point FeatureCollection
   - Use: Cesium / Kepler.gl visualization (point cloud)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from .manager import GLOBAL_MANAGER


router = APIRouter(prefix="/voxel", tags=["voxel-airspace-core"])


class BuildResponse(BaseModel):
    inserted_buildings: int
    total_nodes: int
    leaf_nodes: int
    occupied_nodes: int


class PlanRequest(BaseModel):
    start: List[float] = Field(..., description="Start point [x,y,z]")
    end: List[float] = Field(..., description="End point [x,y,z]")

    @field_validator("start", "end")
    @classmethod
    def _validate_len3(cls, v: List[float]) -> List[float]:
        if not isinstance(v, list) or len(v) != 3:
            raise ValueError("Point must be a list of three floats: [x, y, z].")
        return v


class PlanResponse(BaseModel):
    path: List[List[float]]


@router.post("/build", response_model=BuildResponse)
def build(geojson: Dict[str, Any]):
    """
    Reset and build the global octree from GeoJSON.

    Notes:
    - We accept raw GeoJSON dict to support large nested payloads.
    - Any geometry parsing failures are skipped per-feature (robust ingestion).
    """
    try:
        stats = GLOBAL_MANAGER.load_city_model(geojson)
        return BuildResponse(**stats)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to build octree: {e}")


@router.post("/plan", response_model=PlanResponse)
def plan(req: PlanRequest):
    """
    Plan a 3D voxel path using A*.

    Error handling:
    - start/end in obstacle or out of bounds => 400
    - model not built => 409 (conflict with required /build)
    """
    try:
        path = GLOBAL_MANAGER.find_path(
            start=(float(req.start[0]), float(req.start[1]), float(req.start[2])),
            end=(float(req.end[0]), float(req.end[1]), float(req.end[2])),
        )
        return PlanResponse(path=[[float(x), float(y), float(z)] for (x, y, z) in path])
    except ValueError as e:
        msg = str(e)
        if "has not been built" in msg:
            raise HTTPException(status_code=409, detail=msg)
        raise HTTPException(status_code=400, detail=msg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected planning error: {e}")


@router.get("/debug/voxels")
def debug_voxels():
    """
    Export occupied leaf voxels as a GeoJSON Point FeatureCollection.

    Each feature:
    - geometry: Point(x,y,z) voxel center
    - properties: {"size": voxel_edge_length}
    """
    try:
        return GLOBAL_MANAGER.export_occupied_voxels_geojson()
    except ValueError as e:
        msg = str(e)
        if "has not been built" in msg:
            raise HTTPException(status_code=409, detail=msg)
        raise HTTPException(status_code=400, detail=msg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected export error: {e}")


