"""
Voxel-based 3D A* Pathfinder using SparseOctree occupancy queries.

Design Goals
------------
1) **Voxel graph search**: A* over a regular 3D voxel grid (6-connectivity).
2) **Sparse collision checks**: Do NOT build a dense occupancy grid. Instead, check
   collisions online via `octree.query(point)`.
3) **Paper-friendly explanation**: Detailed comments to describe how voxels, neighbors,
   and costs are defined.
4) **Safety margin**: Penalize paths that move close to obstacles (soft constraint),
   without rejecting them outright unless the cell is occupied.

Important Modeling Choice: Regular Grid vs. Adaptive Octree Leaves
------------------------------------------------------------------
`SparseOctree` is adaptive (variable voxel sizes), but path planning on an adaptive
mesh is significantly more complex (requires balancing, neighbor finding across levels).

For this IJCAI-friendly baseline, we plan on a **fixed-resolution voxel lattice**
with step size `resolution`, and use the octree ONLY as an efficient occupancy oracle:

    is_occupied(p) := octree.query(p)

This is a standard and defensible approach in robotics papers: an adaptive index is
used to accelerate queries, while planning uses a regular lattice for simplicity.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np


Point3 = Tuple[float, float, float]
Index3 = Tuple[int, int, int]
HeuristicType = Literal["manhattan", "euclidean"]


@dataclass(frozen=True)
class _NodeRecord:
    """Record used for priority queue ordering."""
    f: float
    g: float
    idx: Index3


class VoxelAStar:
    """
    3D A* pathfinder on a voxel grid with 6-connectivity.

    Parameters
    ----------
    octree:
        Instance of `SparseOctree` (or compatible) providing:
        - `root` with `.get_bounds() -> (min_x, min_y, min_z, max_x, max_y, max_z)`
        - `query(point) -> bool` to test occupancy at a point.
    resolution:
        Voxel edge length for the planning grid (meters). Neighbor moves are exactly
        +/- resolution along one axis.
    heuristic:
        Either "manhattan" or "euclidean" distance between voxel centers.
    safety_margin:
        Distance (meters). If a candidate cell is within `safety_margin` of an occupied
        cell, we **add an extra cost** (penalty) to discourage hugging obstacles.
        This is implemented via neighborhood sampling using `octree.query`.
    safety_weight:
        Multiplier for the safety penalty. Higher => stronger avoidance.
    max_expansions:
        Upper bound on node expansions to prevent runaway searches.
    """

    def __init__(
        self,
        octree,
        resolution: float = 10.0,
        heuristic: HeuristicType = "euclidean",
        safety_margin: float = 0.0,
        safety_weight: float = 5.0,
        max_expansions: int = 2_000_000,
    ) -> None:
        if resolution <= 0:
            raise ValueError("resolution must be positive.")
        if safety_margin < 0:
            raise ValueError("safety_margin must be >= 0.")
        if safety_weight < 0:
            raise ValueError("safety_weight must be >= 0.")

        self.octree = octree
        self.resolution = float(resolution)
        self.heuristic = heuristic
        self.safety_margin = float(safety_margin)
        self.safety_weight = float(safety_weight)
        self.max_expansions = int(max_expansions)

        # Planning bounds: we restrict nodes to the root cube of the octree.
        self.bounds = self._get_bounds()
        self._min_bound = np.array(self.bounds[:3], dtype=np.float64)
        self._max_bound = np.array(self.bounds[3:], dtype=np.float64)

    def _get_bounds(self) -> Tuple[float, float, float, float, float, float]:
        if not hasattr(self.octree, "root") or not hasattr(self.octree.root, "get_bounds"):
            raise AttributeError("octree must have root.get_bounds()")
        return tuple(self.octree.root.get_bounds())

    # -----------------------------
    # Coordinate <-> Grid mapping
    # -----------------------------
    def _point_to_index(self, p: Point3) -> Index3:
        """
        Map a world coordinate to the nearest voxel index.

        We anchor the regular lattice at the octree root min bound. Index (0,0,0)
        corresponds to center at (min + 0.5*res, min + 0.5*res, min + 0.5*res).
        """
        p_arr = np.array(p, dtype=np.float64)
        # Convert to continuous cell coordinates, then floor to get cell index.
        rel = (p_arr - self._min_bound) / self.resolution
        i, j, k = np.floor(rel).astype(int)
        return int(i), int(j), int(k)

    def _index_to_point(self, idx: Index3) -> Point3:
        """Convert a voxel index to the voxel center world coordinate."""
        idx_arr = np.array(idx, dtype=np.float64)
        center = self._min_bound + (idx_arr + 0.5) * self.resolution
        return float(center[0]), float(center[1]), float(center[2])

    def _in_bounds_index(self, idx: Index3) -> bool:
        """Check whether the voxel center is within octree bounds."""
        x, y, z = self._index_to_point(idx)
        min_x, min_y, min_z, max_x, max_y, max_z = self.bounds
        return (min_x <= x <= max_x) and (min_y <= y <= max_y) and (min_z <= z <= max_z)

    # -----------------------------
    # Heuristic
    # -----------------------------
    def _h(self, a: Index3, b: Index3) -> float:
        """Heuristic distance between voxel centers."""
        ax, ay, az = self._index_to_point(a)
        bx, by, bz = self._index_to_point(b)
        dx, dy, dz = abs(ax - bx), abs(ay - by), abs(az - bz)

        if self.heuristic == "manhattan":
            return dx + dy + dz
        # Default: euclidean
        return float(np.sqrt(dx * dx + dy * dy + dz * dz))

    # -----------------------------
    # Occupancy and safety penalty
    # -----------------------------
    def _is_occupied(self, idx: Index3) -> bool:
        """A voxel is considered occupied if its center is occupied in the octree."""
        p = self._index_to_point(idx)
        return bool(self.octree.query(p))

    def _safety_penalty(self, idx: Index3) -> float:
        """
        Soft penalty if the voxel is close to obstacles.

        Definition (practical approximation):
        - Let r = safety_margin.
        - Sample a small neighborhood around the voxel center at offsets of
          {0, ±resolution, ±2*resolution, ...} up to r along each axis.
        - If any sampled point is occupied, we consider this voxel "near obstacles".
        - Penalty increases with the number of occupied samples.

        Why sampling?
        - The octree only provides point queries; exact distance-to-obstacle would require
          additional geometry or a signed distance field (SDF).
        - For planning, a coarse near-obstacle penalty is often sufficient and cheap.
        """
        if self.safety_margin <= 0.0 or self.safety_weight <= 0.0:
            return 0.0

        center = np.array(self._index_to_point(idx), dtype=np.float64)
        step = self.resolution
        r = self.safety_margin

        # Number of lattice steps to cover safety margin.
        n = int(np.ceil(r / step))
        if n <= 0:
            return 0.0

        # Sample a 3D cross + diagonals could be expensive; we keep it moderate:
        # sample the 6 axis directions at 1..n steps (a "star" neighborhood).
        occupied_hits = 0
        total_samples = 0

        for s in range(1, n + 1):
            d = s * step
            offsets = np.array(
                [
                    [d, 0.0, 0.0],
                    [-d, 0.0, 0.0],
                    [0.0, d, 0.0],
                    [0.0, -d, 0.0],
                    [0.0, 0.0, d],
                    [0.0, 0.0, -d],
                ],
                dtype=np.float64,
            )
            pts = center[None, :] + offsets
            # Bounds check for each sample point; ignore out-of-bounds.
            for pt in pts:
                total_samples += 1
                if np.any(pt < self._min_bound) or np.any(pt > self._max_bound):
                    continue
                if self.octree.query((float(pt[0]), float(pt[1]), float(pt[2]))):
                    occupied_hits += 1

        if total_samples == 0:
            return 0.0

        # Normalize by samples and scale by weight. Also scale by step size so the
        # penalty remains meaningful across different resolutions.
        ratio = occupied_hits / float(total_samples)
        return self.safety_weight * ratio * step

    # -----------------------------
    # Neighbor generation
    # -----------------------------
    def _neighbors6(self, idx: Index3) -> List[Index3]:
        """
        6-connectivity neighbors (Von Neumann neighborhood) on the voxel lattice:
        +/- x, +/- y, +/- z.
        """
        i, j, k = idx
        candidates = [
            (i + 1, j, k),
            (i - 1, j, k),
            (i, j + 1, k),
            (i, j - 1, k),
            (i, j, k + 1),
            (i, j, k - 1),
        ]
        return [c for c in candidates if self._in_bounds_index(c)]

    # -----------------------------
    # Public API
    # -----------------------------
    def find_path(self, start_point: Point3, end_point: Point3) -> List[Point3]:
        """
        Run A* and return a list of voxel-center coordinates representing the path.

        Steps:
        ------
        1) Map start/end world points to voxel indices.
        2) A* over indices:
           - Move cost: resolution (grid edge length)
           - + safety penalty for neighbor nodes near obstacles
           - Skip occupied neighbors (octree.query == True)
        3) Reconstruct the path and return world coordinates (centers).
        """
        start_idx = self._point_to_index(start_point)
        goal_idx = self._point_to_index(end_point)

        if not self._in_bounds_index(start_idx):
            raise ValueError("start_point is out of octree bounds.")
        if not self._in_bounds_index(goal_idx):
            raise ValueError("end_point is out of octree bounds.")

        if self._is_occupied(start_idx):
            raise ValueError("start_point is inside an occupied voxel.")
        if self._is_occupied(goal_idx):
            raise ValueError("end_point is inside an occupied voxel.")

        # A* data structures
        open_heap: List[Tuple[float, float, Index3]] = []
        heapq.heappush(open_heap, (self._h(start_idx, goal_idx), 0.0, start_idx))

        came_from: Dict[Index3, Index3] = {}
        g_score: Dict[Index3, float] = {start_idx: 0.0}
        closed: set[Index3] = set()

        expansions = 0

        while open_heap:
            f, g, current = heapq.heappop(open_heap)
            if current in closed:
                continue

            closed.add(current)
            expansions += 1
            if expansions > self.max_expansions:
                # Return empty path to indicate failure under resource limit.
                return []

            if current == goal_idx:
                return self._reconstruct_path(came_from, current)

            for nb in self._neighbors6(current):
                if nb in closed:
                    continue

                # Hard constraint: cannot step into occupied voxel.
                if self._is_occupied(nb):
                    continue

                # Base move cost: one grid step.
                tentative_g = g_score[current] + self.resolution

                # Soft constraint: add penalty if neighbor is close to obstacles.
                tentative_g += self._safety_penalty(nb)

                if tentative_g < g_score.get(nb, float("inf")):
                    came_from[nb] = current
                    g_score[nb] = tentative_g
                    f_score = tentative_g + self._h(nb, goal_idx)
                    heapq.heappush(open_heap, (f_score, tentative_g, nb))

        # No path found
        return []

    def _reconstruct_path(self, came_from: Dict[Index3, Index3], current: Index3) -> List[Point3]:
        """Reconstruct path from came_from map, returning world coordinates (voxel centers)."""
        idx_path = [current]
        while current in came_from:
            current = came_from[current]
            idx_path.append(current)
        idx_path.reverse()
        return [self._index_to_point(idx) for idx in idx_path]


