
"""
Density-Based Adaptive Octree (DBAO)
===================================

This module implements a **Density-Based Adaptive Octree (DBAO)** for 3D UAM
airspace management. Unlike a static octree index, DBAO dynamically **splits**
high-density regions (zoom-in) and **merges** low-density regions (zoom-out)
based on real-time traffic density (e.g., drones, temporary obstacles).

The design favors:
- **Deterministic spatial partitioning** (axis-aligned cubic bounds)
- **Density-driven refinement** (Split/Merge thresholds + hysteresis)
- **Thread-safety mindset** for high-concurrency updates (RLock as coarse-grain guard)

Only drone position update is modeled here; other dynamic objects can be added
by reusing the same insertion/removal API patterns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import threading
import time
from typing import Dict, Iterable, List, Optional, Set, Tuple


Vec3 = Tuple[float, float, float]


@dataclass(slots=True)
class AdaptiveOctreeNode:
    """
    A single node in a Density-Based Adaptive Octree (DBAO).

    Bounds representation:
        The node stores an **axis-aligned cube** using `(x, y, z, size)` where:
        - (x, y, z) is the **minimum corner** of the cube
        - size is the cube edge length (must be > 0)
    """

    x: float
    y: float
    z: float
    size: float
    depth: int
    max_depth: int

    # Dynamic density metrics
    traffic_density: int = 0
    last_update_time: float = 0.0

    # Content of this node (only tracked for leaves in this implementation)
    drone_ids: Set[str] = field(default_factory=set)

    # Children: None means leaf node; otherwise 8 children exist.
    children: Optional[List["AdaptiveOctreeNode"]] = None

    def is_leaf(self) -> bool:
        return self.children is None

    def bounds_tuple(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.z, self.size)

    def contains_point(self, x: float, y: float, z: float) -> bool:
        """Return True iff the point is inside this cube (inclusive min, exclusive max)."""
        return (
            self.x <= x < self.x + self.size
            and self.y <= y < self.y + self.size
            and self.z <= z < self.z + self.size
        )

    def _child_index_for_point(self, x: float, y: float, z: float) -> int:
        """
        Compute child octant index for a point.

        Index bit layout:
        - bit0: x high half
        - bit1: y high half
        - bit2: z high half
        """
        hx = self.x + self.size / 2.0
        hy = self.y + self.size / 2.0
        hz = self.z + self.size / 2.0
        ix = 1 if x >= hx else 0
        iy = 1 if y >= hy else 0
        iz = 1 if z >= hz else 0
        return ix | (iy << 1) | (iz << 2)

    def _make_children(self) -> List["AdaptiveOctreeNode"]:
        half = self.size / 2.0
        children: List[AdaptiveOctreeNode] = []
        for iz in (0, 1):
            for iy in (0, 1):
                for ix in (0, 1):
                    cx = self.x + ix * half
                    cy = self.y + iy * half
                    cz = self.z + iz * half
                    children.append(
                        AdaptiveOctreeNode(
                            x=cx,
                            y=cy,
                            z=cz,
                            size=half,
                            depth=self.depth + 1,
                            max_depth=self.max_depth,
                            traffic_density=0,
                            last_update_time=self.last_update_time,
                        )
                    )
        # Order produced by loops is: (ix changes fastest) -> matches index bits described
        return children

    def subdivide(self, positions: Dict[str, Vec3]) -> None:
        r"""
        Split Mechanism (Zoom-In): Subdivide this leaf node into 8 children.

        Mathematical / algorithmic logic (for direct citation):
        -----------------------------------------------------
        Let a node \(n\) be a **leaf** node at depth \(d\) with dynamic density
        \(\rho(n)\) representing the number of targets within its bounds.

        **Trigger condition (manager-side check):**
        \[
          \rho(n) > T_{split} \quad \wedge \quad d < D_{max}
        \]
        where \(T_{split}\) is `SPLIT_THRESHOLD` and \(D_{max}\) is `MAX_DEPTH`.

        **Behavior:**
        - Convert \(n\) from leaf to internal node by creating 8 children.
        - **Redistribute** all existing targets in \(n\) to its children based on
          each target's current position.

        **Purpose:**
        Refinement yields higher spatial resolution in dense regions, enabling
        finer collision detection / planning. For example, an area previously
        represented at 100m resolution can be refined to 50m/25m/... depending
        on recursive splits.

        Concurrency / atomicity considerations:
        --------------------------------------
        In a high-concurrency environment, subdivision must appear atomic to
        readers: a node should not be observed in a partially-split state.
        This implementation assumes **external coarse-grain locking** (e.g.,
        `AdaptiveOctreeManager._lock`) to protect:
        - creation of `children`
        - redistribution of `drone_ids` and `traffic_density`
        """
        if not self.is_leaf():
            return
        if self.depth >= self.max_depth:
            return
        if self.size <= 0:
            raise ValueError("Node size must be > 0")

        # Create children first (so we can redistribute deterministically)
        children = self._make_children()

        # Redistribute existing drones into children
        for drone_id in list(self.drone_ids):
            pos = positions.get(drone_id)
            if pos is None:
                # Unknown position: keep it at this node (defensive).
                continue
            px, py, pz = pos
            if not self.contains_point(px, py, pz):
                # Out-of-bounds: also keep here (defensive).
                continue
            idx = self._child_index_for_point(px, py, pz)
            child = children[idx]
            child.drone_ids.add(drone_id)
            child.traffic_density += 1
            child.last_update_time = max(child.last_update_time, self.last_update_time)

        # Finalize: become internal node; clear leaf storage to avoid duplication.
        self.children = children
        self.drone_ids.clear()
        # For internal nodes we define density as sum of children densities.
        self.traffic_density = sum(c.traffic_density for c in children)

    def merge(self) -> None:
        r"""
        Merge Mechanism (Zoom-Out): Remove children and convert back to leaf.

        Mathematical / algorithmic logic (for direct citation):
        -----------------------------------------------------
        Consider an internal node \(p\) with 8 children \(\{c_i\}_{i=1}^{8}\).
        Each child has dynamic density \(\rho(c_i)\). The parent density is:
        \[
          \rho(p) = \sum_{i=1}^{8} \rho(c_i)
        \]

        **Trigger condition (manager-side check):**
        \[
          \sum_{i=1}^{8} \rho(c_i) < T_{merge}
        \]
        where \(T_{merge}\) is `MERGE_THRESHOLD`.

        **Hysteresis requirement:**
        To avoid flickering around thresholds, enforce:
        \[
          T_{merge} < T_{split}
        \]
        This creates a dead-band so the structure does not oscillate
        split/merge when density fluctuates near a single boundary.

        **Behavior:**
        - Destroy all children nodes and reclaim memory.
        - Reset \(p\) to a leaf node; optionally aggregate remaining targets.

        Concurrency / atomicity considerations:
        --------------------------------------
        Merge must also appear atomic. This implementation assumes external
        coarse-grain locking to prevent observing a partially-merged subtree.
        """
        if self.is_leaf():
            return

        # Aggregate drones from children into this node as a leaf
        drone_ids: Set[str] = set()
        last_ts = self.last_update_time
        total_density = 0
        assert self.children is not None
        for c in self.children:
            drone_ids |= c.drone_ids
            total_density += c.traffic_density
            last_ts = max(last_ts, c.last_update_time)

        self.children = None
        self.drone_ids = drone_ids
        self.traffic_density = total_density
        self.last_update_time = last_ts

    def _walk_postorder(self) -> Iterable["AdaptiveOctreeNode"]:
        """Yield nodes in post-order (children before parent)."""
        if self.children:
            for c in self.children:
                yield from c._walk_postorder()
        yield self


class AdaptiveOctreeManager:
    """
    Manages a DBAO octree and exposes public APIs for dynamic updates.

    Thread safety:
        This manager uses a coarse-grain `threading.RLock` to make each public
        API call (update/prune) appear atomic from the perspective of other
        callers. This is a pragmatic concurrency strategy for correctness;
        if higher throughput is required, the lock can be refined to node-level
        locks (with careful deadlock avoidance).
    """

    def __init__(
        self,
        *,
        root_bounds: Tuple[float, float, float, float],
        split_threshold: int = 5,
        merge_threshold: int = 2,
        max_depth: int = 6,
    ) -> None:
        if merge_threshold >= split_threshold:
            raise ValueError(
                "Hysteresis violated: merge_threshold must be < split_threshold"
            )
        if max_depth < 0:
            raise ValueError("max_depth must be >= 0")

        rx, ry, rz, rsize = root_bounds
        if rsize <= 0:
            raise ValueError("root size must be > 0")

        self.split_threshold = int(split_threshold)
        self.merge_threshold = int(merge_threshold)
        self.max_depth = int(max_depth)

        self.root = AdaptiveOctreeNode(
            x=rx,
            y=ry,
            z=rz,
            size=rsize,
            depth=0,
            max_depth=self.max_depth,
            traffic_density=0,
            last_update_time=time.time(),
        )

        # Drone state
        self._positions: Dict[str, Vec3] = {}
        self._drone_leaf: Dict[str, AdaptiveOctreeNode] = {}

        # Coarse-grain lock for atomic-like operations
        self._lock = threading.RLock()

    def update_drone_position(self, drone_id: str, x: float, y: float, z: float) -> None:
        """
        Update a drone position and trigger adaptive split operations.

        Contract:
        - Inserts new drone if unseen.
        - Moves drone between leaves if it crosses node boundaries.
        - May trigger one or more splits along the descent path.
        """
        now = time.time()
        with self._lock:
            # Update stored position first (so redistribute can see it)
            self._positions[drone_id] = (x, y, z)

            # If previously placed, remove it from its leaf density
            old_leaf = self._drone_leaf.get(drone_id)
            if old_leaf is not None and drone_id in old_leaf.drone_ids:
                old_leaf.drone_ids.remove(drone_id)
                # Keep density consistent with the actual content set.
                old_leaf.traffic_density = len(old_leaf.drone_ids)
                old_leaf.last_update_time = now

            # Insert into (possibly newly-created) leaf node
            leaf = self._insert_into_tree(drone_id, x, y, z, now)
            self._drone_leaf[drone_id] = leaf

    def prune_tree(self) -> int:
        """
        Periodically called to perform bottom-up merge checks.

        Returns:
            Number of merge operations performed.
        """
        merges = 0
        with self._lock:
            # Post-order ensures we merge children before considering parents.
            for node in self.root._walk_postorder():
                if node.is_leaf():
                    continue
                assert node.children is not None
                # Only merge when all children are leaves; otherwise prune deeper first.
                if any(not c.is_leaf() for c in node.children):
                    continue
                total_density = sum(c.traffic_density for c in node.children)
                if total_density < self.merge_threshold:
                    # Merge and refresh drone->leaf mapping for drones now stored at parent
                    node.merge()
                    merges += 1
                    for did in node.drone_ids:
                        self._drone_leaf[did] = node
        return merges

    def check_collision_point(self, x: float, y: float, z: float) -> bool:
        """
        Query if a point (x, y, z) falls into any occupied voxel.
        Complexity: O(log N) where N is tree depth.

        Returns:
            True if the leaf node containing the point has density > 0.
        """
        # No lock needed for read-only traversal if we accept eventual consistency,
        # but RLock is reentrant so we can use it for safety.
        with self._lock:
            if not self.root.contains_point(x, y, z):
                return False

            node = self.root
            while True:
                if node.is_leaf():
                    return node.traffic_density > 0
                
                # Internal node: descend
                assert node.children is not None
                idx = node._child_index_for_point(x, y, z)
                node = node.children[idx]

    # -------------------------
    # Internal helpers
    # -------------------------

    def _insert_into_tree(
        self, drone_id: str, x: float, y: float, z: float, now: float
    ) -> AdaptiveOctreeNode:
        """
        Descend from root to a leaf, splitting as needed, then place drone there.
        """
        if not self.root.contains_point(x, y, z):
            # Out-of-root bounds: keep it at root as a fallback (caller may choose to expand root).
            self.root.drone_ids.add(drone_id)
            self.root.traffic_density = len(self.root.drone_ids)
            self.root.last_update_time = now
            return self.root

        node = self.root
        node.last_update_time = max(node.last_update_time, now)

        while True:
            if node.is_leaf():
                # Place here
                node.drone_ids.add(drone_id)
                node.traffic_density = len(node.drone_ids)
                node.last_update_time = now

                # Split check (Zoom-in)
                if (
                    node.traffic_density > self.split_threshold
                    and node.depth < self.max_depth
                ):
                    node.subdivide(self._positions)
                    # IMPORTANT: subdivision redistributes many drones; refresh the leaf index
                    # so subsequent updates can remove from the correct leaf node.
                    if not node.is_leaf():
                        assert node.children is not None
                        for c in node.children:
                            for did in c.drone_ids:
                                self._drone_leaf[did] = c
                    # After subdivide, re-route this drone to its child (if split happened)
                    if not node.is_leaf():
                        assert node.children is not None
                        idx = node._child_index_for_point(x, y, z)
                        node = node.children[idx]
                        continue
                return node

            # Internal: descend to correct child
            assert node.children is not None
            idx = node._child_index_for_point(x, y, z)
            node = node.children[idx]
            node.last_update_time = max(node.last_update_time, now)


