"""
Query Performance Benchmark: DBAO vs Brute-Force
================================================

Objective
---------
Demonstrate that the DBAO structure provides O(log N) collision query performance,
significantly outperforming O(M) brute-force checks (where M is number of drones).

Scenario
--------
- 500 Drones clustered in a 100x100x100m volume.
- 10,000 Random query points (mixed hit/miss).
- Compare total time to process all queries.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple
import random

import numpy as np
try:
    from rtree import index
    HAS_RTREE = True
except ImportError:
    HAS_RTREE = False


# Ensure repo root importability
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

# Centralized figure output directory (paper-ready)
PAPER3D_FIG_DIR = REPO_ROOT / "research" / "papers" / "paper3D" / "figures"

def load_adaptive_octree_by_path():
    """
    Load `adaptive_octree.py` directly by path to avoid optional deps pulled by
    `modules.voxel_airspace_core.__init__` (e.g., shapely).
    """
    import importlib.util

    mod_name = "skynet_adaptive_octree"
    path = REPO_ROOT / "modules" / "voxel_airspace_core" / "adaptive_octree.py"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load adaptive octree module from: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module

_oct = load_adaptive_octree_by_path()
AdaptiveOctreeManager = _oct.AdaptiveOctreeManager  # type: ignore[attr-defined]

Vec3 = Tuple[float, float, float]

def try_set_academic_style() -> None:
    import matplotlib.pyplot as plt
    for style in ("seaborn-whitegrid", "seaborn-v0_8-whitegrid"):
        try:
            plt.style.use(style)
            return
        except Exception:
            continue

@dataclass(frozen=True)
class QueryConfig:
    n_drones: int = 500
    n_queries: int = 10000
    root_extent: float = 1000.0
    congest_side: float = 100.0
    collision_dist: float = 5.0  # meters (for brute force check)
    seed: int = 42

def setup_scene(cfg: QueryConfig) -> Tuple[AdaptiveOctreeManager, Dict[str, Vec3]]:
    random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    # Initialize Tree
    root_bounds = (-cfg.root_extent, -cfg.root_extent, -cfg.root_extent, 2 * cfg.root_extent)
    tree = AdaptiveOctreeManager(
        root_bounds=root_bounds,
        split_threshold=5,
        merge_threshold=2,
        max_depth=9  # High precision
    )

    # Place Drones (Congested)
    positions: Dict[str, Vec3] = {}
    half = cfg.congest_side / 2.0
    xyz = rng.uniform(-half, half, size=(cfg.n_drones, 3))
    
    for i in range(cfg.n_drones):
        did = f"drone-{i:04d}"
        x, y, z = float(xyz[i, 0]), float(xyz[i, 1]), float(xyz[i, 2])
        positions[did] = (x, y, z)
        tree.update_drone_position(did, x, y, z)
    
    return tree, positions

def generate_queries(cfg: QueryConfig) -> List[Vec3]:
    rng = np.random.default_rng(cfg.seed + 1)
    queries: List[Vec3] = []
    
    # 50% points in congestion zone (likely hits), 50% random in root (likely misses)
    n_inner = cfg.n_queries // 2
    n_outer = cfg.n_queries - n_inner
    
    # Inner
    half = cfg.congest_side / 2.0
    inner_xyz = rng.uniform(-half, half, size=(n_inner, 3))
    for i in range(n_inner):
        queries.append((inner_xyz[i, 0], inner_xyz[i, 1], inner_xyz[i, 2]))
        
    # Outer
    ext = cfg.root_extent
    outer_xyz = rng.uniform(-ext, ext, size=(n_outer, 3))
    for i in range(n_outer):
        queries.append((outer_xyz[i, 0], outer_xyz[i, 1], outer_xyz[i, 2]))
        
    random.shuffle(queries)
    return queries

def run_benchmark(cfg: QueryConfig):
    tree, positions = setup_scene(cfg)
    queries = generate_queries(cfg)
    pos_list = list(positions.values()) # For fast iteration in brute force
    
    print(f"Benchmark: {cfg.n_drones} Drones, {cfg.n_queries} Queries")
    
    # --- Experiment A: DBAO ---
    t0 = time.perf_counter()
    hits_dbao = 0
    for (qx, qy, qz) in queries:
        if tree.check_collision_point(qx, qy, qz):
            hits_dbao += 1
    t1 = time.perf_counter()
    time_dbao = t1 - t0
    
    # --- Experiment B: R-Tree ---
    time_rtree = 0.0
    hits_rtree = 0
    if HAS_RTREE:
        p = index.Property()
        p.dimension = 3
        idx = index.Index(properties=p)
        # Bulk load or insert? Let's use insert to match setup time or just pre-build?
        # We benchmark QUERY time, so build time doesn't matter for this metric.
        # But we should build it before the timer.
        for i, (did, (dx, dy, dz)) in enumerate(positions.items()):
            idx.insert(i, (dx, dy, dz, dx, dy, dz)) # Point insert
            
        # Match Voxel Depth 9 size ~ 3.9m
        voxel_size = cfg.root_extent * 2 / (2**9) 
        half = voxel_size / 2.0
        
        t0 = time.perf_counter()
        for (qx, qy, qz) in queries:
            # Check intersection with a "Voxel-sized" box around query point
            # Use next() to stop at first match (boolean check)
            match = next(idx.intersection((qx-half, qy-half, qz-half, qx+half, qy+half, qz+half)), None)
            if match is not None:
                hits_rtree += 1
        t1 = time.perf_counter()
        time_rtree = t1 - t0
    
    # --- Experiment C: Naive Brute Force ---
    # Check if point is within `collision_dist` of ANY drone
    sq_dist_thresh = cfg.collision_dist ** 2
    t0 = time.perf_counter()
    hits_naive = 0
    for (qx, qy, qz) in queries:
        # Check all drones
        found = False
        for (dx, dy, dz) in pos_list:
            d2 = (qx-dx)**2 + (qy-dy)**2 + (qz-dz)**2
            if d2 < sq_dist_thresh:
                found = True
                break
        if found:
            hits_naive += 1
    t1 = time.perf_counter()
    time_naive = t1 - t0
    
    return time_dbao, time_rtree, time_naive, hits_dbao, hits_rtree, hits_naive

def plot_results(time_dbao: float, time_rtree: float, time_naive: float, cfg: QueryConfig, out_path: Path, show: bool):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    try_set_academic_style()
    
    labels = ['DBAO (Octree)', 'R-Tree', 'Naive (Linear)']
    times = [time_dbao, time_rtree, time_naive]
    colors = ['tab:red', 'tab:blue', 'tab:gray']
    
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    bars = ax.bar(labels, times, color=colors, alpha=0.8, width=0.6)
    
    ax.set_ylabel('Total Processing Time (seconds)')
    ax.set_title(f'Collision Query Performance (Lower is Better)\n({cfg.n_queries} queries against {cfg.n_drones} drones)')
    
    # Add text on top
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f} s',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        else:
             ax.text(bar.get_x() + bar.get_width()/2., 0,
                    'N/A',
                    ha='center', va='bottom', fontsize=11)
                
    # Calculate speedup
    if time_dbao > 0:
        speedup = time_naive / time_dbao
        ax.text(0.15, 0.9, f"DBAO vs Naive:\n{speedup:.1f}x Speedup", 
                transform=ax.transAxes, ha='center', fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    
    if time_rtree > 0:
        qps_rtree = cfg.n_queries / time_rtree
    else:
        qps_rtree = 0

    if time_dbao > 0:
        qps_dbao = cfg.n_queries / time_dbao
    else:
        qps_dbao = 0
        
    qps_naive = cfg.n_queries / time_naive
    
    print(f"\nResults:")
    print(f"DBAO:   {time_dbao:.4f}s ({qps_dbao:.1f} QPS)")
    print(f"R-Tree: {time_rtree:.4f}s ({qps_rtree:.1f} QPS)")
    print(f"Naive:  {time_naive:.4f}s ({qps_naive:.1f} QPS)")
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight')
    print(f"Saved figure -> {out_path}")
    
    if show:
        plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(PAPER3D_FIG_DIR / "query_performance.png"))
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    
    cfg = QueryConfig()
    t_dbao, t_rtree, t_naive, h_dbao, h_rtree, h_naive = run_benchmark(cfg)
    
    plot_results(t_dbao, t_rtree, t_naive, cfg, Path(args.out), args.show)

if __name__ == "__main__":
    main()

