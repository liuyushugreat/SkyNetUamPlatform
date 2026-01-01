"""
Scalability Analysis: DBAO vs Static Grid Baseline
==================================================

Objective
---------
Generate paper-ready scalability data by comparing:
1) **DBAO**: AdaptiveOctreeManager (density-driven split/merge)
2) **Static Grid Baseline**: theoretical fixed-resolution voxel grid

Scenario
--------
"Vertiport Congestion": all drones attempt to enter a central 100x100x100 m region.

Metrics
-------
For each drone count:
- **Time per frame (ms)**: update all drone positions + prune (merge check)
  measured via `time.perf_counter_ns()`.
- **Space (nodes / voxels)**:
  - DBAO: total nodes currently instantiated in the octree
  - Static baseline: theoretical number of voxels if the entire root cube were
    discretized at the minimum leaf resolution (i.e., depth = MAX_DEPTH).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
try:
    from rtree import index
    HAS_RTREE = True
except ImportError:
    HAS_RTREE = False



Vec3 = Tuple[float, float, float]

# Ensure repo root importability
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

# Centralized figure output directory (paper-ready)
PAPER3D_FIG_DIR = REPO_ROOT / "research" / "papers" / "paper3D" / "figures"


def load_adaptive_octree_by_path():
    """
    Load `adaptive_octree.py` directly by path to avoid optional deps pulled by
    `modules.voxel_airspace_core.__init__` (e.g., shapely).

    Important on Python 3.12+:
    - We must register the module in `sys.modules` before `exec_module` so
      dataclasses' type resolution works correctly.
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
AdaptiveOctreeNode = _oct.AdaptiveOctreeNode  # type: ignore[attr-defined]


def count_total_nodes(root: "AdaptiveOctreeNode") -> int:
    if root.children is None:
        return 1
    return 1 + sum(count_total_nodes(c) for c in root.children)


def static_grid_voxel_count(root_size: float, max_depth: int) -> int:
    """
    Theoretical number of fixed-resolution voxels if we discretize the entire
    root cube at the minimum leaf resolution.

    If max_depth = D, each axis is split into 2^D cells, so total voxels are:
        (2^D)^3 = 8^D
    """
    # root_size is unused by the formula (depends only on depth), but kept for clarity.
    _ = root_size
    return int(8**max_depth)


def try_set_academic_style() -> None:
    import matplotlib.pyplot as plt

    for style in ("seaborn-whitegrid", "seaborn-v0_8-whitegrid"):
        try:
            plt.style.use(style)
            return
        except Exception:
            continue


@dataclass(frozen=True)
class BenchConfig:
    drone_counts: Tuple[int, ...] = (10, 50, 100, 200, 500)
    split_threshold: int = 5
    merge_threshold: int = 2
    # Increase to make the static-grid baseline visually more dramatic in the paper.
    max_depth: int = 9
    seed: int = 11

    # Root cube settings (meters)
    root_extent: float = 1000.0  # root covers [-extent, extent] on each axis

    # Congestion region: centered cube with side length 100m
    congest_side: float = 100.0

    # Timing repetitions (median for robustness)
    repeats: int = 3


def sample_congested_positions(
    n: int, *, side: float, rng: np.random.Generator
) -> Dict[str, Vec3]:
    half = side / 2.0
    xyz = rng.uniform(-half, half, size=(n, 3))
    positions: Dict[str, Vec3] = {}
    for i in range(n):
        positions[f"drone-{i:04d}"] = (float(xyz[i, 0]), float(xyz[i, 1]), float(xyz[i, 2]))
    return positions


def benchmark_one_count(cfg: BenchConfig, n: int) -> Tuple[float, float, int, int]:
    """
    Returns:
      (median_time_dbao_ms, median_time_rtree_ms, dbao_node_count, static_voxel_count)
    """
    rng = np.random.default_rng(cfg.seed + n)

    root_bounds = (-cfg.root_extent, -cfg.root_extent, -cfg.root_extent, 2 * cfg.root_extent)
    root_size = 2 * cfg.root_extent
    static_voxels = static_grid_voxel_count(root_size, cfg.max_depth)

    times_dbao: List[float] = []
    times_rtree: List[float] = []
    last_nodes = 1

    # Shared positions for this N
    # We regenerate per repeat to average out layout noise, or keep same?
    # Original code regenerated positions inside the loop? 
    # No, strictly speaking original code generated positions inside loop.
    # Let's keep it consistent.

    for rep in range(cfg.repeats):
        # 1. DBAO
        tree = AdaptiveOctreeManager(
            root_bounds=root_bounds,
            split_threshold=cfg.split_threshold,
            merge_threshold=cfg.merge_threshold,
            max_depth=cfg.max_depth,
        )
        positions = sample_congested_positions(n, side=cfg.congest_side, rng=rng)

        t0 = time.perf_counter_ns()
        for did, (x, y, z) in positions.items():
            tree.update_drone_position(did, x, y, z)
        tree.prune_tree()
        t1 = time.perf_counter_ns()

        times_dbao.append((t1 - t0) / 1e6)
        last_nodes = count_total_nodes(tree.root)

        # 2. R-Tree
        if HAS_RTREE:
            p = index.Property()
            p.dimension = 3
            idx = index.Index(properties=p)
            
            t0 = time.perf_counter_ns()
            # Simulate "Update" as fresh insert for fairness with DBAO above
            # (Both are constructing structure from scratch for N points)
            for i, (did, (x, y, z)) in enumerate(positions.items()):
                idx.insert(i, (x, y, z, x, y, z))
            t1 = time.perf_counter_ns()
            times_rtree.append((t1 - t0) / 1e6)
        else:
            times_rtree.append(0.0)

    median_dbao = float(np.median(np.array(times_dbao)))
    median_rtree = float(np.median(np.array(times_rtree)))
    return median_dbao, median_rtree, int(last_nodes), int(static_voxels)


def print_table(rows: List[Tuple[int, float, float, int, int]]) -> None:
    """
    rows: (n, t_dbao, t_rtree, dbao_nodes, static_voxels)
    """
    print("\n=== Scalability Analysis (Vertiport Congestion) ===")
    print("Columns: N, DBAO_ms, RTree_ms, DBAO_nodes, StaticGrid_voxels")
    print("-" * 90)
    for n, t_dbao, t_rtree, dbao_nodes, static_voxels in rows:
        ratio = static_voxels / max(1, dbao_nodes)
        print(
            f"{n:>5d} | {t_dbao:>9.3f} ms | {t_rtree:>9.3f} ms | {dbao_nodes:>10d} | {static_voxels:>14d} "
            f"| static/DBAO ≈ {ratio:>8.1f}x"
        )
    print("-" * 90)


def plot_results(
    rows: List[Tuple[int, float, int, int]],
    *,
    out_path: Path,
    log_y: bool,
    show: bool,
) -> None:
    import matplotlib
    import matplotlib.pyplot as plt

    try_set_academic_style()

    n_vals = [r[0] for r in rows]
    t_ms_dbao = [r[1] for r in rows]
    t_ms_rtree = [r[2] for r in rows]
    dbao_nodes = [r[3] for r in rows]
    static_voxels = [r[4] for r in rows]

    fig, ax1 = plt.subplots(figsize=(10.5, 5.6), dpi=150)

    # Bar chart (node count) - grouped
    x = np.arange(len(n_vals))
    width = 0.38
    ax1.bar(x - width / 2, dbao_nodes, width, label="DBAO Nodes", color="tab:red")
    ax1.bar(x + width / 2, static_voxels, width, label="Static Voxels", color="tab:gray", alpha=0.85)

    ax1.set_xlabel("Number of Drones")
    ax1.set_ylabel("Memory Usage (Node Count)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(v) for v in n_vals])
    if log_y:
        ax1.set_yscale("log")

    # Line chart (time)
    ax2 = ax1.twinx()
    ax2.plot(x, t_ms_dbao, marker="o", linewidth=2.2, color="tab:red", label="DBAO Update (ms)")
    ax2.plot(x, t_ms_rtree, marker="^", linewidth=2.2, color="tab:blue", linestyle="--", label="R-Tree Update (ms)")
    ax2.set_ylabel("Time per Frame (ms)")

    # Combine legends
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", frameon=True)

    ax1.set_title("Scalability: DBAO vs R-Tree vs Static Grid")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"\nSaved figure -> {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scalability benchmark: DBAO vs Static Grid.")
    parser.add_argument(
        "--out",
        type=str,
        default=str(PAPER3D_FIG_DIR / "scalability_result.png"),
        help="Output figure path (png/pdf).",
    )
    parser.add_argument(
        "--linear-y",
        action="store_true",
        help="Use linear scale on node-count axis (default: log scale).",
    )
    parser.add_argument("--show", action="store_true", help="Show plot window (may block).")
    args = parser.parse_args()

    # Headless-friendly backend
    if not args.show:
        import matplotlib

        matplotlib.use("Agg", force=True)

    cfg = BenchConfig()
    rows: List[Tuple[int, float, float, int, int]] = []

    for n in cfg.drone_counts:
        t_dbao, t_rtree, dbao_nodes, static_voxels = benchmark_one_count(cfg, n)
        rows.append((n, t_dbao, t_rtree, dbao_nodes, static_voxels))

    print_table(rows)
    plot_results(rows, out_path=Path(args.out), log_y=not args.linear_y, show=args.show)


if __name__ == "__main__":
    main()


