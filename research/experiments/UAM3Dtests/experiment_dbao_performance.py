"""
DBAO Performance Experiment: Congestion -> Dispersion
====================================================

Goal
----
Simulate a "swarm converges then disperses" scenario to validate the
Density-Based Adaptive Octree (DBAO) behavior implemented in
`modules/voxel_airspace_core/adaptive_octree.py`.

We record the total octree node count over time and plot a performance curve
that is suitable for paper figures.

Expected qualitative behavior
-----------------------------
- Phase 1 (Congestion): drones converge to a common center -> density increases
  -> octree triggers `subdivide()` ("zoom in") -> total node count spikes.
- Phase 2 (Dispersion): drones spread out -> density decreases
  -> octree triggers `merge()` ("zoom out") -> total node count falls.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import random
import sys
from typing import Dict, List, Tuple

import numpy as np

# Matplotlib import is intentionally delayed; in headless environments we switch backend.


# Ensure repo root is on sys.path so `modules.*` is importable when running as a script.
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

# Centralized figure output directory (paper-ready)
PAPER3D_FIG_DIR = REPO_ROOT / "research" / "papers" / "paper3D" / "figures"

# IMPORTANT:
# `modules/voxel_airspace_core/__init__.py` may import optional heavy deps (e.g. shapely).
# For this experiment we only need `adaptive_octree.py`, so we load it directly by path
# to avoid requiring shapely in the experimental environment.
import importlib.util  # noqa: E402

_ADAPTIVE_PATH = REPO_ROOT / "modules" / "voxel_airspace_core" / "adaptive_octree.py"
_spec = importlib.util.spec_from_file_location("skynet_adaptive_octree", _ADAPTIVE_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Failed to load adaptive octree module from: {_ADAPTIVE_PATH}")
_mod = importlib.util.module_from_spec(_spec)
# Register module before exec_module so dataclasses' type resolution works on Python 3.12+.
sys.modules["skynet_adaptive_octree"] = _mod
_spec.loader.exec_module(_mod)

AdaptiveOctreeManager = _mod.AdaptiveOctreeManager  # type: ignore[attr-defined]
AdaptiveOctreeNode = _mod.AdaptiveOctreeNode  # type: ignore[attr-defined]


Vec3 = Tuple[float, float, float]


def try_set_academic_style() -> None:
    """Best-effort apply an academic plot style across matplotlib versions."""
    import matplotlib.pyplot as plt  # local import

    # Matplotlib renamed seaborn styles around v3.6/v3.7; try both.
    for style in ("seaborn-whitegrid", "seaborn-v0_8-whitegrid"):
        try:
            plt.style.use(style)
            return
        except Exception:
            continue


def count_total_nodes(root: AdaptiveOctreeNode) -> int:
    """Count total nodes in the octree via DFS traversal."""
    if root.children is None:
        return 1
    return 1 + sum(count_total_nodes(c) for c in root.children)


def attach_console_hooks(verbose: bool = True) -> None:
    """
    Monkey-patch `AdaptiveOctreeNode.subdivide/merge` to print key events.

    This keeps the core module clean while still providing paper-friendly
    experimental logs like "Splitting node..." and "Merging node...".
    """

    if not verbose:
        return

    orig_subdivide = AdaptiveOctreeNode.subdivide
    orig_merge = AdaptiveOctreeNode.merge

    def subdivide_with_log(self: AdaptiveOctreeNode, positions: Dict[str, Vec3]) -> None:
        print(
            f"Splitting node... depth={self.depth}, bounds={self.bounds_tuple()}, "
            f"density={self.traffic_density}"
        )
        return orig_subdivide(self, positions)

    def merge_with_log(self: AdaptiveOctreeNode) -> None:
        # For internal nodes, density is the sum of children densities.
        density = self.traffic_density
        print(
            f"Merging node... depth={self.depth}, bounds={self.bounds_tuple()}, "
            f"child_sum_density={density}"
        )
        return orig_merge(self)

    AdaptiveOctreeNode.subdivide = subdivide_with_log  # type: ignore[method-assign]
    AdaptiveOctreeNode.merge = merge_with_log  # type: ignore[method-assign]


@dataclass(frozen=True)
class ExperimentConfig:
    n_drones: int = 10
    split_threshold: int = 5
    merge_threshold: int = 2
    max_depth: int = 7
    t_congest_end: int = 50
    t_end: int = 100
    init_range: float = 500.0
    root_padding: float = 200.0
    seed: int = 7


def generate_initial_positions(cfg: ExperimentConfig) -> Dict[str, Vec3]:
    """
    Initialize drones far apart within [-init_range, init_range].
    This reduces initial density so the tree starts coarse (near 1 node).
    """
    rng = np.random.default_rng(cfg.seed)
    positions: Dict[str, Vec3] = {}
    for i in range(cfg.n_drones):
        did = f"drone-{i:02d}"
        x, y, z = rng.uniform(-cfg.init_range, cfg.init_range, size=3).tolist()
        positions[did] = (float(x), float(y), float(z))
    return positions


def converge_step(p0: Vec3, t: int, t_end: int, center: Vec3 = (0.0, 0.0, 0.0)) -> Vec3:
    """Linear interpolation towards center for congestion phase."""
    alpha = min(1.0, max(0.0, t / float(t_end)))
    return (
        (1 - alpha) * p0[0] + alpha * center[0],
        (1 - alpha) * p0[1] + alpha * center[1],
        (1 - alpha) * p0[2] + alpha * center[2],
    )


def disperse_step(
    p: Vec3,
    *,
    step_size: float,
    rng: np.random.Generator,
    clamp: float,
) -> Vec3:
    """Random walk dispersal away from the cluster center."""
    direction = rng.normal(0.0, 1.0, size=3)
    norm = float(np.linalg.norm(direction) + 1e-9)
    direction = direction / norm
    nx = float(p[0] + direction[0] * step_size)
    ny = float(p[1] + direction[1] * step_size)
    nz = float(p[2] + direction[2] * step_size)
    # Keep within root bounds (defensive) so drones don't escape the root cube.
    nx = max(-clamp, min(clamp, nx))
    ny = max(-clamp, min(clamp, ny))
    nz = max(-clamp, min(clamp, nz))
    return (nx, ny, nz)


def run_experiment(cfg: ExperimentConfig, *, verbose: bool = True) -> Tuple[List[int], List[int]]:
    # Seed both numpy & python random for reproducibility.
    random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    init_positions = generate_initial_positions(cfg)

    # Root cube should safely cover the entire motion envelope.
    # We use a cube centered at origin, represented by min-corner (x,y,z,size).
    extent = cfg.init_range + cfg.root_padding
    root_bounds = (-extent, -extent, -extent, 2 * extent)

    tree = AdaptiveOctreeManager(
        root_bounds=root_bounds,
        split_threshold=cfg.split_threshold,
        merge_threshold=cfg.merge_threshold,
        max_depth=cfg.max_depth,
    )

    # Prime the octree with initial positions
    for did, (x, y, z) in init_positions.items():
        tree.update_drone_position(did, x, y, z)

    times: List[int] = []
    node_counts: List[int] = []

    # Pre-store initial positions for convergence interpolation
    p0_map = init_positions
    positions = dict(init_positions)

    for t in range(0, cfg.t_end + 1):
        before = count_total_nodes(tree.root)

        if t <= cfg.t_congest_end:
            # Phase 1: congestion (converge to origin)
            for did, p0 in p0_map.items():
                positions[did] = converge_step(p0, t, cfg.t_congest_end, center=(0.0, 0.0, 0.0))
        else:
            # Phase 2: dispersion (random walk away)
            # Step size chosen to visibly reduce density over ~50 steps.
            for did, p in positions.items():
                positions[did] = disperse_step(
                    p,
                    step_size=35.0,
                    rng=rng,
                    clamp=extent * 0.95,
                )

        # Update octree with new positions (may trigger splits)
        for did, (x, y, z) in positions.items():
            tree.update_drone_position(did, x, y, z)

        # Periodic prune to enable merges (zoom-out)
        merges = tree.prune_tree()

        after = count_total_nodes(tree.root)
        times.append(t)
        node_counts.append(after)

        # Console logs (paper-friendly)
        if verbose:
            if t <= cfg.t_congest_end:
                print(f"[Step {t:03d}] Drones converging... Tree Nodes: {before} -> {after}")
                if after > before and after >= 9:
                    print(
                        f"[Step {t:03d}] Drones clustered! Tree Nodes: {before} -> {after} "
                        f"(High Precision Mode)"
                    )
            else:
                print(
                    f"[Step {t:03d}] Drones dispersing... Tree Nodes: {before} -> {after} "
                    f"(merges={merges})"
                )

    return times, node_counts


def plot_results(
    times: List[int],
    node_counts: List[int],
    *,
    cfg: ExperimentConfig,
    out_path: Path,
    show: bool,
) -> None:
    import matplotlib
    import matplotlib.pyplot as plt

    try_set_academic_style()

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=150)
    ax.plot(times, node_counts, linewidth=2.0, color="#1f77b4")
    ax.set_xlabel("Simulation Time Step")
    ax.set_ylabel("Total Voxel Nodes in Octree")
    ax.set_title("DBAO Adaptive Octree: Node Count vs. Time")

    # Phase annotations
    # Find a reasonable y for arrow anchors
    y_max = max(node_counts) if node_counts else 1
    ax.annotate(
        "Congestion Event\n(High Resolution)",
        xy=(cfg.t_congest_end, node_counts[cfg.t_congest_end]),
        xytext=(cfg.t_congest_end - 25, y_max * 0.85),
        arrowprops=dict(arrowstyle="->", linewidth=1.2),
        fontsize=10,
        ha="center",
    )
    ax.annotate(
        "Dispersion Event\n(Low Resolution)",
        xy=(cfg.t_congest_end + 10, node_counts[min(cfg.t_congest_end + 10, len(node_counts) - 1)]),
        xytext=(cfg.t_congest_end + 35, y_max * 0.45),
        arrowprops=dict(arrowstyle="->", linewidth=1.2),
        fontsize=10,
        ha="center",
    )

    ax.axvspan(0, cfg.t_congest_end, color="#ff7f0e", alpha=0.08, label="Congestion Phase")
    ax.axvspan(cfg.t_congest_end, cfg.t_end, color="#2ca02c", alpha=0.06, label="Dispersion Phase")
    ax.legend(loc="upper right", frameon=True)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"\nSaved figure -> {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="DBAO performance experiment (congestion -> dispersion).")
    parser.add_argument("--no-verbose", action="store_true", help="Disable per-step console logs.")
    parser.add_argument("--show", action="store_true", help="Show plot window (may block).")
    parser.add_argument(
        "--out",
        type=str,
        default=str(PAPER3D_FIG_DIR / "experiment_dbao_performance.png"),
        help="Output figure path (png/pdf).",
    )
    args = parser.parse_args()

    # Headless-friendly backend selection (safe if already set).
    if not args.show:
        import matplotlib

        matplotlib.use("Agg", force=True)

    cfg = ExperimentConfig()

    attach_console_hooks(verbose=not args.no_verbose)
    times, node_counts = run_experiment(cfg, verbose=not args.no_verbose)
    plot_results(times, node_counts, cfg=cfg, out_path=Path(args.out), show=args.show)


if __name__ == "__main__":
    main()


