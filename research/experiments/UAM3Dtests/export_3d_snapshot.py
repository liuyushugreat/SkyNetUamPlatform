"""
3D Structure Visualization (Figure 4)
=====================================

Generates a 3D visualization of the Adaptive Octree structure.
Demonstrates "Multi-Resolution" property:
- High density area -> Small voxels (Refined)
- Empty area -> Large voxels (Coarse)
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import random
import time
from typing import List, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

# Centralized figure output directory (paper-ready)
PAPER3D_FIG_DIR = REPO_ROOT / "research" / "papers" / "paper3D" / "figures"

# Dynamic load
import importlib.util
def load_adaptive_octree():
    mod_name = "skynet_adaptive_octree_vis"
    path = REPO_ROOT / "modules" / "voxel_airspace_core" / "adaptive_octree.py"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if not spec or not spec.loader:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod

_mod = load_adaptive_octree()
AdaptiveOctreeManager = _mod.AdaptiveOctreeManager
AdaptiveOctreeNode = _mod.AdaptiveOctreeNode

def get_leaf_boxes(node, boxes: List[Tuple[float, float, float, float, bool]]):
    """Recursively collect (x, y, z, size, is_occupied) for all leaves."""
    if node.is_leaf():
        is_occupied = node.traffic_density > 0
        boxes.append((node.x, node.y, node.z, node.size, is_occupied))
    else:
        for c in node.children:
            get_leaf_boxes(c, boxes)

def draw_cube(ax, x, y, z, size, color, alpha, linewidth):
    """Draw a wireframe cube."""
    # Efficient Cube Wireframe:
    X = [x, x+size, x+size, x, x,   x, x+size, x+size, x, x]
    Y = [y, y, y+size, y+size, y,   y, y, y+size, y+size, y]
    Z = [z, z, z, z, z,             z+size, z+size, z+size, z+size, z+size]
    
    ax.plot(X, Y, Z, color=color, alpha=alpha, linewidth=linewidth)
    
    # Pillars
    for dx in (0, size):
        for dy in (0, size):
            ax.plot([x+dx, x+dx], [y+dy, y+dy], [z, z+size], color=color, alpha=alpha, linewidth=linewidth)

def draw_ground(ax, bounds):
    """Draw a ground plane grid at min_z."""
    min_x, min_y, min_z, size = bounds
    max_x = min_x + size
    max_y = min_y + size
    
    # Create grid
    grid_res = 10
    x = np.linspace(min_x, max_x, grid_res)
    y = np.linspace(min_y, max_y, grid_res)
    xx, yy = np.meshgrid(x, y)
    zz = np.full_like(xx, min_z)
    
    # Plot wireframe grid
    ax.plot_wireframe(xx, yy, zz, color='gray', alpha=0.2, linewidth=0.5)
    
    # Plot surface (optional, maybe too heavy)
    # ax.plot_surface(xx, yy, zz, color='lightgray', alpha=0.1)

def draw_buildings(ax):
    """Add static building obstacles."""
    # Define (x, y, z, dx, dy, dz)
    buildings = [
        (-200, -200, -500, 150, 150, 300),  # Building 1
        (200, -300, -500, 100, 200, 400),   # Building 2
        (-300, 200, -500, 200, 100, 250),   # Building 3
    ]
    
    for (x, y, z, dx, dy, dz) in buildings:
        ax.bar3d(x, y, z, dx, dy, dz, color='gray', alpha=0.3, edgecolor='black', linewidth=0.5, zsort='average')

def main():
    # 1. Setup Scene
    print("Initializing DBAO...")
    root_bounds = (-500, -500, -500, 1000) # 1000m size
    manager = AdaptiveOctreeManager(
        root_bounds=root_bounds,
        split_threshold=3,
        merge_threshold=1,
        max_depth=5 
    )
    
    # Cluster of drones
    np.random.seed(42)
    # Cluster 1: Center (Avoid placing inside buildings)
    for _ in range(30):
        manager.update_drone_position(f"d{_}", 
                                      np.random.uniform(0, 100),
                                      np.random.uniform(0, 100),
                                      np.random.uniform(0, 100))
                                      
    # Cluster 2: Corner (High up)
    for _ in range(20):
        manager.update_drone_position(f"c{_}", 
                                      np.random.uniform(300, 400),
                                      np.random.uniform(300, 400),
                                      np.random.uniform(300, 400))

    # 2. Extract Structure
    boxes = []
    get_leaf_boxes(manager.root, boxes)
    print(f"Extracted {len(boxes)} leaf nodes.")
    
    # 3. Visualize
    fig = plt.figure(figsize=(10, 10), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect((1, 1, 1))
    
    # Draw Ground
    draw_ground(ax, root_bounds)
    
    # Draw Buildings
    draw_buildings(ax)
    
    # Draw boxes
    count_drawn = 0
    for (x, y, z, s, occupied) in boxes:
        if occupied:
            draw_cube(ax, x, y, z, s, color='tab:red', alpha=0.8, linewidth=1.5)
            # Center point
            ax.scatter(x+s/2, y+s/2, z+s/2, color='darkred', s=10)
            count_drawn += 1
        else:
            # Draw empty nodes
            draw_cube(ax, x, y, z, s, color='lightgray', alpha=0.1, linewidth=0.5)
            count_drawn += 1
            
    print(f"Drawn {count_drawn} elements.")
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title("DBAO Structure in Urban Environment")
    
    # Legend
    from matplotlib.lines import Line2D
    import matplotlib.patches as mpatches
    legend_elements = [
        Line2D([0], [0], color='tab:red', lw=2, label='Occupied Voxel (High Res)'),
        Line2D([0], [0], color='lightgray', lw=1, label='Empty Voxel (Coarse)'),
        mpatches.Patch(color='gray', alpha=0.3, label='Static Obstacles (Buildings)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Adjust view
    ax.view_init(elev=25, azim=45)
    
    out_path = PAPER3D_FIG_DIR / "3d_structure_view.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight')
    print(f"Saved 3D snapshot to {out_path}")

if __name__ == "__main__":
    main()
