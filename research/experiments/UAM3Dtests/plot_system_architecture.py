"""
System Architecture Diagram Generator (Figure 2)
================================================

Generates a high-resolution, academic-style flowchart for the DBAO system architecture.
Flow: Drone Telemetry -> Spatial Hasher -> DBAO Engine -> Query API -> UTM Service

Output: ../../papers/paper3D/figures/figure_architecture.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def draw_box(ax, xy, width, height, text, subtext=None, highlighted=False):
    """Draw a text box with optional subtext and highlighting."""
    x, y = xy
    # Style
    fc = '#e6e6e6' if highlighted else 'white'
    ec = 'black'
    lw = 2.0 if highlighted else 1.2
    
    # Box
    rect = patches.FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.1",
        linewidth=lw, edgecolor=ec, facecolor=fc,
        zorder=2
    )
    ax.add_patch(rect)
    
    # Main Text
    cx = x + width / 2
    cy = y + height / 2
    if subtext:
        cy += height * 0.15 # Shift up slightly
        
    ax.text(cx, cy, text, ha='center', va='center', fontsize=11, fontweight='bold', zorder=3)
    
    # Subtext
    if subtext:
        ax.text(cx, y + height * 0.25, subtext, ha='center', va='center', fontsize=9, style='italic', zorder=3)
        
    return (x + width, y + height / 2), (x, y + height / 2) # Return connection points (right, left)

def draw_arrow(ax, start, end):
    """Draw an arrow from start (x,y) to end (x,y)."""
    ax.annotate("",
                xy=end, xycoords='data',
                xytext=start, textcoords='data',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="arc3", 
                                color="black", lw=1.5),
                zorder=1)

def main():
    # Setup Figure
    fig, ax = plt.subplots(figsize=(12, 4), dpi=300)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis('off') # Turn off axes
    
    # Nodes Config
    box_w = 1.8
    box_h = 1.2
    y_center = 1.5
    
    # Coordinates (manual layout for precision)
    x_gap = 0.5
    x_start = 0.5
    
    nodes = []
    
    # 1. Input
    pos1 = (x_start, y_center)
    r1, l1 = draw_box(ax, pos1, box_w, box_h, "Drone\nTelemetry", subtext="(GPS/IMU)")
    
    # 2. Preprocessing
    pos2 = (pos1[0] + box_w + x_gap, y_center)
    r2, l2 = draw_box(ax, pos2, box_w, box_h, "Spatial\nHasher")
    
    # 3. Core (Highlighted)
    pos3 = (pos2[0] + box_w + x_gap, y_center)
    r3, l3 = draw_box(ax, pos3, box_w, box_h, "DBAO Engine", subtext="Split/Merge Logic", highlighted=True)
    
    # 4. Interface
    pos4 = (pos3[0] + box_w + x_gap, y_center)
    r4, l4 = draw_box(ax, pos4, box_w, box_h, "Query API", subtext="check_collision()")
    
    # 5. Output
    pos5 = (pos4[0] + box_w + x_gap, y_center)
    r5, l5 = draw_box(ax, pos5, box_w, box_h, "UTM Service", subtext="Collision Alert")
    
    # Connections
    draw_arrow(ax, r1, l2)
    draw_arrow(ax, r2, l3)
    draw_arrow(ax, r3, l4)
    draw_arrow(ax, r4, l5)
    
    # Title (optional, usually handled by LaTeX caption)
    # ax.set_title("DBAO System Architecture", fontsize=14, pad=20)
    
    # Save
    out_dir = Path(__file__).resolve().parents[2] / "papers" / "paper3D" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "figure_architecture.png"
    
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    print(f"Saved architecture diagram to: {out_path}")

if __name__ == "__main__":
    main()

