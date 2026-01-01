"""
Hysteresis Mechanism Concept Plot (Figure 3)
============================================

Visualizes the "Schmidt Trigger" like behavior of the DBAO Split/Merge logic.
Ensures structural stability when traffic density fluctuates.

State Logic:
- Initial State: Low Depth (0)
- Transition Low -> High: if Density > TAU_SPLIT (5)
- Transition High -> Low: if Density < TAU_MERGE (2)
- Hysteresis Zone: [2, 5] -> No Change
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PAPER3D_FIG_DIR = REPO_ROOT / "research" / "papers" / "paper3D" / "figures"

def try_set_academic_style():
    for style in ("seaborn-whitegrid", "seaborn-v0_8-whitegrid"):
        try:
            plt.style.use(style)
            return
        except Exception:
            continue

def main():
    try_set_academic_style()
    
    # Parameters
    TAU_SPLIT = 5
    TAU_MERGE = 2
    T_MAX = 100
    
    # Generate Synthetic Density Signal
    t = np.linspace(0, T_MAX, 500)
    # Signal: Rise, Fluctuate in middle, Peak, Fall, Fluctuate in middle, Bottom out
    # We construct a specific curve to demo all transitions
    # 0-20: Rising
    # 20-40: Fluctuating in [3, 4.5] (Should stay LOW if started low? No, let's force a split first)
    # Let's use a simpler sine wave with noise that crosses thresholds cleanly
    density = 4 + 4 * np.sin(t * 0.1) + np.random.normal(0, 0.2, len(t))
    # Clip negative
    density = np.maximum(0, density)
    
    # Simulate State Machine
    depth_state = []
    current_depth = 0 # 0=Low (Merged), 1=High (Split)
    
    events = [] # (time, type)
    
    for i, val in enumerate(density):
        if current_depth == 0:
            if val > TAU_SPLIT:
                current_depth = 1
                events.append((t[i], "Split\n(>5)"))
        else: # current_depth == 1
            if val < TAU_MERGE:
                current_depth = 0
                events.append((t[i], "Merge\n(<2)"))
        
        depth_state.append(current_depth)
        
    depth_state = np.array(depth_state)
    
    # --- Plotting ---
    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=150)
    
    # Plot Density (Left Axis)
    color = 'tab:blue'
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Traffic Density (Agents)', color=color)
    ax1.plot(t, density, color=color, alpha=0.6, label='Traffic Density')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 10)
    
    # Threshold Lines
    ax1.axhline(y=TAU_SPLIT, color='r', linestyle='--', alpha=0.5, label=r'$\tau_{split}=5$')
    ax1.axhline(y=TAU_MERGE, color='g', linestyle='--', alpha=0.5, label=r'$\tau_{merge}=2$')
    
    # Hysteresis Band
    ax1.fill_between(t, TAU_MERGE, TAU_SPLIT, color='gray', alpha=0.1, label='Hysteresis Zone')
    
    # Plot Depth (Right Axis)
    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Octree Node Depth (State)', color=color)
    # Scale depth for visibility: 0 -> Low, 1 -> High
    ax2.step(t, depth_state, where='post', color=color, linewidth=2.5, label='Tree Depth')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Low (Coarse)', 'High (Refined)'])
    ax2.set_ylim(-0.2, 1.2)
    
    # Annotate Events
    for t_evt, label in events:
        # Find y position on density curve
        # idx = int(t_evt / T_MAX * 500)
        # y_val = density[min(idx, 499)]
        ax2.annotate(label, xy=(t_evt, 0.5), xytext=(t_evt, 0.5),
                     ha='center', va='center', fontsize=9, fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))
                     
    # Title and Layout
    plt.title('Hysteresis Mechanism for Structural Stability')
    fig.tight_layout()
    
    # Save
    out_path = PAPER3D_FIG_DIR / "hysteresis_mechanism.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    print(f"Saved hysteresis plot to {out_path}")
    
    # Also save as PDF for latex
    # plt.savefig(out_path.with_suffix('.pdf'))

if __name__ == "__main__":
    main()
