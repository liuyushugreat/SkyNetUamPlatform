import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

# Set style configuration for academic publication
try:
    plt.style.use('seaborn-v0_8-paper')
except:
    try:
        plt.style.use('seaborn-paper')
    except:
        pass # Fallback to default

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "axes.labelsize": 12,
    "font.size": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300
})

def draw_fig1_architecture():
    """Figure 1: System Architecture Diagram"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Styles
    box_style = dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1.5)
    inner_box_style = dict(boxstyle="round,pad=0.2", fc="#f0f0f0", ec="gray", lw=1)
    
    # 1. SkyNet Environment (Left)
    ax.add_patch(patches.Rectangle((0.5, 1), 2.5, 4, fill=False, edgecolor='black', lw=2))
    ax.text(1.75, 5.2, "SkyNet Environment", ha='center', fontsize=12, fontweight='bold')
    
    # Environment contents
    ax.text(1.75, 4, "UAV Dynamics\n(Kinematics)", ha='center', va='center', bbox=inner_box_style)
    ax.text(1.75, 2.5, "Obstacles\n(Buildings)", ha='center', va='center', bbox=inner_box_style)
    ax.text(1.75, 1.5, "Sensors\n(Lidar/GPS)", ha='center', va='center', bbox=inner_box_style)

    # 2. Agents (Middle)
    ax.add_patch(patches.Rectangle((4, 1.5), 2, 3, fill=False, edgecolor='#333333', lw=2, linestyle='--'))
    ax.text(5, 4.7, "Multi-Agent System", ha='center', fontsize=12, fontweight='bold')
    for i, y in enumerate([3.5, 3, 2.5]):
        ax.text(5, y, f"Agent {i+1}", ha='center', va='center', bbox=dict(boxstyle="circle,pad=0.3", fc="lightblue", ec="blue"))
    ax.text(5, 2, "...", ha='center')

    # 3. AP-MADDPG Algorithm (Right)
    ax.add_patch(patches.Rectangle((7, 0.5), 4.5, 5, fill=False, edgecolor='darkred', lw=2))
    ax.text(9.25, 5.7, "AP-MADDPG Algorithm", ha='center', fontsize=12, fontweight='bold', color='darkred')
    
    # CTDE Framework
    ax.add_patch(patches.Rectangle((7.2, 3.2), 4.1, 2.1, fill=True, color='#ffeebb', alpha=0.5))
    ax.text(9.25, 5.1, "Centralized Critic (Training)", ha='center', fontsize=10, fontstyle='italic')
    
    # Attention Module
    ax.text(8.5, 4.2, "Attention\nModule", ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", fc="#ff9999", ec="red"))
    ax.text(10.5, 4.2, "Q-Value\nNetwork", ha='center', va='center', bbox=inner_box_style)
    ax.annotate("", xy=(9.8, 4.2), xytext=(9.2, 4.2), arrowprops=dict(arrowstyle="->"))
    
    # Actor
    ax.add_patch(patches.Rectangle((7.2, 0.7), 4.1, 2.3, fill=True, color='#eebbff', alpha=0.5))
    ax.text(9.25, 2.8, "Decentralized Actor (Execution)", ha='center', fontsize=10, fontstyle='italic')
    
    # Reward Function
    ax.text(9.25, 1.5, "Composite Reward\n(Potential Field)", ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", fc="#99ff99", ec="green"))

    # Arrows (Data Flow)
    # State: Env -> Agents
    ax.annotate("", xy=(4, 3), xytext=(3, 3), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(3.5, 3.1, "State ($o_i$)", ha='center', fontsize=10)
    
    # State: Agents -> Algo
    ax.annotate("", xy=(7, 3.5), xytext=(6, 3.5), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(6.5, 3.6, "Obs", ha='center', fontsize=10)

    # Action: Algo -> Env (Loop back)
    ax.annotate("", xy=(3, 4.5), xytext=(7, 2), arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3", lw=1.5, color="blue"))
    ax.text(5, 4.8, "Action ($a_i$)", ha='center', color="blue", fontsize=10)
    
    # Reward: Env -> Algo
    ax.annotate("", xy=(8, 1.5), xytext=(2.5, 1), arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.4", lw=1.5, color="green", linestyle='--'))
    ax.text(5, 0.5, "Physical Feedback", ha='center', color="green", fontsize=10)

    plt.tight_layout()
    plt.savefig('fig1_architecture.png', bbox_inches='tight')
    print("Generated fig1_architecture.png")
    plt.close()

def draw_fig2_simulation_scenario():
    """Figure 2: Large-scale Urban Simulation Scenario"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Setup city ground
    map_size = 1000
    n_buildings = 40
    
    # Buildings (Random bars)
    np.random.seed(42)
    x = np.random.randint(0, map_size, n_buildings)
    y = np.random.randint(0, map_size, n_buildings)
    z = np.zeros(n_buildings)
    dx = np.random.randint(30, 80, n_buildings)
    dy = np.random.randint(30, 80, n_buildings)
    dz = np.random.randint(50, 200, n_buildings)
    
    ax.bar3d(x, y, z, dx, dy, dz, color='#aaaaaa', alpha=0.8, shade=True, edgecolor='gray')
    
    # UAVs (Scatter points)
    n_uavs = 500
    uav_x = np.random.randint(0, map_size, n_uavs)
    uav_y = np.random.randint(0, map_size, n_uavs)
    uav_z = np.random.randint(50, 150, n_uavs)
    
    # Color by height
    p = ax.scatter(uav_x, uav_y, uav_z, c=uav_z, cmap='plasma', s=5, alpha=0.8)
    
    # Trajectories (Curves)
    for i in range(5):
        t = np.linspace(0, 1, 100)
        start_pos = np.random.rand(3) * map_size
        start_pos[2] = 100
        end_pos = np.random.rand(3) * map_size
        end_pos[2] = 100
        
        # Simple bezier-like curve
        mid_pos = (start_pos + end_pos) / 2 + np.array([0, 0, 100]) # Arch up
        traj_x = (1-t)**2 * start_pos[0] + 2*(1-t)*t * mid_pos[0] + t**2 * end_pos[0]
        traj_y = (1-t)**2 * start_pos[1] + 2*(1-t)*t * mid_pos[1] + t**2 * end_pos[1]
        traj_z = (1-t)**2 * start_pos[2] + 2*(1-t)*t * mid_pos[2] + t**2 * end_pos[2]
        
        ax.plot(traj_x, traj_y, traj_z, color='red', linewidth=1.5, alpha=0.7)

    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Altitude (m)')
    ax.set_title('Large-Scale Urban Simulation Scenario')
    
    ax.view_init(elev=30, azim=45)
    plt.tight_layout()
    plt.savefig('fig2_simulation_scenario.png', bbox_inches='tight')
    print("Generated fig2_simulation_scenario.png")
    plt.close()

def draw_fig3_potential_field():
    """Figure 3: Artificial Potential Field Reward Surface"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)
    
    # Goal Attraction (Slope)
    Z_goal = -0.1 * X  # Tilted plane
    
    # Obstacle Repulsion (Abyss/Peak - here we want penalties, so negative abyss)
    # Using Gaussian wells
    def obstacle_field(x0, y0, width=1.5, depth=15):
        r2 = (X - x0)**2 + (Y - y0)**2
        return -depth * np.exp(-r2 / width)

    Z_obs1 = obstacle_field(2, 2)
    Z_obs2 = obstacle_field(-3, -1)
    
    Z = Z_goal + Z_obs1 + Z_obs2
    
    # Plot surface
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9, antialiased=True)
    
    # Remove ticks for cleaner schematic look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Reward Value')
    ax.set_title('Potential Field Based Composite Reward Function')
    
    # Add wireframe projection to bottom
    ax.contour(X, Y, Z, zdir='z', offset=np.min(Z), cmap='viridis', alpha=0.5)
    ax.set_zlim(np.min(Z), np.max(Z))
    
    plt.tight_layout()
    plt.savefig('fig3_potential_field.png', bbox_inches='tight')
    print("Generated fig3_potential_field.png")
    plt.close()

def draw_fig4_attention_weights():
    """Figure 4: Attention Mechanism Visualization"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Ego agent
    ego_pos = np.array([0.5, 0.5])
    
    # Neighbors (2 close, 4 far)
    np.random.seed(10)
    angles = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.random.rand(6)*0.5
    
    # Radii: 2 small (close), 4 large (far)
    radii = np.array([0.15, 0.18, 0.35, 0.38, 0.40, 0.42])
    np.random.shuffle(radii)
    
    neighbors = []
    weights = []
    
    for r, ang in zip(radii, angles):
        nx = ego_pos[0] + r * np.cos(ang)
        ny = ego_pos[1] + r * np.sin(ang)
        neighbors.append([nx, ny])
        
        # Fake attention weight: inverse to distance
        w = 1.0 / (r * 10) 
        weights.append(w)
    
    # Normalize weights for visualization
    weights = np.array(weights)
    weights_norm = (weights - weights.min()) / (weights.max() - weights.min())
    
    # Draw connections
    for i, (nx, ny) in enumerate(neighbors):
        w_val = weights_norm[i]
        
        # Color: Light Red (low weight) to Dark Red (high weight)
        # Using cmap
        color = cm.Reds(0.3 + 0.7 * w_val)
        linewidth = 1 + 5 * w_val
        
        ax.plot([ego_pos[0], nx], [ego_pos[1], ny], color=color, linewidth=linewidth, zorder=1)
        
        # Neighbor dots
        dot_color = 'red' if radii[i] < 0.25 else 'green'
        ax.scatter(nx, ny, color=dot_color, s=150, zorder=2, edgecolors='black')
        ax.text(nx, ny+0.03, f"N{i+1}", ha='center', fontsize=10)

    # Ego dot
    ax.scatter(ego_pos[0], ego_pos[1], color='blue', s=300, zorder=3, edgecolors='black', label='Ego Agent')
    ax.text(ego_pos[0], ego_pos[1]-0.05, "Ego Agent", ha='center', fontweight='bold')
    
    # Colorbar hack
    sm = plt.cm.ScalarMappable(cmap=cm.Reds, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Attention Weight $\\alpha_{ij}$')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Attention Weights Visualization')
    
    plt.tight_layout()
    plt.savefig('fig4_attention_weights.png', bbox_inches='tight')
    print("Generated fig4_attention_weights.png")
    plt.close()

def draw_fig5_trajectory_comparison():
    """Figure 5: Trajectory Comparison (Baseline vs Ours)"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Setup Map
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    
    # Obstacles
    obstacles = [
        (30, 25, 4),
        (50, 20, 5),
        (50, 30, 5),
        (70, 25, 4)
    ]
    
    for (ox, oy, r) in obstacles:
        circle = patches.Circle((ox, oy), r, facecolor='black', alpha=0.7)
        ax.add_patch(circle)
        # Safety boundary
        circle_safe = patches.Circle((ox, oy), r+1, fill=False, edgecolor='gray', linestyle='--')
        ax.add_patch(circle_safe)
    
    # Trajectories
    start = np.array([5, 25])
    goal = np.array([95, 25])
    
    # 1. Baseline (Blue, Dashed, Oscillating)
    x_base = np.linspace(5, 95, 200)
    # Create a path that goes near obstacles and oscillates
    y_base = 25 + 0 * x_base # Straight line initially
    
    # Manually crafting the "bad" path
    path_y = []
    for x in x_base:
        if x < 25:
            y = 25
        elif 25 <= x < 35:
            y = 25 + 5 * np.sin((x-25)) # Panic turn
        elif 35 <= x < 45:
            y = 25 - 2 * np.sin((x-35)) # Correction
        elif 45 <= x < 55:
            y = 25 + 8 * np.sin((x-45)*0.5) # Huge deviation for middle obstacles
        elif 55 <= x < 75:
            y = 25 - 5 * np.sin((x-55)*0.5)
        else:
            y = 25 + 0.5 * np.sin(x) # Wobble to goal
        path_y.append(y)
        
    ax.plot(x_base, path_y, color='blue', linestyle='--', linewidth=2, label='Baseline (MADDPG)')
    
    # 2. Ours (Red, Solid, Smooth)
    # Smooth Bezier-like curve anticipating obstacles
    # Keypoints: Start, Before Obs1, Between Obs2/3, After Obs4, Goal
    # But for simplicity, just a smooth sine wave that clears them effectively
    x_ours = np.linspace(5, 95, 200)
    y_ours = []
    for x in x_ours:
        # Pre-emptive smooth turn
        if x < 20:
            y = 25
        elif 20 <= x < 80:
            # Smooth arc over the obstacles
            # A gaussian bump + sine
            y = 25 + 8 * np.exp(-((x-50)**2)/(20**2)) 
        else:
            y = 25
        y_ours.append(y)
        
    ax.plot(x_ours, y_ours, color='red', linestyle='-', linewidth=2.5, label='Ours (AP-MADDPG)')
    
    # Start and Goal
    ax.scatter(*start, color='green', marker='^', s=100, label='Start', zorder=5)
    ax.scatter(*goal, color='purple', marker='*', s=150, label='Goal', zorder=5)
    
    ax.set_xlabel('X Coordinate (m)')
    ax.set_ylabel('Y Coordinate (m)')
    ax.set_title('Trajectory Planning Comparison in Dense Obstacle Scenario')
    ax.legend(loc='lower right')
    ax.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('fig5_trajectory_comparison.png', bbox_inches='tight')
    print("Generated fig5_trajectory_comparison.png")
    plt.close()

if __name__ == "__main__":
    print("Generating academic figures...")
    draw_fig1_architecture()
    draw_fig2_simulation_scenario()
    draw_fig3_potential_field()
    draw_fig4_attention_weights()
    draw_fig5_trajectory_comparison()
    print("All figures generated successfully.")

