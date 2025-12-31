# Swarm Visualization Guide for IJCAI 2026 Demo Video

This guide explains how to generate high-quality video demonstrations of your 100,000-agent swarm simulations.

## Quick Start

### 1. Generate Trajectory Data

If you don't have trajectory data yet, use the helper script:

```bash
# Generate trajectory data from simulation
python generate_trajectory_data.py --num_agents 10000 --num_steps 500 --output trajectories.npy

# Optionally generate obstacles
python generate_trajectory_data.py --num_agents 10000 --num_steps 500 --output trajectories.npy --obstacles obstacles.npy --num_obstacles 2000
```

### 2. Create Video

```bash
# Basic visualization
python visualize_swarm.py --input trajectories.npy --output swarm_demo.mp4

# With obstacles and custom settings
python visualize_swarm.py \
    --input trajectories.npy \
    --obstacles obstacles.npy \
    --output swarm_demo.mp4 \
    --max_agents 10000 \
    --fps 30 \
    --color_by velocity \
    --colormap hsv
```

## Data Format Requirements

### Option 1: NumPy Array (Recommended)

**File:** `trajectories.npy`

**Shape:** `(T, N, 7)` where:
- `T` = number of time steps
- `N` = number of agents
- `7` = `[x, y, z, vx, vy, vz, cluster_id]`

**Example:**
```python
import numpy as np

# Create sample data
T, N = 500, 10000
trajectories = np.zeros((T, N, 7))

# Fill with your simulation data
for t in range(T):
    for agent_id in range(N):
        trajectories[t, agent_id, 0] = x_position
        trajectories[t, agent_id, 1] = y_position
        trajectories[t, agent_id, 2] = z_position
        trajectories[t, agent_id, 3] = vx_velocity
        trajectories[t, agent_id, 4] = vy_velocity
        trajectories[t, agent_id, 5] = vz_velocity
        trajectories[t, agent_id, 6] = cluster_id  # Optional

np.save('trajectories.npy', trajectories)
```

### Option 2: CSV File

**File:** `trajectories.csv`

**Columns:** `time_step, agent_id, x, y, z, vx, vy, vz, cluster_id`

**Example:**
```csv
time_step,agent_id,x,y,z,vx,vy,vz,cluster_id
0,0,100.5,200.3,50.0,5.2,-3.1,0.1,0
0,1,150.2,180.7,45.0,4.8,-2.9,0.0,0
1,0,105.7,197.2,50.1,5.2,-3.1,0.1,0
1,1,155.0,177.8,45.0,4.8,-2.9,0.0,0
...
```

### Option 3: From Your Simulation

If you're running AP-MADDPG or MAPPO training, modify your training loop to log trajectories:

```python
import numpy as np

# During training/evaluation
trajectories = []
for episode in range(num_episodes):
    obs = env.reset()
    episode_trajectory = []
    
    for step in range(max_steps):
        actions = get_actions(obs)  # Your policy
        next_obs, rewards, dones, infos = env.step(actions)
        
        # Extract positions and velocities
        step_data = []
        for agent_id in sorted(obs.keys()):
            agent_obs = obs[agent_id]
            # Format: [x, y, z, vx, vy, vz, cluster_id]
            step_data.append([
                agent_obs[0],  # x
                agent_obs[1],  # y
                agent_obs[2],  # z
                agent_obs[3],  # vx
                agent_obs[4],  # vy
                agent_obs[5],  # vz
                0  # cluster_id (compute separately if needed)
            ])
        
        episode_trajectory.append(step_data)
        obs = next_obs
    
    trajectories.append(np.array(episode_trajectory))

# Save
trajectories = np.array(trajectories)  # Shape: (episodes, steps, agents, 7)
# For single episode:
np.save('trajectories.npy', trajectories[0])  # Shape: (steps, agents, 7)
```

## Obstacles Format

**File:** `obstacles.npy` or `obstacles.csv`

**Format:** Array of shape `(M, 4)` where each row is `[x, y, radius, height]`

**Example:**
```python
obstacles = np.array([
    [1000.0, 2000.0, 500.0, 100.0],  # Obstacle at (1000, 2000) with radius 500, height 100
    [5000.0, 3000.0, 300.0, 150.0],
    # ... more obstacles
])
np.save('obstacles.npy', obstacles)
```

## Visualization Options

### Color Coding

**By Velocity (Default):**
- Hue = direction (angle in radians)
- Brightness = magnitude (speed)
- Shows emergent flow patterns

```bash
python visualize_swarm.py --input trajectories.npy --color_by velocity --colormap hsv
```

**By Cluster:**
- Each cluster gets a unique color
- Shows spatial grouping behavior

```bash
python visualize_swarm.py --input trajectories.npy --color_by cluster --colormap tab20
```

### Performance Optimization

For 100k agents, the script automatically downsamples to `--max_agents` (default: 10,000) for rendering performance. You can adjust this:

```bash
# Render all agents (slower)
python visualize_swarm.py --input trajectories.npy --max_agents 100000

# Render fewer for faster generation
python visualize_swarm.py --input trajectories.npy --max_agents 5000
```

### Output Quality

```bash
# High quality (slower, larger file)
python visualize_swarm.py --input trajectories.npy --dpi 300 --fps 60

# Standard quality (faster, smaller file)
python visualize_swarm.py --input trajectories.npy --dpi 150 --fps 30
```

## Complete Example

```bash
# 1. Generate trajectory data (if needed)
python generate_trajectory_data.py \
    --num_agents 10000 \
    --num_steps 500 \
    --output trajectories.npy \
    --obstacles obstacles.npy \
    --num_obstacles 2000

# 2. Create video with velocity coloring
python visualize_swarm.py \
    --input trajectories.npy \
    --obstacles obstacles.npy \
    --output swarm_demo_velocity.mp4 \
    --max_agents 10000 \
    --fps 30 \
    --color_by velocity \
    --colormap hsv \
    --dpi 150

# 3. Create video with cluster coloring
python visualize_swarm.py \
    --input trajectories.npy \
    --obstacles obstacles.npy \
    --output swarm_demo_clusters.mp4 \
    --max_agents 10000 \
    --fps 30 \
    --color_by cluster \
    --colormap tab20 \
    --dpi 150
```

## Troubleshooting

### "FFmpeg not found"
Install FFmpeg:
- **Windows:** Download from https://ffmpeg.org/download.html
- **Linux:** `sudo apt-get install ffmpeg`
- **Mac:** `brew install ffmpeg`

### "Out of Memory"
- Reduce `--max_agents` (e.g., `--max_agents 5000`)
- Reduce `--dpi` (e.g., `--dpi 100`)
- Process shorter time sequences

### "Video is too slow/fast"
- Adjust `--fps` (default: 30)
- Higher FPS = smoother but larger file
- Lower FPS = smaller file but choppier

### "Colors look wrong"
- Try different colormaps: `hsv`, `viridis`, `plasma`, `tab20`
- For velocity: `hsv` works best (circular color space)
- For clusters: `tab20` or `Set3` work well

## Tips for IJCAI Submission

1. **Show Emergent Behavior**: Use `--color_by velocity` to highlight coordinated motion
2. **Highlight Scalability**: Use `--max_agents 100000` to show all agents (if feasible)
3. **Include Obstacles**: Add `--obstacles` to show constraint navigation
4. **High Quality**: Use `--dpi 300 --fps 60` for presentation
5. **Multiple Views**: Create videos with different color schemes to show different aspects

## File Sizes

Approximate output file sizes:
- 500 steps, 10k agents, 30fps, 150dpi: ~50-100 MB
- 500 steps, 10k agents, 60fps, 300dpi: ~200-400 MB
- 1000 steps, 100k agents (downsampled to 10k), 30fps, 150dpi: ~100-200 MB

