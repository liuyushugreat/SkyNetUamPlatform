"""
Helper script to generate trajectory data from simulation for visualization.

This script demonstrates how to log agent positions and velocities during
simulation to create the input files needed for visualize_swarm.py.

Usage:
    python generate_trajectory_data.py --num_agents 10000 --num_steps 500 --output trajectories.npy
"""

import numpy as np
import argparse
import sys
import os
from typing import Tuple

# Add repo root directory to sys.path for imports.
# This script lives at: <repo_root>/PaperExperiments/experimentsMADDPG/generate_trajectory_data.py
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(repo_root)

from nexus_core.mas.environment import SkyNetEnv
from PaperExperiments.experimentsMADDPG.emergence_metrics import calculate_cluster_entropy


def simulate_and_save_trajectories(num_agents: int = 1000,
                                   num_steps: int = 500,
                                   output_npy: str = 'trajectories.npy',
                                   output_csv: str = None,
                                   save_clusters: bool = True):
    """
    Run a simulation and save trajectory data.
    
    Args:
        num_agents: Number of agents
        num_steps: Number of simulation steps
        output_npy: Output NumPy file path
        output_csv: Optional CSV output path
        save_clusters: Whether to compute and save cluster IDs
    """
    print(f"Simulating {num_agents} agents for {num_steps} steps...")
    
    # Create environment
    env = SkyNetEnv(num_agents=num_agents)
    obs = env.reset()
    
    # Initialize trajectory storage
    # Format: (T, N, 7) = [x, y, z, vx, vy, vz, cluster_id]
    trajectories = np.zeros((num_steps, num_agents, 7))
    
    # Mock policy: random actions (replace with your trained policy)
    def get_action(obs_dict):
        """Simple random policy - replace with your trained agent."""
        actions = {}
        for agent_id in obs_dict.keys():
            # Random velocity ratios and bid
            actions[agent_id] = np.random.rand(4).astype(np.float32)
        return actions
    
    # Run simulation
    for t in range(num_steps):
        # Get actions (replace with your policy)
        actions = get_action(obs)
        
        # Step environment
        next_obs, rewards, dones, infos = env.step(actions)
        
        # Extract positions and velocities from observations
        agent_keys = sorted(obs.keys())
        for i, agent_id in enumerate(agent_keys):
            agent_obs = obs[agent_id]
            
            # Observation format: [x, y, z, vx, vy, vz, bat, price, dest_x, dest_y]
            if len(agent_obs) >= 6:
                trajectories[t, i, 0] = agent_obs[0]  # x
                trajectories[t, i, 1] = agent_obs[1]  # y
                trajectories[t, i, 2] = agent_obs[2]  # z
                trajectories[t, i, 3] = agent_obs[3]  # vx
                trajectories[t, i, 4] = agent_obs[4]  # vy
                trajectories[t, i, 5] = agent_obs[5]  # vz
            else:
                # Fallback: use random data if observation is incomplete
                trajectories[t, i, :3] = np.random.randn(3) * 100
                trajectories[t, i, 3:6] = np.random.randn(3) * 10
        
        # Compute cluster IDs if requested
        if save_clusters and t % 10 == 0:  # Compute every 10 steps for efficiency
            try:
                # Extract states for clustering
                states = trajectories[t, :, :6].reshape(1, num_agents, 6)
                # Calculate cluster entropy (which also computes clusters)
                try:
                    result = calculate_cluster_entropy(
                        states, eps=50.0, min_samples=5
                    )
                    # Handle different return formats
                    if isinstance(result, tuple):
                        cluster_entropy, cluster_info = result
                    else:
                        cluster_info = result if isinstance(result, dict) else {}
                    
                    # Assign cluster IDs (if available)
                    if 'cluster_labels' in cluster_info:
                        cluster_labels = cluster_info['cluster_labels']
                        trajectories[t, :, 6] = cluster_labels
                    else:
                        trajectories[t, :, 6] = 0  # No clusters
                except Exception as e:
                    trajectories[t, :, 6] = 0  # No clusters
            except Exception as e:
                print(f"Warning: Could not compute clusters at step {t}: {e}")
                trajectories[t, :, 6] = 0
        
        obs = next_obs
        
        if t % 50 == 0:
            print(f"  Step {t}/{num_steps}")
    
    # Save NumPy file
    print(f"Saving trajectories to {output_npy}...")
    np.save(output_npy, trajectories)
    print(f"Saved {trajectories.shape} array to {output_npy}")
    
    # Save CSV if requested
    if output_csv:
        print(f"Saving CSV to {output_csv}...")
        save_trajectories_csv(trajectories, output_csv)
        print(f"Saved CSV to {output_csv}")
    
    return trajectories


def save_trajectories_csv(trajectories: np.ndarray, output_path: str):
    """
    Save trajectories to CSV format.
    
    Args:
        trajectories: Array of shape (T, N, 7)
        output_path: Output CSV file path
    """
    T, N, D = trajectories.shape
    
    rows = []
    for t in range(T):
        for agent_id in range(N):
            row = {
                'time_step': t,
                'agent_id': agent_id,
                'x': trajectories[t, agent_id, 0],
                'y': trajectories[t, agent_id, 1],
                'z': trajectories[t, agent_id, 2],
                'vx': trajectories[t, agent_id, 3],
                'vy': trajectories[t, agent_id, 4],
                'vz': trajectories[t, agent_id, 5],
            }
            if D >= 7:
                row['cluster_id'] = int(trajectories[t, agent_id, 6])
            rows.append(row)
    
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)


def generate_sample_obstacles(num_obstacles: int = 2000,
                              bounds: Tuple[float, float, float, float] = (-50000, 50000, -50000, 50000),
                              output_path: str = 'obstacles.npy'):
    """
    Generate sample obstacles for visualization.
    
    Args:
        num_obstacles: Number of obstacles
        bounds: (x_min, x_max, y_min, y_max)
        output_path: Output file path
    """
    x_min, x_max, y_min, y_max = bounds
    
    obstacles = []
    for _ in range(num_obstacles):
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        radius = np.random.uniform(50, 500)  # Obstacle radius
        height = np.random.uniform(50, 200)  # Obstacle height
        obstacles.append([x, y, radius, height])
    
    obstacles = np.array(obstacles)
    np.save(output_path, obstacles)
    print(f"Generated {num_obstacles} obstacles saved to {output_path}")
    
    return obstacles


def main():
    parser = argparse.ArgumentParser(description='Generate trajectory data from simulation')
    parser.add_argument('--num_agents', type=int, default=1000,
                       help='Number of agents (default: 1000)')
    parser.add_argument('--num_steps', type=int, default=500,
                       help='Number of simulation steps (default: 500)')
    parser.add_argument('--output', type=str, default='trajectories.npy',
                       help='Output NumPy file (default: trajectories.npy)')
    parser.add_argument('--output_csv', type=str, default=None,
                       help='Optional CSV output file')
    parser.add_argument('--obstacles', type=str, default=None,
                       help='Generate obstacles file (default: None)')
    parser.add_argument('--num_obstacles', type=int, default=2000,
                       help='Number of obstacles to generate (default: 2000)')
    parser.add_argument('--no_clusters', action='store_true',
                       help='Skip cluster ID computation (faster)')
    
    args = parser.parse_args()
    
    # Generate trajectory data
    trajectories = simulate_and_save_trajectories(
        num_agents=args.num_agents,
        num_steps=args.num_steps,
        output_npy=args.output,
        output_csv=args.output_csv,
        save_clusters=not args.no_clusters
    )
    
    # Generate obstacles if requested
    if args.obstacles:
        generate_sample_obstacles(
            num_obstacles=args.num_obstacles,
            output_path=args.obstacles
        )
    
    print("\nData generation complete!")
    print(f"\nNext steps:")
    print(f"  1. Visualize: python visualize_swarm.py --input {args.output} --output swarm_demo.mp4")
    if args.obstacles:
        print(f"  2. With obstacles: python visualize_swarm.py --input {args.output} --obstacles {args.obstacles} --output swarm_demo.mp4")


if __name__ == '__main__':
    main()

