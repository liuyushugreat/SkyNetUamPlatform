"""
Swarm Visualization Script for IJCAI 2026 Demo Video.

This script generates high-quality MP4 videos of multi-agent swarm simulations,
specifically designed for visualizing 100,000+ agents with emergent behaviors.

Data Format Requirements:
-------------------------
Option 1: NumPy Array (trajectories.npy)
    Shape: (T, N, 7) where:
        T = number of time steps
        N = number of agents
        7 = [x, y, z, vx, vy, vz, cluster_id] (or [x, y, z, vx, vy, vz, 0] if no cluster)
    
Option 2: CSV File (trajectories.csv)
    Columns: time_step, agent_id, x, y, z, vx, vy, vz, cluster_id (optional)
    
Option 3: Dictionary format (from simulation)
    {
        'positions': np.array (T, N, 3),  # x, y, z
        'velocities': np.array (T, N, 3), # vx, vy, vz
        'cluster_ids': np.array (T, N)    # optional
    }

Usage:
------
    # From NumPy array:
    python visualize_swarm.py --input trajectories.npy --output swarm_demo.mp4
    
    # From CSV:
    python visualize_swarm.py --input trajectories.csv --output swarm_demo.mp4
    
    # With obstacles:
    python visualize_swarm.py --input trajectories.npy --obstacles obstacles.npy --output swarm_demo.mp4
    
    # Custom downsampling (for performance):
    python visualize_swarm.py --input trajectories.npy --max_agents 10000 --output swarm_demo.mp4
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
import pandas as pd
import argparse
import os
from typing import Optional, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# Try to import cv2 for faster rendering (optional)
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Note: OpenCV not available. Using matplotlib animation (slower but works).")


class SwarmVisualizer:
    """
    High-performance swarm visualization for 100k+ agents.
    """
    
    def __init__(self, 
                 trajectories: np.ndarray,
                 obstacles: Optional[np.ndarray] = None,
                 max_agents: int = 10000,
                 fps: int = 30,
                 color_by: str = 'velocity',
                 colormap: str = 'hsv'):
        """
        Initialize visualizer.
        
        Args:
            trajectories: Array of shape (T, N, 7) or (T, N, 6) containing
                         [x, y, z, vx, vy, vz, cluster_id] or [x, y, z, vx, vy, vz]
            obstacles: Array of shape (M, 4) containing [x, y, radius, height] for each obstacle
            max_agents: Maximum number of agents to render (downsampling for performance)
            fps: Frames per second for output video
            color_by: 'velocity' (direction/magnitude) or 'cluster' (cluster ID)
            colormap: Matplotlib colormap name ('hsv', 'viridis', 'plasma', etc.)
        """
        self.trajectories = trajectories
        self.obstacles = obstacles
        self.max_agents = max_agents
        self.fps = fps
        self.color_by = color_by
        self.colormap = colormap
        
        # Extract dimensions
        self.T, self.N, self.dim = trajectories.shape
        print(f"Loaded trajectories: {self.T} time steps, {self.N} agents, {self.dim} dimensions")
        
        # Downsample if needed
        if self.N > max_agents:
            print(f"Downsampling from {self.N} to {max_agents} agents for visualization...")
            self.agent_indices = np.linspace(0, self.N - 1, max_agents, dtype=int)
            self.trajectories = self.trajectories[:, self.agent_indices, :]
            self.N = max_agents
        else:
            self.agent_indices = np.arange(self.N)
        
        # Extract components
        self.positions = self.trajectories[:, :, :3]  # (T, N, 3) - x, y, z
        if self.dim >= 6:
            self.velocities = self.trajectories[:, :, 3:6]  # (T, N, 3) - vx, vy, vz
        else:
            # Compute velocities from positions
            self.velocities = np.diff(self.positions, axis=0, prepend=self.positions[0:1])
        
        # Cluster IDs (if available)
        if self.dim >= 7:
            self.cluster_ids = self.trajectories[:, :, 6].astype(int)
        else:
            self.cluster_ids = None
        
        # Compute velocity magnitudes and directions
        self.velocity_magnitudes = np.linalg.norm(self.velocities, axis=2)  # (T, N)
        self.velocity_directions = np.arctan2(self.velocities[:, :, 1], 
                                             self.velocities[:, :, 0])  # (T, N) - angle in radians
        
        # Setup figure
        self.fig, self.ax = plt.subplots(figsize=(16, 9), facecolor='black')
        self.ax.set_facecolor('black')
        self.ax.set_aspect('equal')
        
        # Set axis limits based on data
        x_min, x_max = self.positions[:, :, 0].min(), self.positions[:, :, 0].max()
        y_min, y_max = self.positions[:, :, 1].min(), self.positions[:, :, 1].max()
        margin = 0.05 * max(x_max - x_min, y_max - y_min)
        self.ax.set_xlim(x_min - margin, x_max + margin)
        self.ax.set_ylim(y_min - margin, y_max + margin)
        
        # Draw obstacles (static background)
        if obstacles is not None:
            self._draw_obstacles()
        
        # Initialize scatter plot for agents
        self.scatter = None
        self.time_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes,
                                      fontsize=14, color='white', verticalalignment='top',
                                      bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        # Statistics text
        self.stats_text = self.ax.text(0.98, 0.98, '', transform=self.ax.transAxes,
                                       fontsize=12, color='white', verticalalignment='top',
                                       horizontalalignment='right',
                                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    def _draw_obstacles(self):
        """Draw static obstacles as gray circles."""
        if self.obstacles is None:
            return
        
        for obs in self.obstacles:
            x, y, radius, height = obs[:4]
            circle = plt.Circle((x, y), radius, color='gray', alpha=0.6, zorder=0)
            self.ax.add_patch(circle)
    
    def _get_colors(self, t: int) -> np.ndarray:
        """
        Get colors for agents at time step t.
        
        Args:
            t: Time step index
            
        Returns:
            Array of shape (N, 4) RGBA colors
        """
        if self.color_by == 'velocity':
            # Color by velocity direction (hue) and magnitude (brightness)
            angles = self.velocity_directions[t]  # (N,)
            magnitudes = self.velocity_magnitudes[t]  # (N,)
            
            # Normalize angles to [0, 1] for colormap
            angles_norm = (angles + np.pi) / (2 * np.pi)  # [-pi, pi] -> [0, 1]
            
            # Normalize magnitudes to [0.3, 1.0] for brightness (avoid too dark)
            if magnitudes.max() > 0:
                mag_norm = 0.3 + 0.7 * (magnitudes / magnitudes.max())
            else:
                mag_norm = np.ones_like(magnitudes) * 0.5
            
            # Get colormap
            cmap = plt.get_cmap(self.colormap)
            colors = cmap(angles_norm)
            colors[:, 3] = mag_norm  # Set alpha based on magnitude
            
        elif self.color_by == 'cluster':
            if self.cluster_ids is not None:
                # Color by cluster ID
                cluster_ids = self.cluster_ids[t]
                unique_clusters = np.unique(cluster_ids)
                n_clusters = len(unique_clusters)
                
                cmap = plt.get_cmap(self.colormap)
                cluster_colors = cmap(np.linspace(0, 1, n_clusters))
                
                # Map cluster IDs to colors
                cluster_to_color = {cid: cluster_colors[i] for i, cid in enumerate(unique_clusters)}
                colors = np.array([cluster_to_color.get(cid, [0.5, 0.5, 0.5, 1.0]) 
                                  for cid in cluster_ids])
            else:
                # Fallback to uniform color
                colors = np.ones((self.N, 4)) * [0.5, 0.5, 0.5, 1.0]
        else:
            # Default: uniform color
            colors = np.ones((self.N, 4)) * [1.0, 0.0, 0.0, 1.0]  # Red
        
        return colors
    
    def _update_frame(self, t: int):
        """Update frame for animation."""
        # Get positions at time t (project to 2D: x, y)
        pos_2d = self.positions[t, :, :2]  # (N, 2)
        
        # Get colors
        colors = self._get_colors(t)
        
        # Update scatter plot
        if self.scatter is None:
            self.scatter = self.ax.scatter(pos_2d[:, 0], pos_2d[:, 1], 
                                          s=2, c=colors, alpha=0.8, 
                                          edgecolors='none', zorder=10)
        else:
            self.scatter.set_offsets(pos_2d)
            self.scatter.set_color(colors)
        
        # Update time text
        self.time_text.set_text(f'Time Step: {t}/{self.T-1}')
        
        # Update statistics
        avg_speed = self.velocity_magnitudes[t].mean()
        max_speed = self.velocity_magnitudes[t].max()
        if self.cluster_ids is not None:
            n_clusters = len(np.unique(self.cluster_ids[t]))
            stats_str = f'Avg Speed: {avg_speed:.2f}\nMax Speed: {max_speed:.2f}\nClusters: {n_clusters}'
        else:
            stats_str = f'Avg Speed: {avg_speed:.2f}\nMax Speed: {max_speed:.2f}'
        self.stats_text.set_text(stats_str)
        
        return [self.scatter, self.time_text, self.stats_text]
    
    def animate(self, output_path: str, dpi: int = 150):
        """
        Create and save animation.
        
        Args:
            output_path: Path to save MP4 file
            dpi: Resolution (dots per inch)
        """
        print(f"Creating animation ({self.T} frames)...")
        
        # Create animation
        anim = animation.FuncAnimation(
            self.fig, 
            self._update_frame,
            frames=self.T,
            interval=1000 / self.fps,
            blit=True,
            repeat=True
        )
        
        # Save to MP4
        print(f"Saving to {output_path}...")
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=self.fps, bitrate=1800, codec='libx264')
        
        anim.save(output_path, writer=writer, dpi=dpi)
        print(f"Video saved to {output_path}")
        
        plt.close(self.fig)


def load_trajectories_numpy(filepath: str) -> np.ndarray:
    """Load trajectories from NumPy file."""
    data = np.load(filepath)
    
    # Handle different shapes
    if data.ndim == 3:
        # Already in (T, N, D) format
        return data
    elif data.ndim == 2:
        # Assume (T*N, D) format - need to reshape
        # This requires knowing T or N - for now, assume square-ish
        total = data.shape[0]
        dim = data.shape[1]
        # Try to infer T and N (assume T ≈ N for square-ish)
        T = int(np.sqrt(total))
        N = total // T
        if T * N == total:
            return data.reshape(T, N, dim)
        else:
            raise ValueError(f"Cannot auto-reshape 2D array of shape {data.shape}. "
                           f"Please provide (T, N, D) format.")
    else:
        raise ValueError(f"Unsupported array shape: {data.shape}. Expected (T, N, D) or (T*N, D).")


def load_trajectories_csv(filepath: str) -> np.ndarray:
    """
    Load trajectories from CSV file.
    
    Expected columns: time_step, agent_id, x, y, z, vx, vy, vz, [cluster_id]
    """
    df = pd.read_csv(filepath)
    
    # Get unique time steps and agents
    time_steps = sorted(df['time_step'].unique())
    agent_ids = sorted(df['agent_id'].unique())
    
    T = len(time_steps)
    N = len(agent_ids)
    
    # Determine dimensions
    has_velocity = all(col in df.columns for col in ['vx', 'vy', 'vz'])
    has_cluster = 'cluster_id' in df.columns
    
    if has_velocity and has_cluster:
        dim = 7  # x, y, z, vx, vy, vz, cluster_id
    elif has_velocity:
        dim = 6  # x, y, z, vx, vy, vz
    else:
        dim = 3  # x, y, z only
    
    # Initialize array
    trajectories = np.zeros((T, N, dim))
    
    # Fill array
    for t_idx, t in enumerate(time_steps):
        t_data = df[df['time_step'] == t].sort_values('agent_id')
        for a_idx, agent_id in enumerate(agent_ids):
            agent_data = t_data[t_data['agent_id'] == agent_id]
            if len(agent_data) > 0:
                row = agent_data.iloc[0]
                trajectories[t_idx, a_idx, 0] = row['x']
                trajectories[t_idx, a_idx, 1] = row['y']
                trajectories[t_idx, a_idx, 2] = row.get('z', 0.0)
                
                if has_velocity:
                    trajectories[t_idx, a_idx, 3] = row['vx']
                    trajectories[t_idx, a_idx, 4] = row['vy']
                    trajectories[t_idx, a_idx, 5] = row['vz']
                
                if has_cluster:
                    trajectories[t_idx, a_idx, 6] = row.get('cluster_id', 0)
    
    return trajectories


def load_obstacles(filepath: str) -> np.ndarray:
    """
    Load obstacles from file.
    
    Format: NumPy array of shape (M, 4) where each row is [x, y, radius, height]
    Or CSV with columns: x, y, radius, height
    """
    if filepath.endswith('.npy'):
        return np.load(filepath)
    elif filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
        return df[['x', 'y', 'radius', 'height']].values
    else:
        raise ValueError(f"Unsupported obstacle file format: {filepath}")


def main():
    parser = argparse.ArgumentParser(description='Visualize swarm simulation for IJCAI demo video')
    parser.add_argument('--input', type=str, required=True,
                       help='Input file: trajectories.npy or trajectories.csv')
    parser.add_argument('--output', type=str, default='swarm_demo.mp4',
                       help='Output MP4 file path')
    parser.add_argument('--obstacles', type=str, default=None,
                       help='Obstacles file (optional): obstacles.npy or obstacles.csv')
    parser.add_argument('--max_agents', type=int, default=10000,
                       help='Maximum number of agents to render (downsampling, default: 10000)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Frames per second (default: 30)')
    parser.add_argument('--color_by', type=str, default='velocity',
                       choices=['velocity', 'cluster'],
                       help='Color coding: velocity (direction/magnitude) or cluster (cluster ID)')
    parser.add_argument('--colormap', type=str, default='hsv',
                       help='Matplotlib colormap name (default: hsv)')
    parser.add_argument('--dpi', type=int, default=150,
                       help='Resolution in DPI (default: 150)')
    
    args = parser.parse_args()
    
    # Load trajectories
    print(f"Loading trajectories from {args.input}...")
    if args.input.endswith('.npy'):
        trajectories = load_trajectories_numpy(args.input)
    elif args.input.endswith('.csv'):
        trajectories = load_trajectories_csv(args.input)
    else:
        raise ValueError(f"Unsupported input format: {args.input}")
    
    # Load obstacles (if provided)
    obstacles = None
    if args.obstacles:
        print(f"Loading obstacles from {args.obstacles}...")
        obstacles = load_obstacles(args.obstacles)
    
    # Create visualizer
    visualizer = SwarmVisualizer(
        trajectories=trajectories,
        obstacles=obstacles,
        max_agents=args.max_agents,
        fps=args.fps,
        color_by=args.color_by,
        colormap=args.colormap
    )
    
    # Generate animation
    visualizer.animate(args.output, dpi=args.dpi)
    
    print(f"\nVisualization complete!")
    print(f"  Output: {args.output}")
    print(f"  Frames: {visualizer.T}")
    print(f"  Agents rendered: {visualizer.N} (downsampled from {trajectories.shape[1]})")


if __name__ == '__main__':
    main()

