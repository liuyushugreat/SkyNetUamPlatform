"""
Metrics Logger for Multi-Agent RL Training.

Logs training metrics including:
- Standard RL metrics (reward, loss, etc.)
- Emergence metrics (velocity alignment, cluster entropy)
- Performance metrics (collision rate, success rate)

Supports both TensorBoard and CSV logging.
"""

import os
import csv
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: TensorBoard not available. CSV logging only.")


class MetricsLogger:
    """
    Unified metrics logger for training and evaluation.
    """
    def __init__(self, log_dir: str, use_tensorboard: bool = True):
        """
        Args:
            log_dir: Directory to save logs
            use_tensorboard: Whether to use TensorBoard (if available)
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.use_tensorboard = use_tensorboard and TENSORBOARD_AVAILABLE
        if self.use_tensorboard:
            self.writer = SummaryWriter(log_dir=os.path.join(log_dir, 'tensorboard'))
        else:
            self.writer = None
        
        # CSV file for metrics
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(log_dir, f'metrics_{timestamp}.csv')
        
        # Initialize CSV with headers
        self.csv_headers = [
            'episode', 'step', 'algorithm',
            'reward_mean', 'reward_std', 'reward_min', 'reward_max',
            'collision_rate', 'success_rate', 'avg_episode_length',
            'actor_loss', 'critic_loss', 'entropy',
            'order_parameter', 'local_alignment', 'alignment_std',
            'cluster_entropy', 'num_clusters', 'largest_cluster_ratio',
            'avg_cluster_size'
        ]
        
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_headers)
            writer.writeheader()
        
        self.episode_count = 0
        self.step_count = 0
    
    def log_training_step(self, metrics: Dict[str, float], algorithm: str = 'MAPPO'):
        """
        Log metrics from a training step.
        
        Args:
            metrics: Dictionary of metric names to values
            algorithm: Algorithm name (e.g., 'MAPPO', 'AP-MADDPG')
        """
        self.step_count += 1
        
        # Write to CSV
        row = {
            'episode': self.episode_count,
            'step': self.step_count,
            'algorithm': algorithm,
            **{k: metrics.get(k, 0.0) for k in self.csv_headers[3:]}
        }
        
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_headers)
            writer.writerow(row)
        
        # Write to TensorBoard
        if self.writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f'{algorithm}/{key}', value, self.step_count)
    
    def log_episode(self, episode_metrics: Dict[str, float], algorithm: str = 'MAPPO'):
        """
        Log metrics from a complete episode.
        
        Args:
            episode_metrics: Dictionary of episode-level metrics
            algorithm: Algorithm name
        """
        self.episode_count += 1
        
        # Add episode prefix to metrics
        tb_metrics = {f'episode/{k}': v for k, v in episode_metrics.items()}
        
        if self.writer:
            for key, value in tb_metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f'{algorithm}/{key}', value, self.episode_count)
    
    def log_evaluation(self, eval_metrics: Dict[str, float], algorithm: str = 'MAPPO'):
        """
        Log evaluation metrics (typically computed over multiple episodes).
        
        Args:
            eval_metrics: Dictionary of evaluation metrics
            algorithm: Algorithm name
        """
        if self.writer:
            for key, value in eval_metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f'{algorithm}/eval/{key}', value, self.episode_count)
    
    def log_emergence_metrics(self, emergence_metrics: Dict[str, float], 
                              step: Optional[int] = None, algorithm: str = 'MAPPO'):
        """
        Log emergence metrics specifically.
        
        Args:
            emergence_metrics: Dictionary from calculate_emergence_metrics()
            step: Step number (uses self.step_count if None)
            algorithm: Algorithm name
        """
        if step is None:
            step = self.step_count
        
        if self.writer:
            for key, value in emergence_metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f'{algorithm}/emergence/{key}', value, step)
    
    def log_histogram(self, name: str, values: np.ndarray, step: Optional[int] = None):
        """
        Log histogram of values.
        
        Args:
            name: Name of the histogram
            values: Array of values
            step: Step number
        """
        if step is None:
            step = self.step_count
        
        if self.writer:
            self.writer.add_histogram(name, values, step)
    
    def flush(self):
        """Flush all pending writes."""
        if self.writer:
            self.writer.flush()
    
    def close(self):
        """Close logger and flush all writes."""
        if self.writer:
            self.writer.close()
        self.flush()

