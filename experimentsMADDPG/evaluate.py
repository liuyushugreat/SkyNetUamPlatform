import time
import numpy as np
import pandas as pd
import os
from typing import Dict, List

# Import metrics
from emergence_metrics import calculate_emergence_metrics, calculate_cluster_entropy

class NumpyActor:
    """
    Simple MLP Actor implemented in NumPy for benchmarking inference latency.
    Architecture matches the PyTorch ActorNetwork.
    """
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Initialize weights (randomly)
        self.W1 = np.random.randn(obs_dim, hidden_dim).astype(np.float32) * 0.1
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        
        self.W2 = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * 0.1
        self.b2 = np.zeros(hidden_dim, dtype=np.float32)
        
        self.W3 = np.random.randn(hidden_dim, action_dim).astype(np.float32) * 0.1
        self.b3 = np.zeros(action_dim, dtype=np.float32)
        
    def forward(self, x):
        """
        Forward pass for a batch of agents.
        x: (batch_size, obs_dim)
        """
        # Layer 1
        z1 = np.dot(x, self.W1) + self.b1
        a1 = np.maximum(0, z1) # ReLU
        
        # Layer 2
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = np.maximum(0, z2) # ReLU
        
        # Output Layer
        z3 = np.dot(a2, self.W3) + self.b3
        a3 = 1.0 / (1.0 + np.exp(-z3)) # Sigmoid
        
        return a3

def measure_inference_latency(num_agents: int, 
                              obs_dim: int = 18, 
                              action_dim: int = 3, 
                              hidden_dim: int = 64,
                              num_steps: int = 50) -> float:
    """
    Measure average inference time per step for N agents using NumPy.
    
    Args:
        num_agents: Number of agents
        obs_dim: Observation dimension
        action_dim: Action dimension
        hidden_dim: Hidden layer dimension
        num_steps: Number of steps to average over
        
    Returns:
        Average inference time in milliseconds
    """
    print(f"Benchmarking N={num_agents} (NumPy)...")
    
    # Instantiate Actor
    actor = NumpyActor(obs_dim, action_dim, hidden_dim)
    
    # Generate dummy observations for all agents
    # Shape: (num_agents, obs_dim)
    obs = np.random.randn(num_agents, obs_dim).astype(np.float32)
    
    # Warmup
    for _ in range(5):
        _ = actor.forward(obs)
            
    # Benchmark
    start_time = time.time()
    
    for _ in range(num_steps):
        _ = actor.forward(obs)
            
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time_ms = (total_time / num_steps) * 1000.0
    
    print(f"  Average Inference Time: {avg_time_ms:.4f} ms")
    return avg_time_ms

def verify_metrics_implementation(num_agents: int = 2000):
    """
    Verify Cluster Entropy and other emergence metrics on dummy data.
    Uses smaller N (e.g. 2000) to keep pdist fast.
    """
    print(f"\nVerifying Emergence Metrics (N={num_agents})...")
    
    # Generate random states: [x, y, z, vx, vy, vz]
    # Simulate some clustering
    states = np.random.rand(num_agents, 6) * 1000.0 # 1000m x 1000m
    
    # Create artificial clusters
    for i in range(5):
        center = np.random.rand(3) * 1000.0
        indices = np.random.choice(num_agents, num_agents // 10, replace=False)
        states[indices, :3] = center + np.random.randn(len(indices), 3) * 20.0
    
    # 1. Velocity Alignment
    # Make velocities aligned in clusters
    states[:, 3:6] = np.random.randn(num_agents, 3) # Random velocities
    
    start = time.time()
    metrics = calculate_emergence_metrics(states, neighbor_radius=100.0, cluster_eps=50.0)
    end = time.time()
    
    print(f"  Metrics Calculation Time: {(end - start)*1000:.2f} ms")
    print("  Calculated Metrics:")
    for k, v in metrics.items():
        print(f"    {k}: {v:.4f}")
        
    return metrics

def run_scalability_benchmark():
    """
    Run the full scalability benchmark and save results.
    """
    agent_counts = [10000, 50000, 100000]
    results = {
        'num_agents': [],
        'inference_time_ms': []
    }
    
    for n in agent_counts:
        latency = measure_inference_latency(n)
        results['num_agents'].append(n)
        results['inference_time_ms'].append(latency)
        
    # Save results
    df = pd.DataFrame(results)
    df.to_csv('scalability_results.csv', index=False)
    print("\nScalability results saved to scalability_results.csv")
    
    return df

if __name__ == "__main__":
    # 1. Verify Metrics Implementation
    verify_metrics_implementation()
    
    # 2. Run Latency Benchmark
    run_scalability_benchmark()
