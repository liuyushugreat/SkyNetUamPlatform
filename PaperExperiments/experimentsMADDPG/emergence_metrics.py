"""
Emergence Metrics for Multi-Agent Swarm Systems.

This module implements metrics to quantify "emergent behavior" in large-scale
multi-agent systems, which is crucial for IJCAI-level publications.

Metrics:
1. Order Parameter / Velocity Alignment: Measures how well agents align their
   velocities with neighbors (indicator of coordinated motion).
2. Cluster Entropy: Measures spatial clustering/grouping behavior.

Reference:
    Vicsek, T., Czirók, A., Ben-Jacob, E., et al. (1995).
    Novel type of phase transition in a system of self-driven particles.
    Physical Review Letters, 75(6), 1226.
"""

import numpy as np
from typing import List, Dict, Tuple
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def calculate_velocity_alignment(states: np.ndarray, 
                                 neighbor_radius: float = 50.0,
                                 velocity_indices: Tuple[int, int, int] = (3, 4, 5)) -> Dict[str, float]:
    """
    Calculate velocity alignment (Order Parameter) for agent swarm.
    
    The order parameter measures the degree of velocity alignment:
        φ = (1/N) * |Σ_i v_i / |v_i||
    
    Values range from 0 (random motion) to 1 (perfect alignment).
    
    Args:
        states: Agent states array (N, state_dim)
                Expected format: [x, y, z, vx, vy, vz, ...]
        neighbor_radius: Radius for considering neighbors (meters)
        velocity_indices: Indices of velocity components in state vector
    
    Returns:
        Dictionary with metrics:
        - 'order_parameter': Global alignment (0-1)
        - 'local_alignment': Average local alignment per agent
        - 'alignment_std': Standard deviation of local alignments
    """
    N = states.shape[0]
    if N < 2:
        return {
            'order_parameter': 0.0,
            'local_alignment': 0.0,
            'alignment_std': 0.0
        }
    
    # Extract positions and velocities
    positions = states[:, :3]  # (N, 3) - x, y, z
    velocities = states[:, velocity_indices[0]:velocity_indices[2]+1]  # (N, 3) - vx, vy, vz
    
    # Compute pairwise distances
    # Note: For N > 10k, this might be slow/memory intensive.
    # We assume metrics are calculated on a subset or checking N before pdist.
    if N > 10000:
        # Fallback for very large N: only compute global order parameter to save memory
        velocity_norms = np.linalg.norm(velocities, axis=1, keepdims=True)
        velocity_norms = np.where(velocity_norms < 1e-6, 1.0, velocity_norms)
        normalized_velocities = velocities / velocity_norms
        velocity_sum = np.sum(normalized_velocities, axis=0)
        global_order = np.linalg.norm(velocity_sum) / N
        return {
            'order_parameter': float(global_order),
            'local_alignment': float(global_order), # Approximation
            'alignment_std': 0.0
        }

    distances = squareform(pdist(positions))
    
    # Normalize velocities
    velocity_norms = np.linalg.norm(velocities, axis=1, keepdims=True)
    velocity_norms = np.where(velocity_norms < 1e-6, 1.0, velocity_norms)  # Avoid division by zero
    normalized_velocities = velocities / velocity_norms
    
    # Global order parameter
    velocity_sum = np.sum(normalized_velocities, axis=0)
    global_order = np.linalg.norm(velocity_sum) / N
    
    # Local alignment (per agent)
    local_alignments = []
    for i in range(N):
        # Find neighbors within radius
        neighbor_mask = (distances[i] < neighbor_radius) & (distances[i] > 0)
        neighbor_indices = np.where(neighbor_mask)[0]
        
        if len(neighbor_indices) == 0:
            local_alignments.append(0.0)
            continue
        
        # Compute cosine similarity with neighbors
        v_i = normalized_velocities[i]
        v_neighbors = normalized_velocities[neighbor_indices]
        
        # Cosine similarity: dot product of normalized vectors
        cosines = np.dot(v_neighbors, v_i)
        local_alignment = np.mean(cosines)
        local_alignments.append(local_alignment)
    
    local_alignments = np.array(local_alignments)
    
    return {
        'order_parameter': float(global_order),
        'local_alignment': float(np.mean(local_alignments)),
        'alignment_std': float(np.std(local_alignments))
    }


def calculate_cluster_entropy(states: np.ndarray,
                              eps: float = 30.0,
                              min_samples: int = 3) -> Dict[str, float]:
    """
    Calculate cluster entropy to measure spatial grouping behavior.
    
    Uses connected components clustering (simplified DBSCAN-like) to identify agent clusters, 
    then computes entropy of cluster size distribution.
    
    Higher entropy indicates more uniform cluster sizes (less polarization).
    Lower entropy indicates few large clusters (strong grouping).
    
    Args:
        states: Agent states array (N, state_dim)
        eps: Cluster radius (connection distance)
        min_samples: Minimum samples per cluster (used to filter noise if strictly following DBSCAN,
                     here we include all components but could filter small ones)
    
    Returns:
        Dictionary with metrics:
        - 'cluster_entropy': Entropy of cluster size distribution
        - 'num_clusters': Number of identified clusters
        - 'largest_cluster_ratio': Ratio of agents in largest cluster
        - 'avg_cluster_size': Average cluster size
    """
    N = states.shape[0]
    if N < min_samples:
        return {
            'cluster_entropy': 0.0,
            'num_clusters': 0,
            'largest_cluster_ratio': 0.0,
            'avg_cluster_size': 0.0
        }
    
    # Extract positions
    positions = states[:, :3]  # (N, 3)
    
    if N > 10000:
        # Avoid full distance matrix for large N
        # For evaluating 100k agents, one would typically use spatial indexing (KDTree).
        # Here we return placeholders if N is too large for pdist
        return {
            'cluster_entropy': 0.0,
            'num_clusters': 0,
            'largest_cluster_ratio': 0.0,
            'avg_cluster_size': 0.0
        }

    # Compute distance matrix
    dist_matrix = squareform(pdist(positions))
    
    # Build adjacency matrix based on eps
    adj_matrix = csr_matrix(dist_matrix < eps)
    
    # Find connected components
    n_components, labels = connected_components(csgraph=adj_matrix, directed=False, return_labels=True)
    
    # Count cluster sizes
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    # Filter small clusters if needed (mimicking min_samples of DBSCAN)
    # For entropy, we usually consider all groupings.
    # If we want to strictly follow DBSCAN 'noise', we would exclude small components.
    # Here we count all.
    
    if len(unique_labels) == 0:
        return {
            'cluster_entropy': 0.0,
            'num_clusters': 0,
            'largest_cluster_ratio': 0.0,
            'avg_cluster_size': 0.0
        }
    
    # Compute cluster size distribution
    cluster_sizes = counts.astype(float)
    cluster_probs = cluster_sizes / cluster_sizes.sum()
    
    # Compute entropy: H = -Σ p_i * log(p_i)
    entropy = -np.sum(cluster_probs * np.log(cluster_probs + 1e-10))
    
    # Normalize by max entropy (uniform distribution)
    # Max entropy is log(N) (all agents separate) or log(num_clusters)?
    # Usually normalized by log(num_clusters) to see balance among clusters,
    # or log(N) to see global fragmentation.
    # We use log(num_clusters) if num_clusters > 1.
    if len(unique_labels) > 1:
        max_entropy = np.log(len(unique_labels))
    else:
        max_entropy = 1.0
        
    normalized_entropy = entropy / (max_entropy + 1e-10)
    
    # Additional statistics
    largest_cluster_size = np.max(cluster_sizes)
    largest_cluster_ratio = largest_cluster_size / N
    avg_cluster_size = np.mean(cluster_sizes)
    
    return {
        'cluster_entropy': float(normalized_entropy),
        'num_clusters': int(n_components),
        'largest_cluster_ratio': float(largest_cluster_ratio),
        'avg_cluster_size': float(avg_cluster_size)
    }


def calculate_emergence_metrics(states: np.ndarray,
                                neighbor_radius: float = 50.0,
                                cluster_eps: float = 30.0) -> Dict[str, float]:
    """
    Calculate all emergence metrics in one call.
    
    This is the main function to call during evaluation.
    
    Args:
        states: Agent states array (N, state_dim)
                Expected format: [x, y, z, vx, vy, vz, ...]
        neighbor_radius: Radius for velocity alignment calculation
        cluster_eps: Epsilon for clustering
    
    Returns:
        Dictionary combining all metrics
    """
    velocity_metrics = calculate_velocity_alignment(states, neighbor_radius)
    cluster_metrics = calculate_cluster_entropy(states, cluster_eps)
    
    # Combine metrics
    all_metrics = {**velocity_metrics, **cluster_metrics}
    
    return all_metrics


def batch_calculate_emergence_metrics(states_batch: List[np.ndarray],
                                      neighbor_radius: float = 50.0,
                                      cluster_eps: float = 30.0) -> Dict[str, np.ndarray]:
    """
    Calculate emergence metrics for a batch of state snapshots.
    
    Useful for processing multiple timesteps or parallel environments.
    
    Args:
        states_batch: List of state arrays, each (N_i, state_dim)
        neighbor_radius: Radius for velocity alignment
        cluster_eps: Epsilon for clustering
    
    Returns:
        Dictionary with arrays of metrics (one value per state snapshot)
    """
    all_results = []
    
    for states in states_batch:
        metrics = calculate_emergence_metrics(states, neighbor_radius, cluster_eps)
        all_results.append(metrics)
    
    # Convert to arrays
    result_dict = {}
    for key in all_results[0].keys():
        result_dict[key] = np.array([r[key] for r in all_results])
    
    return result_dict
