import numpy as np
from ripser import ripser
from typing import Dict, Any, List

def calculate_integrity_score(point_cloud: np.ndarray, threshold: float = 0.5) -> float:
    """
    Quantifies the topological integrity of the market manifold.
    
    Args:
        point_cloud: Numpy array of shape (N, D) representing the learned latent space.
        threshold: Persistence threshold to consider a topological feature 'significant'.
        
    Returns:
        risk_score: A float between 0.0 (Perfect Torus) and 1.0 (Broken Topology).
    """
    
    # 1. Compute Persistent Homology
    # maxdim=2 because we care about H0, H1, H2 for a Torus
    # Check if point cloud is too large, subsample if needed for performance
    if point_cloud.shape[0] > 1000:
        indices = np.random.choice(point_cloud.shape[0], 1000, replace=False)
        sub_cloud = point_cloud[indices]
    else:
        sub_cloud = point_cloud

    try:
        diagrams = ripser(sub_cloud, maxdim=2)['dgms']
    except Exception as e:
        print(f"Topological computation failed: {e}")
        return 1.0 # High risk if topology cannot be computed

    # 2. Calculate Betti Numbers based on lifetime threshold
    betti_numbers = []
    for dim, dgm in enumerate(diagrams):
        if len(dgm) == 0:
            betti_numbers.append(0)
            continue
            
        # Lifetime = death - birth
        lifetimes = dgm[:, 1] - dgm[:, 0]
        
        # Filter infinite death (essential features)
        # In ripser, essential features have death = infinity
        finite_mask = np.isfinite(dgm[:, 1])
        essential_count = np.sum(~finite_mask)
        
        # Significant finite features
        significant_finite = np.sum(lifetimes[finite_mask] > threshold)
        
        betti_numbers.append(essential_count + significant_finite)
    
    # Pad betti_numbers if H2 was empty
    while len(betti_numbers) < 3:
        betti_numbers.append(0)
        
    beta_0, beta_1, beta_2 = betti_numbers[0], betti_numbers[1], betti_numbers[2]
    
    # 3. Compute "Torus Distance"
    # Target: Beta0=1, Beta1=2, Beta2=1 (for a 2-Torus)
    # Note: Depending on the embedding, Beta2 might be 0 or 1. 
    # For a flat torus in 4D (Cliffords Torus), Beta2=1.
    # We penalize deviation from (1, 2, 1).
    
    diff_0 = abs(beta_0 - 1)
    diff_1 = abs(beta_1 - 2)
    diff_2 = abs(beta_2 - 1)
    
    total_deviation = diff_0 + diff_1 + diff_2
    
    # Normalize risk score using a sigmoid-like function
    # 0 deviation -> Risk 0
    # Any deviation increases risk asymptotically to 1
    risk_score = 1.0 - np.exp(-total_deviation)
    
    return float(risk_score)

def get_betti_numbers(point_cloud: np.ndarray, threshold: float = 0.5) -> List[int]:
    """Helper to return raw Betti numbers for debugging."""
    # ... (Simplified logic similar to above)
    # Re-running logic for clarity in return type
    if point_cloud.shape[0] > 800:
        indices = np.random.choice(point_cloud.shape[0], 800, replace=False)
        sub_cloud = point_cloud[indices]
    else:
        sub_cloud = point_cloud
        
    diagrams = ripser(sub_cloud, maxdim=2)['dgms']
    bettis = []
    for dgm in diagrams:
        lifetimes = dgm[:, 1] - dgm[:, 0]
        finite_mask = np.isfinite(dgm[:, 1])
        count = np.sum(~finite_mask) + np.sum(lifetimes[finite_mask] > threshold)
        bettis.append(int(count))
    while len(bettis) < 3: bettis.append(0)
    return bettis

