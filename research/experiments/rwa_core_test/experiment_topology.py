import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from ripser import ripser
from persim import plot_diagrams
import sys
import os
from pathlib import Path

# Setup paths
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

from modules.rwa_core.neural_pricing import TorusPricingModel, PizzaPricingModel

# Configuration
CONFIG = {
    'time_mod': 24,
    'route_mod': 60,
    'hidden_dim': 128,
    'lr': 0.001,
    'epochs': 500,
    'batch_size': 256,
    'seed': 42
}

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def generate_synthetic_data(n_samples=5000):
    """
    Generate synthetic 'Fair Pricing' data.
    Price approx (Time + RouteID) % n
    Using continuous approximation for smoothness.
    """
    time_idx = torch.randint(0, CONFIG['time_mod'], (n_samples,))
    route_idx = torch.randint(0, CONFIG['route_mod'], (n_samples,))
    
    # Target function: simple periodic interference
    # Normalize to [0, 1]
    t_norm = time_idx.float() / CONFIG['time_mod']
    r_norm = route_idx.float() / CONFIG['route_mod']
    
    # Construct a target that respects the topology (continuous on torus)
    # Price = sin(2*pi*t) + cos(2*pi*r)
    target = torch.sin(2 * np.pi * t_norm) + torch.cos(2 * np.pi * r_norm)
    target = target.view(-1, 1)
    
    return time_idx, route_idx, target

def train_model(model, time_data, route_data, target_data):
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    criterion = nn.MSELoss()
    
    dataset = torch.utils.data.TensorDataset(time_data, route_data, target_data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    
    loss_history = []
    print(f"Training {model.__class__.__name__}...")
    
    for epoch in range(CONFIG['epochs']):
        epoch_loss = 0
        for t, r, y in loader:
            optimizer.zero_grad()
            pred = model(t, r)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        loss_history.append(epoch_loss / len(loader))
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{CONFIG['epochs']}, Loss: {loss_history[-1]:.4f}")
            
    return loss_history

def extract_representations(model):
    """
    Extract hidden representations for all possible (time, route) pairs.
    """
    model.eval()
    
    # Create grid of all combinations
    times = torch.arange(CONFIG['time_mod'])
    routes = torch.arange(CONFIG['route_mod'])
    grid_t, grid_r = torch.meshgrid(times, routes, indexing='ij')
    
    flat_t = grid_t.flatten()
    flat_r = grid_r.flatten()
    
    # Forward pass to trigger hooks
    with torch.no_grad():
        _ = model(flat_t, flat_r)
        
    # Retrieve captured pre-activations
    # Shape: (time_mod * route_mod, hidden_dim)
    return model.h_pre.numpy()

def analyze_topology(point_cloud, label="Model"):
    """
    Perform PCA and Persistent Homology.
    """
    print(f"\nAnalyzing Topology for {label}...")
    
    # 1. PCA Projection to 3D
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(point_cloud)
    
    # Plot 3D Point Cloud
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # Color by Time to visualize the structure
    colors = np.arange(len(point_cloud)) % CONFIG['route_mod']
    scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], 
                        c=colors, cmap='hsv', s=5, alpha=0.6)
    plt.colorbar(scatter, label='Route ID')
    ax.set_title(f'PCA Projection of Hidden Layer ({label})')
    plt.savefig(f'{label}_pca_3d.png')
    plt.close()
    
    # 2. Persistent Homology
    # Subsample if too large for ripser (max ~1000-2000 points usually fine)
    if len(point_cloud) > 1000:
        indices = np.random.choice(len(point_cloud), 1000, replace=False)
        sub_cloud = point_cloud[indices]
    else:
        sub_cloud = point_cloud
        
    # Compute persistence diagrams (H0, H1, H2)
    # H0: Connected components
    # H1: Loops (Cycles) -> Torus should have 2 significant loops
    # H2: Voids (Cavities) -> Torus should have 1 significant void
    dgms = ripser(sub_cloud, maxdim=2)['dgms']
    
    # Plot Persistence Diagrams
    plt.figure(figsize=(6, 6))
    plot_diagrams(dgms, show=False)
    plt.title(f"Persistence Diagram ({label})")
    plt.savefig(f'{label}_persistence.png')
    plt.close()
    
    return dgms

def calculate_betti_numbers(dgms, threshold=0.5):
    """
    Estimate Betti numbers based on lifetime of features.
    Simple heuristic: features with lifetime > threshold are 'significant'.
    Note: Thresholding in TDA is non-trivial; this is a simplified metric for the experiment.
    """
    betti = []
    for dim, dgm in enumerate(dgms):
        # Lifetime = Death - Birth
        lifetimes = dgm[:, 1] - dgm[:, 0]
        # Filter out infinite death (essential features) or long-lived ones
        # Ripser uses infinity for H0 essential feature
        valid = lifetimes[np.isfinite(lifetimes)]
        
        # For H0, usually 1 connected component is infinite
        if dim == 0:
            # Count infinite + significant finite
            count = 1 + np.sum(valid > threshold)
        else:
            count = np.sum(valid > threshold)
        betti.append(count)
    return betti

def main():
    set_seed(CONFIG['seed'])
    
    # 1. Prepare Data
    t_train, r_train, y_train = generate_synthetic_data()
    
    # 2. Initialize Models
    # Pizza Model (MLP-Add) -> Expecting Cylinder/Möbius or collapsing to Circle
    # Torus Model (MLP-Concat) -> Expecting Torus
    pizza_model = PizzaPricingModel(CONFIG['time_mod'], CONFIG['route_mod'], CONFIG['hidden_dim'])
    torus_model = TorusPricingModel(CONFIG['time_mod'], CONFIG['route_mod'], CONFIG['hidden_dim'])
    
    # 3. Train
    print("--- Training Pizza Model ---")
    train_model(pizza_model, t_train, r_train, y_train)
    
    print("\n--- Training Torus Model ---")
    train_model(torus_model, t_train, r_train, y_train)
    
    # 4. Extract & Analyze
    # Pizza Analysis
    h_pizza = extract_representations(pizza_model)
    dgms_pizza = analyze_topology(h_pizza, "PizzaModel")
    
    # Torus Analysis
    h_torus = extract_representations(torus_model)
    dgms_torus = analyze_topology(h_torus, "TorusModel")
    
    # 5. Report Betti Numbers
    # We look for persistence relative to the scale of the point cloud.
    # A robust way is to look at the persistence diagram visually, 
    # but for automated reporting we use a heuristic.
    print("\n--- Topological Summary ---")
    print(f"Pizza Model Persistence generated at 'PizzaModel_persistence.png'")
    print(f"Torus Model Persistence generated at 'TorusModel_persistence.png'")
    
    # Check for Torus signature (Betti 1, 2, 1) roughly
    # In practice, noise might make this hard to pinpoint exactly without manual tuning.
    print("\nCheck the diagrams:")
    print("- Torus Topology signature: H1 has 2 points far from diagonal, H2 has 1 point.")
    print("- Cylinder/Circle signature: H1 has 1 point.")

if __name__ == "__main__":
    main()

