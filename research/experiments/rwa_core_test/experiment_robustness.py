import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sys
import os
from pathlib import Path

# Setup paths
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

from modules.SkyRwa.neural_pricing import TorusPricingModel
from modules.SkyRwa.adversarial import ArbitrageInjector
from modules.SkyRwa.topology_metrics import calculate_integrity_score, get_betti_numbers

# Configuration
CONFIG = {
    'time_mod': 24,
    'route_mod': 60,
    'hidden_dim': 128,
    'lr': 0.002,
    'epochs': 600,
    'batch_size': 256,
    'seed': 1337,
    'output_dir': project_root / 'research/papers/SkyRwa_papers/figures'
}

CONFIG['output_dir'].mkdir(parents=True, exist_ok=True)

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def generate_data(n_samples=6000, poison=False):
    """
    Generate synthetic data.
    If poison=True, injects arbitrage loops.
    """
    # Base pattern (Clean)
    time_idx = torch.randint(0, CONFIG['time_mod'], (n_samples,))
    route_idx = torch.randint(0, CONFIG['route_mod'], (n_samples,))
    
    t_norm = time_idx.float() / CONFIG['time_mod']
    r_norm = route_idx.float() / CONFIG['route_mod']
    
    # Target: Torus function
    # Price = 5 + 2*sin(t) + 2*cos(r)
    target = 5.0 + 2.0 * torch.sin(2 * np.pi * t_norm) + 2.0 * torch.cos(2 * np.pi * r_norm)
    target = target.view(-1, 1)
    
    if poison:
        print("Injecting Arbitrage Attacks...")
        injector = ArbitrageInjector(seed=CONFIG['seed'])
        # Inject loop
        time_idx, route_idx, target = injector.inject_cyclic_loop(
            time_idx, route_idx, target, num_victims=int(n_samples * 0.15)
        )
        # Inject fragmentation
        time_idx, route_idx, target = injector.inject_fragmentation(
            time_idx, route_idx, target, gap_size=8.0
        )
        
    return time_idx, route_idx, target

def train_model(name, time_data, route_data, target_data):
    model = TorusPricingModel(CONFIG['time_mod'], CONFIG['route_mod'], CONFIG['hidden_dim'])
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    criterion = nn.MSELoss()
    
    dataset = torch.utils.data.TensorDataset(time_data, route_data, target_data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    
    print(f"Training {name}...")
    for epoch in range(CONFIG['epochs']):
        epoch_loss = 0
        for t, r, y in loader:
            optimizer.zero_grad()
            pred = model(t, r)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch + 1) % 100 == 0:
            print(f"[{name}] Epoch {epoch+1}, Loss: {epoch_loss / len(loader):.4f}")
            
    return model

def extract_representations(model):
    model.eval()
    times = torch.arange(CONFIG['time_mod'])
    routes = torch.arange(CONFIG['route_mod'])
    grid_t, grid_r = torch.meshgrid(times, routes, indexing='ij')
    flat_t = grid_t.flatten()
    flat_r = grid_r.flatten()
    
    with torch.no_grad():
        _ = model(flat_t, flat_r)
    return model.h_pre.numpy(), flat_r.numpy()

def plot_comparison(clean_rep, clean_colors, adv_rep, adv_colors, clean_score, adv_score):
    print("Generating Comparison Plot...")
    
    pca = PCA(n_components=3)
    
    # Fit PCA on clean, transform both to share axes (roughly)
    # Actually, better to fit separately to see the intrinsic shape of each
    clean_pca = PCA(n_components=3).fit_transform(clean_rep)
    adv_pca = PCA(n_components=3).fit_transform(adv_rep)
    
    fig = plt.figure(figsize=(16, 7))
    
    # Plot Clean
    ax1 = fig.add_subplot(121, projection='3d')
    sc1 = ax1.scatter(clean_pca[:, 0], clean_pca[:, 1], clean_pca[:, 2], 
                     c=clean_colors, cmap='hsv', s=5, alpha=0.6)
    ax1.set_title(f"Clean Market (Baseline)\nIntegrity Risk: {clean_score:.4f} (Low)")
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.set_zlabel("PC3")
    
    # Plot Adversarial
    ax2 = fig.add_subplot(122, projection='3d')
    sc2 = ax2.scatter(adv_pca[:, 0], adv_pca[:, 1], adv_pca[:, 2], 
                     c=adv_colors, cmap='hsv', s=5, alpha=0.6)
    ax2.set_title(f"Under Arbitrage Attack\nIntegrity Risk: {adv_score:.4f} (High)")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.set_zlabel("PC3")
    
    # Add shared colorbar
    # fig.colorbar(sc1, ax=[ax1, ax2], label='Route ID', shrink=0.5)
    
    plt.tight_layout()
    output_path = CONFIG['output_dir'] / 'robustness_comparison.png'
    plt.savefig(output_path, dpi=300)
    print(f"Comparison plot saved to {output_path}")
    
    # Also save PDF for paper
    plt.savefig(CONFIG['output_dir'] / 'robustness_comparison.pdf')

def main():
    set_seed(CONFIG['seed'])
    
    # 1. Clean Experiment
    print("\n--- Phase 1: Clean Market ---")
    t_clean, r_clean, y_clean = generate_data(poison=False)
    model_clean = train_model("CleanModel", t_clean, r_clean, y_clean)
    rep_clean, col_clean = extract_representations(model_clean)
    
    # 2. Poisoned Experiment
    print("\n--- Phase 2: Arbitrage Attack ---")
    t_adv, r_adv, y_adv = generate_data(poison=True)
    model_adv = train_model("AdversarialModel", t_adv, r_adv, y_adv)
    rep_adv, col_adv = extract_representations(model_adv)
    
    # 3. Validation
    print("\n--- Computing Topology Metrics ---")
    score_clean = calculate_integrity_score(rep_clean)
    betti_clean = get_betti_numbers(rep_clean)
    print(f"Clean Model: Betti={betti_clean}, Risk Score={score_clean:.4f}")
    
    score_adv = calculate_integrity_score(rep_adv)
    betti_adv = get_betti_numbers(rep_adv)
    print(f"Adversarial Model: Betti={betti_adv}, Risk Score={score_adv:.4f}")
    
    # 4. Visualize
    plot_comparison(rep_clean, col_clean, rep_adv, col_adv, score_clean, score_adv)
    
    # 5. Save Metrics to text file for LaTeX
    with open(CONFIG['output_dir'] / 'metrics.tex', 'w') as f:
        f.write(f"\\def\\cleanRisk{{{score_clean:.2f}}}\n")
        f.write(f"\\def\\advRisk{{{score_adv:.2f}}}\n")
        f.write(f"\\def\\cleanBetti{{({betti_clean[0]}, {betti_clean[1]}, {betti_clean[2]})}}\n")
        f.write(f"\\def\\advBetti{{({betti_adv[0]}, {betti_adv[1]}, {betti_adv[2]})}}\n")

if __name__ == "__main__":
    main()

