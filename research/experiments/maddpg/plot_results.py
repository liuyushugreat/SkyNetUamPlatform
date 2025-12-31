import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_scalability(csv_path='scalability_results.csv'):
    """
    Plot Inference Time vs. Swarm Size.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: {csv_path} not found. Run evaluate.py first.")
        return

    # Set style
    sns.set(style="whitegrid", context="paper", font_scale=1.2)
    
    plt.figure(figsize=(10, 6))
    
    # Plot Inference Time
    sns.lineplot(x='num_agents', y='inference_time_ms', data=df, marker='o', linewidth=2.5, color='#2ecc71')
    
    # Add titles and labels
    plt.title('Scalability Analysis: Inference Latency vs. Swarm Size', fontsize=16, pad=20)
    plt.xlabel('Number of Agents (N)', fontsize=14)
    plt.ylabel('Inference Time per Step (ms)', fontsize=14)
    
    # Format axes
    plt.xticks([10000, 50000, 100000], ['10k', '50k', '100k'])
    
    # Add threshold line (e.g., 50ms for 20Hz real-time)
    plt.axhline(y=50, color='r', linestyle='--', alpha=0.5)
    plt.text(10000, 52, 'Real-time Threshold (20Hz)', color='r', fontsize=10)
    
    # Save plot
    plt.tight_layout()
    plt.savefig('fig_inference_scalability.png', dpi=300)
    print("Plot saved to fig_inference_scalability.png")
    plt.close()

if __name__ == "__main__":
    plot_scalability()

