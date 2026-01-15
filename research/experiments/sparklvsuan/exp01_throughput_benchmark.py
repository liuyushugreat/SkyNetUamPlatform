import matplotlib.pyplot as plt
import numpy as np
import os

# Constants: Bandwidth definitions (MB/s)
# Ref: [21, 42] in the context of the user request
BANDWIDTHS = {
    "SATA SSD": 500,        # Traditional Local Storage
    "USB 3.0 HDD": 300,     # External Storage
    "GP Spark (NVMe-oF)": 12000 # GP Spark with NVMe-oF High Bandwidth
}

# Scenario: Loading massive UAM training data (e.g., 1 TB of images/Lidar)
DATA_SIZE_GB = 1024 
DATA_SIZE_MB = DATA_SIZE_GB * 1024

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def simulate_loading(data_size_mb, bandwidth_mb_s):
    """
    Simulates data loading time.
    Time (s) = Size (MB) / Bandwidth (MB/s)
    """
    if bandwidth_mb_s <= 0:
        return float('inf')
    return data_size_mb / bandwidth_mb_s

def run_benchmark():
    results = {}
    print(f"--- Simulation Benchmark: Loading {DATA_SIZE_GB} GB of UAM Data ---")
    
    # Run simulation
    for name, bw in BANDWIDTHS.items():
        time_s = simulate_loading(DATA_SIZE_MB, bw)
        results[name] = time_s
        print(f"{name:<20} (BW: {bw:>5} MB/s): {time_s:>8.2f} s ({time_s/60:>6.2f} min)")

    # Calculate improvement
    baseline_time = results["SATA SSD"]
    gp_time = results["GP Spark (NVMe-oF)"]
    improvement_percent = ((baseline_time - gp_time) / baseline_time) * 100
    
    print("-" * 60)
    print(f"[Result] 基于 GP Spark 的加载时间缩短了 {improvement_percent:.2f}%")
    print("-" * 60)

    # Visualization
    names = list(results.keys())
    times = list(results.values())
    
    plt.figure(figsize=(10, 6))
    # Use different colors to highlight GP Spark
    colors = ['#bdc3c7', '#bdc3c7', '#e74c3c'] 
    bars = plt.bar(names, times, color=colors, width=0.6)
    
    # Add text labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + (max(times)*0.01),
                 f'{height:.1f} s',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.ylabel('Loading Time (seconds)', fontsize=12)
    plt.title(f'Data Loading Efficiency Comparison\n(Dataset Size: {DATA_SIZE_GB} TB)', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Save figure
    output_path = os.path.join(OUTPUT_DIR, 'fig_throughput.png')
    plt.savefig(output_path, dpi=300)
    print(f"Figure saved to: {output_path}")

if __name__ == "__main__":
    run_benchmark()
