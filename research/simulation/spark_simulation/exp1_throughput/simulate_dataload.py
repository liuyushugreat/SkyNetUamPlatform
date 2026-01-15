import matplotlib.pyplot as plt
import os

def simulate_data_loading():
    # Parameters
    dataset_size_gb = 1024.0  # 1 TB
    
    # Bandwidth in GB/s
    scenarios = {
        "SATA SSD": 0.5,       # 500 MB/s
        "USB 3.0 HDD": 0.3,    # 300 MB/s
        "GP Spark\n(NVMe-oF)": 12.0 # 12 GB/s
    }
    
    results = {}
    print(f"--- Data Loading Simulation (Dataset: {dataset_size_gb} GB) ---")
    print(f"{'Storage Solution':<20} | {'Bandwidth (GB/s)':<18} | {'Time (s)':<10}")
    print("-" * 55)
    
    for name, bw in scenarios.items():
        time_s = dataset_size_gb / bw
        results[name] = time_s
        clean_name = name.replace('\n', ' ')
        print(f"{clean_name:<20} | {bw:<18.1f} | {time_s:.2f}")

    # Visualization
    names = list(results.keys())
    times = list(results.values())
    
    plt.figure(figsize=(10, 6))
    # Colors: Red for slowest, Orange for medium, Green for fastest
    bars = plt.bar(names, times, color=['#e74c3c', '#f39c12', '#2ecc71'])
    
    plt.ylabel('Loading Time (seconds)')
    plt.title('Data Loading Efficiency Comparison (Dataset: 1 TB)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add text labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + (max(times)*0.01),
                f'{height:.1f}s',
                ha='center', va='bottom', fontweight='bold')
                
    # Save plot
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, 'fig2_throughput.png')
    plt.savefig(output_path, dpi=300)
    print(f"\nPlot saved to: {output_path}")

if __name__ == "__main__":
    simulate_data_loading()
