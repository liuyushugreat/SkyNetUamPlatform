import matplotlib.pyplot as plt
import os

def simulate_rwa_security():
    # Parameters
    configs = ['Software (CPU)', 'GP Spark (ASIC)']
    throughput_mbs = [142.30, 2854.29]
    cpu_usage_pct = [89.7, 9.6]

    print(f"--- RWA Security Simulation ---")
    print(f"{'Configuration':<15} | {'Throughput (MB/s)':<20} | {'CPU Usage (%)':<15}")
    print("-" * 55)
    for i in range(len(configs)):
        print(f"{configs[i]:<15} | {throughput_mbs[i]:<20.2f} | {cpu_usage_pct[i]:<15.1f}")

    # Visualization - Dual Axis Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Bar Chart for Throughput (Left Axis)
    color = 'tab:blue'
    ax1.set_xlabel('Configuration')
    ax1.set_ylabel('Throughput (MB/s)', color=color)
    bars = ax1.bar(configs, throughput_mbs, color=color, alpha=0.6, width=0.4, label='Throughput')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 3200)

    # Add text labels for bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 50,
                f'{height:.1f} MB/s',
                ha='center', va='bottom', color='blue', fontweight='bold')

    # Line/Scatter for CPU Usage (Right Axis)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('CPU Usage (%)', color=color)
    ax2.plot(configs, cpu_usage_pct, color=color, marker='o', linewidth=2, markersize=10, linestyle='--', label='CPU Usage')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 100)

    # Add text labels for points
    for i, txt in enumerate(cpu_usage_pct):
        ax2.text(i, txt + 3, f'{txt}%', ha='center', color='red', fontweight='bold')

    # Annotations for improvements
    speedup = throughput_mbs[1] / throughput_mbs[0]
    cpu_reduction = cpu_usage_pct[0] - cpu_usage_pct[1]
    
    # Speedup arrow
    plt.annotate(f"{speedup:.1f}x Speedup", 
                 xy=(1, throughput_mbs[1]/35), xycoords=('data', 'axes fraction'),
                 xytext=(0.5, 0.5), textcoords='axes fraction',
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 horizontalalignment='right', verticalalignment='top')

    plt.title('RWA Hardware Acceleration Efficiency Comparison')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    # Save plot
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, 'fig4_rwa_efficiency.png')
    plt.savefig(output_path, dpi=300)
    print(f"\nPlot saved to: {output_path}")

if __name__ == "__main__":
    simulate_rwa_security()
