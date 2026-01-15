import time
import random
import matplotlib.pyplot as plt
import numpy as np
import os

# Configuration
DATA_CHUNKS = 100
CHUNK_SIZE_MB = 64  # 64MB per chunk
TOTAL_SIZE_MB = DATA_CHUNKS * CHUNK_SIZE_MB

# Performance Parameters (simulated)
# Software: CPU overhead per MB (s)
CPU_HASH_TIME_PER_MB = 0.005  # Software SHA-256 approx
DISK_WRITE_BW_SW_MB_S = 500   # SATA SSD

# Hardware (GP Spark): Offload efficiency
HARDWARE_OFFLOAD_SPEEDUP = 20  # ASIC is 20x faster for hashing
DISK_WRITE_BW_HW_MB_S = 12000  # NVMe-oF

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def simulate_pipeline(mode="software"):
    """
    Simulates the pipeline: Data -> Hash/Encrypt -> Write -> Blockchain Log
    """
    total_time = 0
    cpu_usage_log = []
    
    print(f"[{mode.upper()}] Processing {DATA_CHUNKS} chunks ({TOTAL_SIZE_MB} MB)...")
    
    for i in range(DATA_CHUNKS):
        # 1. Hashing / Encryption
        if mode == "software":
            # High CPU usage
            process_time = CHUNK_SIZE_MB * CPU_HASH_TIME_PER_MB
            cpu_load = random.uniform(80, 100)
        else:
            # Hardware offload (GP Spark)
            # Low CPU usage (just dispatch), fast processing
            process_time = (CHUNK_SIZE_MB * CPU_HASH_TIME_PER_MB) / HARDWARE_OFFLOAD_SPEEDUP
            cpu_load = random.uniform(5, 15)
            
        # 2. Storage Write
        if mode == "software":
            write_time = CHUNK_SIZE_MB / DISK_WRITE_BW_SW_MB_S
        else:
            write_time = CHUNK_SIZE_MB / DISK_WRITE_BW_HW_MB_S
            
        # 3. Blockchain Log (Simulated async call, low overhead for both)
        chain_overhead = 0.001 
        
        # Total chunk time
        chunk_latency = process_time + write_time + chain_overhead
        
        # Simulate processing variability
        chunk_latency *= random.uniform(0.95, 1.05)
        
        total_time += chunk_latency
        cpu_usage_log.append(cpu_load)
        
        # Simulating work
        # time.sleep(chunk_latency * 0.01) # fast-forward simulation
        
    avg_cpu = sum(cpu_usage_log) / len(cpu_usage_log)
    throughput = TOTAL_SIZE_MB / total_time
    
    return total_time, throughput, avg_cpu

def run_experiment():
    print("--- Simulation: RWA Secure Data Storage & Blockchain Logging ---")
    
    # Run Baseline
    t_sw, thr_sw, cpu_sw = simulate_pipeline("software")
    
    # Run GP Spark
    t_hw, thr_hw, cpu_hw = simulate_pipeline("hardware")
    
    # Metrics
    throughput_loss_sw = (1 - (thr_sw / DISK_WRITE_BW_SW_MB_S)) * 100 # Theoretical vs Realized
    # Note: For HW, we compare against its own Theoretical Max
    throughput_loss_hw = (1 - (thr_hw / DISK_WRITE_BW_HW_MB_S)) * 100 
    
    speedup = thr_hw / thr_sw
    
    print("\n" + "="*50)
    print(f"{'Metric':<25} | {'Software (Baseline)':<20} | {'GP Spark (Hardware)':<20}")
    print("-" * 73)
    print(f"{'Total Time (s)':<25} | {t_sw:<20.4f} | {t_hw:<20.4f}")
    print(f"{'Throughput (MB/s)':<25} | {thr_sw:<20.2f} | {thr_hw:<20.2f}")
    print(f"{'CPU Usage (Avg %)':<25} | {cpu_sw:<20.1f} | {cpu_hw:<20.1f}")
    print("-" * 73)
    print(f"Speedup Factor: {speedup:.2f}x")
    print(f"CPU Load Reduction: {cpu_sw - cpu_hw:.1f} percent points")
    print("="*50 + "\n")

    # Visualization
    labels = ['Software (CPU+SATA)', 'GP Spark (Offload+NVMe)']
    throughput_vals = [thr_sw, thr_hw]
    cpu_vals = [cpu_sw, cpu_hw]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Configuration')
    ax1.set_ylabel('Throughput (MB/s)', color=color)
    bars1 = ax1.bar(labels, throughput_vals, color=color, alpha=0.6, width=0.4, label='Throughput')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Add throughput labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.0f} MB/s',
                 ha='center', va='bottom', color='blue', fontweight='bold')

    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('CPU Usage (%)', color=color)  
    ax2.plot(labels, cpu_vals, color=color, marker='o', linewidth=2, linestyle='dashed', label='CPU Load')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 110)
    
    # Add CPU labels
    for i, v in enumerate(cpu_vals):
        ax2.text(i, v + 5, f'{v:.1f}%', ha='center', color='red', fontweight='bold')

    plt.title('RWA Security: Hardware Offloading Efficiency')
    fig.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, 'fig_rwa_security.png')
    plt.savefig(output_path, dpi=300)
    print(f"Figure saved to: {output_path}")

if __name__ == "__main__":
    run_experiment()
