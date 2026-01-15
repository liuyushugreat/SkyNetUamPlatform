import matplotlib.pyplot as plt
import numpy as np
import os

def simulate_kv_offloading():
    # Parameters
    max_tokens = 200000
    oom_threshold_tokens = 164000
    kv_size_per_token_mb = 0.5
    bandwidth_gb_s = 12.0
    
    # Generate token steps
    tokens = np.arange(0, max_tokens + 1000, 1000)
    
    # Base latency (ms) - hypothetical linear growth
    base_latency_ms = 0.0005 * tokens
    
    # Baseline Scenario (OOM)
    latency_baseline = []
    for t in tokens:
        if t > oom_threshold_tokens:
            latency_baseline.append(np.nan) # Crash
        else:
            latency_baseline.append(0.0005 * t)
            
    # GP Spark Scenario (Offloading)
    latency_ours = []
    swap_overhead_ms = []
    
    for t in tokens:
        compute_ms = 0.0005 * t
        
        if t <= oom_threshold_tokens:
            latency_ours.append(compute_ms)
            swap_overhead_ms.append(0)
        else:
            # Calculate Data to Swap
            # Total Cache needed - VRAM capacity
            excess_tokens = t - oom_threshold_tokens
            data_to_swap_mb = excess_tokens * kv_size_per_token_mb
            
            # Swap Time = Data / Bandwidth
            # Convert MB to GB for bandwidth calc
            data_to_swap_gb = data_to_swap_mb / 1024.0
            swap_time_s = data_to_swap_gb / bandwidth_gb_s
            swap_time_ms = swap_time_s * 1000.0
            
            total_latency = compute_ms + swap_time_ms
            latency_ours.append(total_latency)
            swap_overhead_ms.append(swap_time_ms)

    # Visualization
    plt.figure(figsize=(10, 6))
    
    plt.plot(tokens, latency_baseline, 'r--', label='Baseline (OOM > 164k)', linewidth=2)
    plt.plot(tokens, latency_ours, 'g-', label='Ours (GP Spark Offload)', linewidth=2)
    
    # Mark OOM point
    plt.axvline(x=oom_threshold_tokens, color='k', linestyle=':', alpha=0.5)
    plt.text(oom_threshold_tokens + 2000, max(latency_ours)*0.3, 'VRAM Limit (80GB)\n164k Tokens', color='k')
    
    # Labels
    plt.xlabel('Context Length (Tokens)')
    plt.ylabel('Inference Latency (ms)')
    plt.title('KV Cache Offloading: Inference Latency vs Context Length')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Log specific data points for verification
    print(f"--- KV Cache Offloading Simulation ---")
    print(f"VRAM Limit: {oom_threshold_tokens} tokens")
    print(f"Offload Bandwidth: {bandwidth_gb_s} GB/s")
    print("-" * 50)
    
    check_points = [164000, 180000, 200000]
    print(f"{'Tokens':<10} | {'Baseline (ms)':<15} | {'Ours (ms)':<15} | {'Swap Time (ms)':<15}")
    
    for pt in check_points:
        idx = np.where(tokens == pt)[0][0]
        base = latency_baseline[idx]
        ours = latency_ours[idx]
        swap = swap_overhead_ms[idx]
        print(f"{pt:<10} | {str(base):<15} | {ours:<15.2f} | {swap:<15.2f}")

    # Save plot
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, 'fig3_kv_cache.png')
    plt.savefig(output_path, dpi=300)
    print(f"\nPlot saved to: {output_path}")

if __name__ == "__main__":
    simulate_kv_offloading()
