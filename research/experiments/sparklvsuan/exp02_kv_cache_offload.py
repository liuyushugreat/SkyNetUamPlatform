import matplotlib.pyplot as plt
import numpy as np
import os

# Constants
GPU_VRAM_LIMIT_GB = 80
KV_CACHE_SIZE_PER_TOKEN_MB = 0.5  # Adjusted to represent effective memory per token steps (e.g. Batch Size * Cache)
GP_SPARK_BANDWIDTH_GB_S = 12.0
GP_SPARK_LATENCY_US = 20
SWAP_LATENCY_MS = GP_SPARK_LATENCY_US / 1000.0

# Simulation Parameters
MAX_TOKENS = 200000
STEP_SIZE = 1000

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def simulate_inference(tokens, use_offload=False):
    """
    Simulates inference latency.
    Returns: (latency_ms, is_oom)
    """
    # Calculate required memory for KV cache
    # Memory (GB) = Tokens * MB_per_token / 1024
    required_memory_gb = (tokens * KV_CACHE_SIZE_PER_TOKEN_MB) / 1024
    
    # Base inference latency (assumed linear growth with context length for simplicity)
    # y = 0.05x + 10 (ms) - just a hypothetical model for attention calculation time
    base_latency = (tokens * 0.0005) + 10 

    if required_memory_gb <= GPU_VRAM_LIMIT_GB:
        # Fits in GPU memory
        return base_latency, False
    else:
        # Exceeds GPU memory
        if not use_offload:
            return float('inf'), True # OOM
        else:
            # Calculate swap penalty
            # Data to swap (GB)
            swap_data_gb = required_memory_gb - GPU_VRAM_LIMIT_GB
            
            # Swap Time = (Data Size / Bandwidth) + Latency Overhead
            # We assume bidirectional swap might be needed or just read, let's assume read penalty for attention
            # Time (s) = GB / GB/s
            transfer_time_s = swap_data_gb / GP_SPARK_BANDWIDTH_GB_S
            transfer_time_ms = transfer_time_s * 1000
            
            # Total penalty includes transmission latency overhead
            total_penalty = transfer_time_ms + SWAP_LATENCY_MS
            
            return base_latency + total_penalty, False

def run_simulation():
    print(f"--- Simulation: LLM KV Cache Offloading (Limit: {GPU_VRAM_LIMIT_GB} GB VRAM) ---")
    
    token_counts = np.arange(0, MAX_TOKENS + STEP_SIZE, STEP_SIZE)
    latencies_baseline = []
    latencies_ours = []
    
    oom_point_baseline = None
    
    for t in token_counts:
        # Baseline
        lat, oom = simulate_inference(t, use_offload=False)
        if oom:
            latencies_baseline.append(None)
            if oom_point_baseline is None:
                oom_point_baseline = t
        else:
            latencies_baseline.append(lat)
            
        # Ours (GP Spark)
        lat_offload, _ = simulate_inference(t, use_offload=True)
        latencies_ours.append(lat_offload)

    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Plot Baseline
    valid_baseline = [l for l in latencies_baseline if l is not None]
    valid_tokens = token_counts[:len(valid_baseline)]
    plt.plot(valid_tokens, valid_baseline, 'r--', label='Baseline (Local VRAM Only)', linewidth=2)
    
    # Plot OOM line
    if oom_point_baseline:
        plt.axvline(x=oom_point_baseline, color='k', linestyle=':', label=f'OOM Threshold (~{int(oom_point_baseline/1000)}k Tokens)')
        plt.text(oom_point_baseline + 2000, max(latencies_ours)/2, 'Out of Memory', rotation=90, color='red')

    # Plot Ours
    plt.plot(token_counts, latencies_ours, 'g-', label='Ours (GP Spark Offload)', linewidth=2)

    plt.xlabel('Context Length (Tokens)')
    plt.ylabel('Inference Latency (ms)')
    plt.title('KV Cache Offloading Performance: VRAM Overflow Handling')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure
    output_path = os.path.join(OUTPUT_DIR, 'fig_kv_cache.png')
    plt.savefig(output_path, dpi=300)
    print(f"Figure saved to: {output_path}")
    
    print(f"Baseline Max Tokens: {oom_point_baseline}")
    print(f"Ours Max Tokens: {MAX_TOKENS}+ (Latency at max: {latencies_ours[-1]:.2f} ms)")

if __name__ == "__main__":
    run_simulation()
