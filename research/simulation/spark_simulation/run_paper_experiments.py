import subprocess
import os
import sys

def run_experiments():
    # Get the directory where this script is located
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define the scripts to run
    scripts = [
        os.path.join("exp1_throughput", "simulate_dataload.py"),
        os.path.join("exp2_kv_cache", "simulate_kv_offload.py"),
        os.path.join("exp3_rwa_security", "simulate_rwa_hashing.py")
    ]
    
    print("🚀 Starting SkyNetUamPlatform Paper Simulation Suite...\n")

    for script_rel_path in scripts:
        script_full_path = os.path.join(base_dir, script_rel_path)
        print(f"▶️  Running {script_rel_path}...")
        try:
            # Run the script using the current python interpreter
            # working directory set to the script's directory to ensure relative paths inside them work if needed,
            # though the scripts use __file__ for output so they are robust.
            subprocess.check_call([sys.executable, script_full_path], cwd=base_dir)
            print("✅ Completed.\n")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error running {script_rel_path}: {e}")
            return

    print("✅ All simulations completed for SkyNetUamPlatform.") 
    print("Please check the following files for results:")
    print("    - exp1_throughput/fig2_throughput.png")
    print("    - exp2_kv_cache/fig3_kv_cache.png")
    print("    - exp3_rwa_security/fig4_rwa_efficiency.png")

if __name__ == "__main__":
    run_experiments()
