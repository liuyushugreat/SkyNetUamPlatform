import numpy as np
import sys
import os

# Add the project root to path so we can import nexus_core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from nexus_core.mas.uam_env import SkyNetMultiAgentEnv

def run_check():
    print("=== Starting Environment Verification ===")
    
    # 1. Instantiate the environment
    num_agents = 5
    neighbor_k = 3
    try:
        env = SkyNetMultiAgentEnv(num_agents=num_agents, neighbor_k=neighbor_k, render_mode='console')
        print("[OK] Environment instantiated successfully.")
    except Exception as e:
        print(f"[FAIL] Failed to instantiate environment: {e}")
        return

    # 2. Check Spaces
    print(f"\n--- Space Dimensions ---")
    print(f"Num Agents: {env.num_agents}")
    print(f"Action Space (Per Agent): {env.action_space}")
    print(f"Observation Space (Per Agent): {env.observation_space}")
    
    # Expected Obs Dim: 3 (Pos) + 3 (Vel) + 3 (Goal) + 3*K (Neighbors)
    expected_obs_dim = 3 + 3 + 3 + (3 * neighbor_k)
    assert env.observation_space.shape[0] == expected_obs_dim, \
        f"[FAIL] Obs Dim Mismatch! Expected {expected_obs_dim}, got {env.observation_space.shape[0]}"
    print(f"[OK] Observation dimension matches expected: {expected_obs_dim}")

    # 3. Test Reset
    print(f"\n--- Testing Reset ---")
    obs = env.reset()
    
    if not isinstance(obs, list):
        print(f"[FAIL] Reset should return a list, got {type(obs)}")
        return
        
    if len(obs) != num_agents:
        print(f"[FAIL] Reset returned {len(obs)} observations, expected {num_agents}")
        return
        
    if not isinstance(obs[0], np.ndarray):
        print(f"[FAIL] Observation should be numpy array, got {type(obs[0])}")
        return

    print("[OK] Reset works. Initial observation shape:", obs[0].shape)

    # 4. Simulation Loop
    print(f"\n--- Running 100-Step Simulation ---")
    try:
        for step in range(100):
            # Generate random actions for ALL agents
            # MADDPG input format: List of action arrays
            actions = [env.action_space.sample() for _ in range(num_agents)]
            
            # Step
            next_obs, rewards, dones, infos = env.step(actions)
            
            # Validation Checks
            # A. Check Next Obs
            assert len(next_obs) == num_agents, "Next Obs length mismatch"
            assert next_obs[0].shape == env.observation_space.shape, f"Obs shape changed! {next_obs[0].shape}"
            assert next_obs[0].dtype == np.float32, f"Obs dtype wrong: {next_obs[0].dtype}"
            
            # B. Check Rewards
            assert isinstance(rewards, list), "Rewards must be a list"
            assert len(rewards) == num_agents, "Rewards length mismatch"
            assert isinstance(rewards[0], float), f"Reward type wrong: {type(rewards[0])}"
            
            # C. Check Dones
            assert isinstance(dones, list), "Dones must be a list"
            assert len(dones) == num_agents, "Dones length mismatch"
            assert isinstance(dones[0], (bool, np.bool_)), f"Done type wrong: {type(dones[0])}"
            
            if step % 20 == 0:
                print(f"Step {step:03d} | Mean Reward: {np.mean(rewards):.4f} | Dones: {sum(dones)}")
                
            # If all agents are done, reset (just to keep loop running)
            if all(dones):
                print("All agents done, resetting...")
                env.reset()
                
    except Exception as e:
        print(f"[FAIL] Simulation Error at step {step}: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n[OK] Simulation Loop Completed Successfully.")
    print("=== Verification Passed ===")

if __name__ == "__main__":
    run_check()

