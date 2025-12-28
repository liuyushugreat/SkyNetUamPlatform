import time
import numpy as np
from nexus_core.mas.uam_env import SkyNetMultiAgentEnv
from nexus_core.assetization.integration import AssetizationPipeline

def run_assetization_demo():
    print("Initializing SkyNet Assetization Demo...")
    
    # 1. Start the Assetization Pipeline (The Listener)
    pipeline = AssetizationPipeline()
    
    # 2. Initialize Environment (The Publisher)
    env = SkyNetMultiAgentEnv(num_agents=3, render_mode='console')
    
    obs = env.reset()
    print("Environment Reset. Starting Simulation Loop...")
    
    # 3. Run Simulation
    for _ in range(20): # Run for 20 steps
        # Random Action
        actions = [env.action_space.sample() for _ in range(env.num_agents)]
        
        # Step (Triggers TELEMETRY_UPDATE events internally)
        obs, rewards, dones, info = env.step(actions)
        
        # Simulate time passing
        # time.sleep(0.1) 
        
        if all(dones):
            break
            
    print("\nSimulation Loop Finished.")
    
    # Force trigger mission complete for demo purposes
    # Since we only ran 20 steps, agents might not be done yet.
    # We manually trigger the event to show the assetization output.
    from nexus_core.assetization.event_bus import AssetEventBus, TelemetryEvent
    import uuid
    
    print("\n[Demo] Manually triggering MISSION_COMPLETE for uav_0 to demonstrate assetization...")
    bus = AssetEventBus.get_instance()
    bus.publish(TelemetryEvent(
        event_id=uuid.uuid4().hex,
        event_type="MISSION_COMPLETE",
        source_id="uav_0",
        payload={"status": "COMPLETED", "final_step": 20}
    ))
    
if __name__ == "__main__":
    run_assetization_demo()

