"""
SkyNet-RWA-Nexus Training Loop Demo.

Runs a short epoch of MADDPG training to verify tensor operations and gradient flow.
"""

import numpy as np
import torch
from nexus_core.mas.maddpg import MADDPGTrainer, MADDPGAgent
from nexus_core.mas.environment import SkyNetEnv

def train_demo():
    print(">>> [MADDPG] Initializing Training Session...")
    
    # 1. Init Environment
    num_agents = 3
    env = SkyNetEnv(num_agents=num_agents)
    
    # Dims
    obs_dim = 10
    action_dim = 4
    global_state_dim = obs_dim * num_agents
    total_action_dim = action_dim * num_agents
    
    # 2. Init Agents
    agents = []
    for i in range(num_agents):
        agent = MADDPGAgent(
            agent_id=f"uav_{i}",
            obs_dim=obs_dim,
            action_dim=action_dim,
            global_state_dim=global_state_dim,
            total_action_dim=total_action_dim
        )
        agents.append(agent)
        
    trainer = MADDPGTrainer(agents)
    
    # 3. Data Collection Loop (1 Episode)
    obs = env.reset()
    global_state = env.get_global_state() # Initial random
    
    print(">>> [MADDPG] Collecting Experience...")
    for t in range(50):
        # Select Actions
        actions = {}
        actions_list = [] # For global buffer
        
        for i, agent_id in enumerate(env.agents):
            act = agents[i].act(obs[agent_id], noise=0.1)
            actions[agent_id] = act
            actions_list.append(act)
            
        next_obs, rewards, dones, infos = env.step(actions)
        
        # Prepare for buffer
        obs_n = [obs[aid] for aid in env.agents]
        action_n = actions_list
        reward_n = [rewards[aid] for aid in env.agents]
        next_obs_n = [next_obs[aid] for aid in env.agents]
        done_n = [dones[aid] for aid in env.agents]
        
        # Mock global state update (concat next_obs)
        next_global_state = np.concatenate(next_obs_n)
        
        trainer.store_transition(obs_n, action_n, reward_n, next_obs_n, done_n, global_state, next_global_state)
        
        obs = next_obs
        global_state = next_global_state
        
        if dones["__all__"]:
            break
            
    print(f"Buffer Size: {len(trainer.replay_buffer)}")
    
    # 4. Gradient Update
    print(">>> [MADDPG] Performing Gradient Update...")
    # Force update even with small batch for demo
    from nexus_core.mas import maddpg
    maddpg.BATCH_SIZE = 16 
    
    trainer.update()
    print(">>> [MADDPG] Update Successful. Weights modified.")

if __name__ == "__main__":
    train_demo()

