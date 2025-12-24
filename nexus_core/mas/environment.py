"""
SkyNet-RWA-Nexus RL Environment Wrapper.

This module adapts the SkyNet Simulation into a standard PettingZoo/Gym environment
for training MADDPG agents.

Reward Function:
    $$ R_i(t) = \alpha \cdot \Delta d_i - \beta \cdot I_{crash} - \gamma \cdot C_{toll} $$
    Where:
    - $\Delta d_i$: Progress towards destination.
    - $I_{crash}$: Collision indicator function.
    - $C_{toll}$: Paid congestion fees.
"""

import gym
import numpy as np
from gym import spaces
from typing import Dict, List

# Simulating a ParallelEnv from PettingZoo
class SkyNetEnv:
    """
    Multi-Agent Environment for UAV Traffic Management.
    """
    def __init__(self, num_agents=5):
        self.num_agents = num_agents
        self.agents = [f"uav_{i}" for i in range(num_agents)]
        
        # Observation: [x, y, z, vx, vy, vz, bat, price, dest_x, dest_y]
        self.observation_spaces = {
            agent: spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
            for agent in self.agents
        }
        
        # Action: [vx_ratio, vy_ratio, vz_ratio, bid_ratio]
        self.action_spaces = {
            agent: spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
            for agent in self.agents
        }
        
        self.state_dim = 10 * num_agents
        self.action_dim = 4 * num_agents
        
        self.reset()

    def reset(self):
        """Reset simulation state."""
        self.steps = 0
        obs = {
            agent: np.random.randn(10).astype(np.float32) 
            for agent in self.agents
        }
        return obs

    def step(self, actions: Dict[str, np.ndarray]):
        """
        Execute one tick.
        
        1. Apply physics (Velocity).
        2. Resolve collisions.
        3. Calculate tolls/bids.
        4. Compute rewards.
        """
        self.steps += 1
        
        obs = {}
        rewards = {}
        dones = {}
        infos = {}
        
        global_state_vec = []
        
        for agent_id, action in actions.items():
            # Mock physics update
            current_obs = np.random.randn(10).astype(np.float32)
            obs[agent_id] = current_obs
            
            # Reward Logic (Simplified)
            # Bid is action[3]
            bid = action[3]
            progress = 1.0 # Mock progress
            
            # Reward = Progress - Cost
            rewards[agent_id] = progress - (bid * 0.1)
            
            dones[agent_id] = self.steps >= 100
            infos[agent_id] = {}
            
            global_state_vec.append(current_obs)

        # Global State for Critic
        self.global_state = np.concatenate(global_state_vec)
        
        # All done?
        dones["__all__"] = self.steps >= 100
        
        return obs, rewards, dones, infos

    def get_global_state(self):
        return self.global_state

