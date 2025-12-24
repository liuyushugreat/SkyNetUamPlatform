"""
SkyNet-RWA-Nexus MADDPG Implementation.

This module implements the core Multi-Agent Deep Deterministic Policy Gradient algorithm.
It manages the centralized training and decentralized execution logic.

Algorithm:
    For each agent $i$:
    1. Sample minibatch of $S$ transitions $(x, a, r, x')$ from replay buffer $\mathcal{D}$.
    2. Compute target action for each agent $k$: $a'_k = \mu'_k(o'_k)$.
    3. Compute target Q-value: $y = r_i + \gamma Q'_i(x', a'_1, \dots, a'_N)$.
    4. Update Critic by minimizing loss: $L = \frac{1}{S} \sum (y - Q_i(x, a_1, \dots, a_N))^2$.
    5. Update Actor by maximizing policy gradient:
       $$ \nabla_{\theta_i} J \approx \frac{1}{S} \sum \nabla_{a_i} Q_i(x, a_1, \dots, a_N) \nabla_{\theta_i} \mu_i(o_i) $$
    6. Soft update target networks: $\theta' \leftarrow \tau \theta + (1-\tau)\theta'$.
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Any
from copy import deepcopy
from .networks import ActorNetwork, CriticNetwork

# Hyperparameters
GAMMA = 0.95
TAU = 0.01
LR_ACTOR = 0.01
LR_CRITIC = 0.01
BATCH_SIZE = 1024

class MADDPGAgent:
    """
    Represents a single learning agent within the MADDPG framework.
    """
    def __init__(self, agent_id, obs_dim, action_dim, global_state_dim, total_action_dim):
        self.agent_id = agent_id
        
        # 1. Networks
        self.actor = ActorNetwork(obs_dim, action_dim)
        self.target_actor = deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)

        self.critic = CriticNetwork(global_state_dim, total_action_dim)
        self.target_critic = deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

    def act(self, obs, noise=0.0):
        """Select action based on current policy + exploration noise."""
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs).unsqueeze(0) # Batch size 1
        
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(obs)
        self.actor.train()
        
        # Add exploration noise (e.g., Ornstein-Uhlenbeck or Gaussian)
        if noise > 0:
            action += torch.randn_as(action) * noise
            action = torch.clamp(action, 0, 1) # Bidding percentage
            
        return action.detach().numpy()[0]

    def update_targets(self, tau=TAU):
        """Soft update target parameters."""
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
            
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

class MADDPGTrainer:
    """
    Centralized Trainer. Manages the shared Replay Buffer and updates all agents.
    """
    def __init__(self, agents: List[MADDPGAgent]):
        self.agents = agents
        self.replay_buffer = [] # Simple list for demo; use deque or specialized buffer in prod
        self.buffer_capacity = 100000
        
    def store_transition(self, obs_n, action_n, reward_n, next_obs_n, done_n, global_state, next_global_state):
        """
        Store experience tuple $(x, a, r, x')$.
        Note: '_n' suffix denotes list across all agents.
        """
        if len(self.replay_buffer) >= self.buffer_capacity:
            self.replay_buffer.pop(0)
            
        self.replay_buffer.append((obs_n, action_n, reward_n, next_obs_n, done_n, global_state, next_global_state))

    def update(self):
        """Performs a single gradient update step for all agents."""
        if len(self.replay_buffer) < BATCH_SIZE:
            return

        # Sample batch
        indices = np.random.choice(len(self.replay_buffer), BATCH_SIZE, replace=False)
        batch = [self.replay_buffer[i] for i in indices]
        
        # Unpack batch
        # This part requires careful tensor manipulation
        obs_n_batch = list(zip(*[x[0] for x in batch])) # List of BATCH arrays per agent
        action_n_batch = list(zip(*[x[1] for x in batch]))
        reward_n_batch = list(zip(*[x[2] for x in batch]))
        next_obs_n_batch = list(zip(*[x[3] for x in batch]))
        done_n_batch = list(zip(*[x[4] for x in batch]))
        global_state_batch = torch.FloatTensor(np.array([x[5] for x in batch]))
        next_global_state_batch = torch.FloatTensor(np.array([x[6] for x in batch]))

        # Convert to Tensors
        obs_n_tensors = [torch.FloatTensor(np.array(o)) for o in obs_n_batch]
        next_obs_n_tensors = [torch.FloatTensor(np.array(o)) for o in next_obs_n_batch]
        action_n_tensors = [torch.FloatTensor(np.array(a)) for a in action_n_batch]
        reward_n_tensors = [torch.FloatTensor(np.array(r)).unsqueeze(1) for r in reward_n_batch]
        done_n_tensors = [torch.FloatTensor(np.array(d)).unsqueeze(1) for d in done_n_batch]

        # ---------------------------------------------------------
        # Train each agent
        # ---------------------------------------------------------
        for i, agent in enumerate(self.agents):
            
            # --- 1. Update Critic ---
            
            # Compute target actions for next state (from Target Actors)
            with torch.no_grad():
                target_actions_n = [ag.target_actor(next_obs_n_tensors[j]) for j, ag in enumerate(self.agents)]
                target_actions_cat = torch.cat(target_actions_n, dim=1)
                
                # Compute Q_target
                q_next = agent.target_critic(next_global_state_batch, target_actions_cat)
                y = reward_n_tensors[i] + GAMMA * q_next * (1 - done_n_tensors[i])
            
            # Current Q
            actions_cat = torch.cat(action_n_tensors, dim=1)
            q_current = agent.critic(global_state_batch, actions_cat)
            
            critic_loss = F.mse_loss(q_current, y)
            
            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            agent.critic_optimizer.step()

            # --- 2. Update Actor ---
            
            # We want to maximize Q, so we minimize -Q
            # Action for agent i comes from CURRENT policy
            curr_action_i = agent.actor(obs_n_tensors[i])
            
            # Other agents' actions come from buffer (or current policy? MADDPG usually uses buffer for others in gradient, but standard is current pol)
            # Standard: Use current policy for ALL agents to compute gradient
            all_current_actions = []
            for j, ag in enumerate(self.agents):
                if i == j:
                    all_current_actions.append(curr_action_i)
                else:
                    # Detach others to don't update them
                    all_current_actions.append(ag.actor(obs_n_tensors[j]).detach())
            
            actions_cat_pol = torch.cat(all_current_actions, dim=1)
            
            actor_loss = -agent.critic(global_state_batch, actions_cat_pol).mean()
            
            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            agent.actor_optimizer.step()

            # --- 3. Soft Updates ---
            agent.update_targets()

