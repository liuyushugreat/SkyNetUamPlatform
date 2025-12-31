"""
Multi-Agent Proximal Policy Optimization (MAPPO) Implementation.

This module implements MAPPO with:
1. Attention-based Critic (shared architecture with AP-MADDPG for fair comparison)
2. Centralized Training, Decentralized Execution (CTDE)
3. Support for vectorized environments (50k+ agents)

Reference:
    Yu, C., Velu, A., Vinitsky, E., et al. (2021). 
    The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games.
    Neural Information Processing Systems (NeurIPS).

Algorithm:
    For each agent i:
    1. Collect trajectories using current policy π_θ
    2. Compute advantages using centralized value function V_φ
    3. Update policy: maximize clipped objective
       L^CLIP(θ) = E[min(r_t(θ)Â_t, clip(r_t(θ), 1-ε, 1+ε)Â_t)]
    4. Update value function: minimize MSE(V_φ, returns)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque
from .networks import ActorNetwork, AttentionCriticNetwork


# Hyperparameters
GAMMA = 0.99
LAMBDA = 0.95  # GAE lambda
EPSILON = 0.2  # PPO clip parameter
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
LR_ACTOR = 3e-4
LR_CRITIC = 3e-4
MAX_GRAD_NORM = 0.5
UPDATE_EPOCHS = 10
BATCH_SIZE = 64


class MAPPOAgent:
    """
    Single agent within MAPPO framework.
    Each agent has its own actor (policy) but shares centralized critic during training.
    """
    def __init__(self, agent_id, obs_dim, action_dim, num_agents, 
                 use_attention=True, hidden_dim=64):
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Actor network (decentralized policy)
        self.actor = ActorNetwork(obs_dim, action_dim, hidden_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        
        # Note: Critic is shared across agents (managed by MAPPOTrainer)
        # We store a reference here for convenience
        self.critic = None
        
    def get_action(self, obs, deterministic=False):
        """
        Sample action from policy.
        
        Args:
            obs: Observation tensor (batch_size, obs_dim) or (obs_dim,)
            deterministic: If True, return mean action; else sample
        
        Returns:
            action: Action tensor
            log_prob: Log probability of action
            value: Estimated value (if critic available)
        """
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)
        
        # Get action distribution
        action_mean = self.actor(obs)
        
        # For continuous actions, we use a normal distribution
        # In practice, you might want to use TanhNormal or similar
        action_std = torch.ones_like(action_mean) * 0.1  # Fixed std for simplicity
        dist = torch.distributions.Normal(action_mean, action_std)
        
        if deterministic:
            action = action_mean
        else:
            action = dist.sample()
        
        # Clip to valid range [0, 1]
        action = torch.clamp(action, 0, 1)
        
        # Compute log probability
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        return action, log_prob
    
    def evaluate_actions(self, obs, actions):
        """
        Evaluate actions under current policy (for PPO update).
        
        Returns:
            log_probs: Log probabilities of actions
            entropy: Entropy of action distribution
        """
        action_mean = self.actor(obs)
        action_std = torch.ones_like(action_mean) * 0.1
        dist = torch.distributions.Normal(action_mean, action_std)
        
        log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        return log_probs, entropy


class MAPPOTrainer:
    """
    Centralized trainer for MAPPO.
    Manages shared value function (critic) and coordinates policy updates.
    """
    def __init__(self, agents: List[MAPPOAgent], obs_dim, action_dim, 
                 num_agents, use_attention=True, hidden_dim=64):
        self.agents = agents
        self.num_agents = num_agents
        self.use_attention = use_attention
        
        # Shared centralized value function (critic)
        if use_attention:
            # Use attention-based critic (same as AP-MADDPG)
            self.critic = AttentionCriticNetwork(
                obs_dim, action_dim, num_agents, hidden_dim
            )
        else:
            # Fallback to standard critic
            from .networks import StandardCriticNetwork
            full_state_dim = obs_dim * num_agents
            action_dim_total = action_dim * num_agents
            self.critic = StandardCriticNetwork(full_state_dim, action_dim_total, hidden_dim)
        
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        
        # Assign critic reference to agents
        for agent in agents:
            agent.critic = self.critic
        
        # Experience buffer
        self.reset_buffer()
    
    def reset_buffer(self):
        """Reset experience buffer for new episode."""
        self.obs_buffer = [[] for _ in range(self.num_agents)]
        self.action_buffer = [[] for _ in range(self.num_agents)]
        self.reward_buffer = [[] for _ in range(self.num_agents)]
        self.log_prob_buffer = [[] for _ in range(self.num_agents)]
        self.value_buffer = [[] for _ in range(self.num_agents)]
        self.done_buffer = [[] for _ in range(self.num_agents)]
        self.global_state_buffer = []
    
    def store_transition(self, obs_n, action_n, reward_n, log_prob_n, 
                        value_n, done_n, global_state):
        """
        Store transition in buffer.
        
        Args:
            obs_n: List of observations for each agent
            action_n: List of actions for each agent
            reward_n: List of rewards for each agent
            log_prob_n: List of log probabilities for each agent
            value_n: List of value estimates for each agent
            done_n: List of done flags for each agent
            global_state: Global state tensor
        """
        for i in range(self.num_agents):
            self.obs_buffer[i].append(obs_n[i])
            self.action_buffer[i].append(action_n[i])
            self.reward_buffer[i].append(reward_n[i])
            self.log_prob_buffer[i].append(log_prob_n[i])
            self.value_buffer[i].append(value_n[i])
            self.done_buffer[i].append(done_n[i])
        self.global_state_buffer.append(global_state)
    
    def compute_gae(self, rewards, values, dones, next_value=0.0):
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: List of rewards (T,)
            values: List of value estimates (T,)
            dones: List of done flags (T,)
            next_value: Value estimate for next state
        
        Returns:
            advantages: GAE advantages (T,)
            returns: Discounted returns (T,)
        """
        T = len(rewards)
        advantages = torch.zeros(T)
        returns = torch.zeros(T)
        
        gae = 0
        for t in reversed(range(T)):
            if t == T - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value_t = next_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value_t = values[t + 1]
            
            delta = rewards[t] + GAMMA * next_value_t * next_non_terminal - values[t]
            gae = delta + GAMMA * LAMBDA * next_non_terminal * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        return advantages, returns
    
    def update(self):
        """
        Perform PPO update on collected trajectories.
        """
        if len(self.obs_buffer[0]) == 0:
            return {}
        
        # Convert buffers to tensors
        obs_n_tensors = []
        action_n_tensors = []
        old_log_prob_n_tensors = []
        reward_n_tensors = []
        value_n_tensors = []
        done_n_tensors = []
        
        for i in range(self.num_agents):
            obs_n_tensors.append(torch.stack(self.obs_buffer[i]))
            action_n_tensors.append(torch.stack(self.action_buffer[i]))
            old_log_prob_n_tensors.append(torch.stack(self.log_prob_buffer[i]))
            reward_n_tensors.append(torch.tensor(self.reward_buffer[i], dtype=torch.float32))
            value_n_tensors.append(torch.tensor(self.value_buffer[i], dtype=torch.float32))
            done_n_tensors.append(torch.tensor(self.done_buffer[i], dtype=torch.float32))
        
        global_state_tensor = torch.stack(self.global_state_buffer)
        
        # Compute advantages and returns for each agent
        advantages_n = []
        returns_n = []
        
        for i in range(self.num_agents):
            # Get next value (0 if episode ended)
            next_value = 0.0 if done_n_tensors[i][-1] else value_n_tensors[i][-1]
            
            adv, ret = self.compute_gae(
                reward_n_tensors[i],
                value_n_tensors[i],
                done_n_tensors[i],
                next_value
            )
            advantages_n.append(adv)
            returns_n.append(ret)
        
        # Normalize advantages
        for i in range(self.num_agents):
            advantages_n[i] = (advantages_n[i] - advantages_n[i].mean()) / (advantages_n[i].std() + 1e-8)
        
        # Prepare data for mini-batch updates
        T = len(obs_n_tensors[0])
        indices = np.arange(T)
        
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        
        # PPO update epochs
        for epoch in range(UPDATE_EPOCHS):
            np.random.shuffle(indices)
            
            for start in range(0, T, BATCH_SIZE):
                end = start + BATCH_SIZE
                batch_indices = indices[start:end]
                
                # Update each agent
                for i, agent in enumerate(self.agents):
                    obs_batch = obs_n_tensors[i][batch_indices]
                    action_batch = action_n_tensors[i][batch_indices]
                    old_log_prob_batch = old_log_prob_n_tensors[i][batch_indices]
                    adv_batch = advantages_n[i][batch_indices].unsqueeze(1)
                    ret_batch = returns_n[i][batch_indices].unsqueeze(1)
                    
                    # Evaluate actions under current policy
                    new_log_probs, entropy = agent.evaluate_actions(obs_batch, action_batch)
                    
                    # Compute ratio
                    ratio = torch.exp(new_log_probs - old_log_prob_batch)
                    
                    # PPO clipped objective
                    surr1 = ratio * adv_batch
                    surr2 = torch.clamp(ratio, 1.0 - EPSILON, 1.0 + EPSILON) * adv_batch
                    actor_loss = -torch.min(surr1, surr2).mean()
                    
                    # Update actor
                    agent.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), MAX_GRAD_NORM)
                    agent.actor_optimizer.step()
                    
                    total_actor_loss += actor_loss.item()
                    total_entropy += entropy.mean().item()
                
                # Update centralized critic
                if self.use_attention:
                    # Attention-based critic
                    value_preds = []
                    for i in range(self.num_agents):
                        obs_batch_list = [obs_n_tensors[j][batch_indices] for j in range(self.num_agents)]
                        action_batch_list = [action_n_tensors[j][batch_indices] for j in range(self.num_agents)]
                        value_pred, _ = self.critic(obs_batch_list, action_batch_list, i)
                        value_preds.append(value_pred)
                    
                    # Average value across agents (or use agent-specific)
                    value_pred = value_preds[0]  # Use first agent's value for simplicity
                else:
                    # Standard critic
                    global_state_batch = global_state_tensor[batch_indices]
                    actions_cat = torch.cat([action_n_tensors[j][batch_indices] for j in range(self.num_agents)], dim=1)
                    value_pred = self.critic(global_state_batch, actions_cat)
                
                # Critic loss (use average return across agents)
                ret_batch = torch.stack([returns_n[i][batch_indices].unsqueeze(1) for i in range(self.num_agents)]).mean(dim=0)
                critic_loss = F.mse_loss(value_pred, ret_batch)
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), MAX_GRAD_NORM)
                self.critic_optimizer.step()
                
                total_critic_loss += critic_loss.item()
        
        # Reset buffer
        self.reset_buffer()
        
        return {
            'actor_loss': total_actor_loss / (UPDATE_EPOCHS * (T // BATCH_SIZE + 1) * self.num_agents),
            'critic_loss': total_critic_loss / (UPDATE_EPOCHS * (T // BATCH_SIZE + 1)),
            'entropy': total_entropy / (UPDATE_EPOCHS * (T // BATCH_SIZE + 1) * self.num_agents)
        }

