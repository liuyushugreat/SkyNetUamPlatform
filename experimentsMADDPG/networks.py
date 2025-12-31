"""
Enhanced Neural Network Architectures for Multi-Agent RL.

This module provides:
1. Attention-based Critic Network (for AP-MADDPG and MAPPO)
2. Standard Actor Network (shared across algorithms)
3. Support for vectorized environments (50k+ agents)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class ActorNetwork(nn.Module):
    """
    Policy Network: π: O → A
    Maps local observation to continuous action space.
    """
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, action_dim)
        
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.out(x))


class AttentionCriticNetwork(nn.Module):
    """
    Attention-based Critic Network for scalable multi-agent learning.
    
    Implements "Perceptual Filtering & Interaction Attention" as described in the paper.
    Uses multi-head attention to dynamically filter non-critical interactions,
    reducing effective input dimension from N to K (key neighbors).
    
    Reference: Paper Section 4.2
    """
    def __init__(self, obs_dim, action_dim, num_agents, hidden_dim=64, 
                 num_heads=4, key_neighbors=10):
        super(AttentionCriticNetwork, self).__init__()
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.key_neighbors = key_neighbors
        self.d_k = hidden_dim // num_heads
        
        # Embedding layers for Query, Key, Value
        self.query_proj = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.key_proj = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.value_proj = nn.Linear(obs_dim + action_dim, hidden_dim)
        
        # Self-embedding (for agent i)
        self.self_embed = nn.Linear(obs_dim + action_dim, hidden_dim)
        
        # Multi-head attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Post-attention processing
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)  # *2 for self + attended
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
        
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.self_embed.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, obs_n, action_n, agent_idx=0):
        """
        Args:
            obs_n: List of observations for all agents (batch_size, obs_dim) each
            action_n: List of actions for all agents (batch_size, action_dim) each
            agent_idx: Index of the agent whose value we're computing
        
        Returns:
            Q-value tensor (batch_size, 1)
        """
        batch_size = obs_n[0].shape[0]
        
        # Prepare inputs: concatenate obs and action for each agent
        agent_inputs = []
        for obs, act in zip(obs_n, action_n):
            agent_inputs.append(torch.cat([obs, act], dim=1))
        
        # Stack all agents: (batch_size, num_agents, obs_dim + action_dim)
        all_agents = torch.stack(agent_inputs, dim=1)
        
        # Self-embedding for agent i
        self_input = agent_inputs[agent_idx]  # (batch_size, obs_dim + action_dim)
        self_emb = self.self_embed(self_input)  # (batch_size, hidden_dim)
        
        # Compute Query, Key, Value for attention
        queries = self.query_proj(all_agents)  # (batch_size, num_agents, hidden_dim)
        keys = self.key_proj(all_agents)
        values = self.value_proj(all_agents)
        
        # Extract query for agent i
        query_i = queries[:, agent_idx:agent_idx+1, :]  # (batch_size, 1, hidden_dim)
        
        # Multi-head attention: agent i attends to all agents
        attn_output, attn_weights = self.multihead_attn(
            query_i, keys, values, average_attn_weights=False
        )
        # attn_output: (batch_size, 1, hidden_dim)
        # attn_weights: (batch_size, 1, num_agents, num_heads)
        
        attn_output = attn_output.squeeze(1)  # (batch_size, hidden_dim)
        
        # Concatenate self-embedding with attended output
        combined = torch.cat([self_emb, attn_output], dim=1)  # (batch_size, hidden_dim * 2)
        
        # Final processing
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        return self.out(x), attn_weights


class StandardCriticNetwork(nn.Module):
    """
    Standard Critic Network (for baseline comparisons).
    Uses full concatenation without attention.
    """
    def __init__(self, full_state_dim, action_dim_total, hidden_dim=64):
        super(StandardCriticNetwork, self).__init__()
        self.fc1 = nn.Linear(full_state_dim + action_dim_total, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
        
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, state, actions):
        x = torch.cat([state, actions], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

