"""
SkyNet-RWA-Nexus: Neural Network Architectures for MADDPG.

Academic Reference:
    Lowe, R., Wu, Y., Tamar, A., Harb, J., Abbeel, P., & Mordatch, I. (2017). 
    Multi-agent actor-critic for mixed cooperative-competitive environments. 
    Neural Information Processing Systems (NIPS).

Architectural Choices:
    - Actor $\mu(o)$: Maps local observation to continuous action space (Bid Ratio).
    - Critic $Q(s, a)$: Maps global state and joint actions to Q-value.
    - Hidden Layers: 64x64 with ReLU activation (Standard for low-dim inputs).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ActorNetwork(nn.Module):
    """
    Polcy Network: $\pi: S \rightarrow A$
    """
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, action_dim)
        
        # Initialize weights with Xavier uniform
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x):
        """
        Forward pass.
        Returns actions bounded by Tanh (e.g., -1 to 1) or Sigmoid (0 to 1).
        Here we use Sigmoid for bidding percentage (0% to 100% of budget).
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.out(x))

class CriticNetwork(nn.Module):
    """
    Value Network: $Q: (S, A_{all}) \rightarrow \mathbb{R}$
    """
    def __init__(self, full_state_dim, action_dim_total, hidden_dim=64):
        super(CriticNetwork, self).__init__()
        # Input is concatenation of global state + actions of all agents
        self.fc1 = nn.Linear(full_state_dim + action_dim_total, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
        
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, state, actions):
        """
        Args:
            state: Global state tensor (batch_size, state_dim)
            actions: Concatenated actions of all agents (batch_size, act_dim_total)
        """
        x = torch.cat([state, actions], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

