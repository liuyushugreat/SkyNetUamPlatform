"""
Neural pricing models (Phase-1 — retained).

These PyTorch models are **not** deprecated; they remain the core neural
pricing primitives.  To use them within the V2 asset-valuation pipeline,
wrap them with :class:`~SkyRwa.valuation.neural_adapter.NeuralValuationAdapter`
which bridges the ``nn.Module.forward()`` interface into the
``AbstractAssetValuationEngine.evaluate()`` contract.

TODO(adapter): train a production model and load weights in NeuralValuationAdapter.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple

class CyclicEmbedding(nn.Module):
    """
    Learnable Trigonometric Embedding Layer.
    Initializes weights with E_a = [cos(2*pi*f*a/n), sin(2*pi*f*a/n)].
    """
    def __init__(self, num_embeddings: int, embedding_dim: int = 2, frequency: float = 1.0, trainable: bool = True):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        
        # Initialize with trigonometric values
        # Only works intuitively for dim=2 or multiples. 
        # Here we implement the base [cos, sin] for dim=2 as requested.
        if embedding_dim == 2:
            with torch.no_grad():
                indices = torch.arange(num_embeddings).float()
                angle = 2 * np.pi * frequency * indices / num_embeddings
                weights = torch.stack([torch.cos(angle), torch.sin(angle)], dim=1)
                self.embedding.weight.copy_(weights)
        
        self.embedding.weight.requires_grad = trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)

class NeuralPricingModel(nn.Module):
    """Base class for Pricing Models with pre-activation hooks."""
    def __init__(self):
        super().__init__()
        self.h_pre: Optional[torch.Tensor] = None

    def _hook_pre_activation(self, module, input, output):
        # This hook captures the linear output before activation
        self.h_pre = output

class PizzaPricingModel(NeuralPricingModel):
    """
    MLP-Add Architecture ('Pizza' Topology).
    Sums the embeddings of inputs before passing to MLP.
    Analogous to 'Attention 0.0'.
    """
    def __init__(self, time_mod: int = 24, route_mod: int = 60, hidden_dim: int = 128):
        super().__init__()
        
        # 1. Embeddings (2D circle)
        self.time_emb = CyclicEmbedding(time_mod, 2, trainable=True)
        self.route_emb = CyclicEmbedding(route_mod, 2, trainable=True)
        
        # 2. MLP (Input dim = 2 because embeddings are summed)
        self.fc1 = nn.Linear(2, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1) # Scalar output (Price)
        
        # Register hook to capture pre-activations of the first hidden layer
        self.fc1.register_forward_hook(self._hook_pre_activation)

    def forward(self, time_idx: torch.Tensor, route_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time_idx: Tensor of shape (batch_size,) with integers 0-(time_mod-1)
            route_idx: Tensor of shape (batch_size,) with integers 0-(route_mod-1)
        """
        e_time = self.time_emb(time_idx)
        e_route = self.route_emb(route_idx)
        
        # Sum embeddings
        x = e_time + e_route
        
        # MLP
        # h_pre is captured by hook in fc1
        h = self.fc1(x) 
        # Note: self.fc1(x) triggers the hook which sets self.h_pre = h
        # But wait, 'output' in hook is the result of forward.
        # So self.h_pre will be 'h'.
        
        x_act = self.act(h)
        out = self.fc2(x_act)
        return out

class TorusPricingModel(NeuralPricingModel):
    """
    MLP-Concat Architecture ('Torus' Topology).
    Concatenates the embeddings of inputs before passing to MLP.
    """
    def __init__(self, time_mod: int = 24, route_mod: int = 60, hidden_dim: int = 128):
        super().__init__()
        
        # 1. Embeddings
        self.time_emb = CyclicEmbedding(time_mod, 2, trainable=True)
        self.route_emb = CyclicEmbedding(route_mod, 2, trainable=True)
        
        # 2. MLP (Input dim = 2 + 2 = 4 because embeddings are concatenated)
        self.fc1 = nn.Linear(4, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        
        # Register hook
        self.fc1.register_forward_hook(self._hook_pre_activation)

    def forward(self, time_idx: torch.Tensor, route_idx: torch.Tensor) -> torch.Tensor:
        e_time = self.time_emb(time_idx)
        e_route = self.route_emb(route_idx)
        
        # Concatenate embeddings
        x = torch.cat([e_time, e_route], dim=-1)
        
        # MLP
        h = self.fc1(x)
        x_act = self.act(h)
        out = self.fc2(x_act)
        return out

