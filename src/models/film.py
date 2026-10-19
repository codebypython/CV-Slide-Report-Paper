"""Feature-wise Linear Modulation (FiLM) layer for clinical metadata conditioning.

Mathematical Formulation:
Given visual representation feature x and clinical metadata vector m (biological sex):
    [gamma(m), beta(m)] = MLP(m)
    y = (1.0 + gamma(m)) * x + beta(m)

The (1.0 + gamma) residual scaling formulation ensures identity initialization stability.
"""

import torch
import torch.nn as nn
from typing import Optional


class FiLMLayer(nn.Module):
    """Conditions latent visual features using clinical categorical/continuous metadata."""
    def __init__(self, feature_dim: int, metadata_dim: int = 1, hidden_dim: int = 64):
        super().__init__()
        self.feature_dim = feature_dim
        self.metadata_dim = metadata_dim

        self.generator = nn.Sequential(
            nn.Linear(metadata_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim * 2)
        )

        # Initialize the last linear layer weights near zero for gentle initial modulation
        nn.init.zeros_(self.generator[-1].weight)
        nn.init.zeros_(self.generator[-1].bias)

    def forward(self, x: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        """Modulates feature tensor x with clinical metadata meta.
        
        Args:
            x: Tensor of shape [B, D] or [B, L, D] or [B, D, H, W]
            meta: Metadata tensor of shape [B, metadata_dim]
        """
        film_params = self.generator(meta)  # [B, feature_dim * 2]
        gamma, beta = torch.chunk(film_params, 2, dim=-1)  # [B, feature_dim] each

        if x.dim() == 2:
            # [B, D]
            return x * (1.0 + gamma) + beta
        elif x.dim() == 3:
            # [B, L, D] (Sequence of tokens)
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
            return x * (1.0 + gamma) + beta
        elif x.dim() == 4:
            # [B, C, H, W] (Spatial feature map)
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)
            beta = beta.unsqueeze(-1).unsqueeze(-1)
            return x * (1.0 + gamma) + beta
        else:
            raise ValueError(f"Unsupported input dimension for FiLMLayer: {x.dim()}")
