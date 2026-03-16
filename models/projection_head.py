"""
models/projection_head.py — Dense 1×1-conv MLP projection head.

Maps (B, embed_dim, H, W) → (B, proj_dim, H, W).
Applied during SSL pretraining only; discarded at fine-tune time.
"""

import torch
import torch.nn as nn


class DenseProjectionHead(nn.Module):
    """
    Two-layer 1×1-conv MLP with BN + ReLU in the hidden layer.

    Architecture: embed_dim → hidden_dim (BN + ReLU) → proj_dim
    This preserves spatial resolution and outputs a per-pixel embedding.

    Args:
        embed_dim:  input channels (U-Net output, e.g. 1024)
        hidden_dim: hidden-layer channels (e.g. 512)
        proj_dim:   output channels used for the Barlow Twins loss (e.g. 256)
    """

    def __init__(self, embed_dim: int = 1024, hidden_dim: int = 512, proj_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, proj_dim, 1, bias=False),
            nn.BatchNorm2d(proj_dim),           # BN on output (no ReLU, per SimCLR/BT convention)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, embed_dim, H, W)

        Returns:
            z: (B, proj_dim, H, W)
        """
        return self.net(x)
