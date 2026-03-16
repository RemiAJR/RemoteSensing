"""
losses/barlow_twins_pixel.py — Pixel-level Barlow Twins loss for dense SSL.

Algorithm (Pix2Rep-v2 adaptation of Barlow Twins [Zbontar et al. 2021]):

  Given two dense projection maps Z, Z' ∈ (B, D, H, W):
  1. Sample M random pixel coordinates that lie within the spatially transformed
     region (non-zero mask of Z').
  2. Gather per-pixel embeddings → Z_M, Z'_M ∈ (N, D) where N = B × M.
  3. Normalise each feature dimension to zero mean and unit std across N.
  4. Compute cross-correlation matrix C = Z_M^T · Z'_M / N  ∈ (D, D).
  5. Barlow Twins objective:
       L = Σ_i (1 - C_ii)²  +  λ Σ_{i≠j} C_ij²

No memory bank, no queue, no large batch: memory cost is O(D²).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelBarlowTwinsLoss(nn.Module):
    """
    Pixel-level Barlow Twins loss.

    Args:
        proj_dim:  embedding dimensionality D (e.g. 256)
        n_pixels:  number of pixels M sampled per image (e.g. 1000)
        lambda_:   weight of the off-diagonal regularisation term (e.g. 5e-3)
    """

    def __init__(self, proj_dim: int = 256, n_pixels: int = 1000, lambda_: float = 5e-3):
        super().__init__()
        self.D = proj_dim
        self.M = n_pixels
        self.lam = lambda_

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _valid_mask(z_prime: torch.Tensor) -> torch.Tensor:
        """
        Boolean mask (B, H, W) indicating pixels where z_prime is non-zero
        (i.e. inside the spatial transform bounding box).
        """
        # A pixel is 'valid' if at least one channel is non-zero
        return z_prime.abs().sum(dim=1) > 0          # (B, H, W)

    @staticmethod
    def _sample_pixels(
        mask: torch.Tensor,
        M: int,
    ) -> torch.Tensor:
        """
        Sample M pixel indices per image from valid locations.

        Args:
            mask: (B, H, W) bool
            M:    number of pixels to sample per image

        Returns:
            coords: (B, M, 2) int64 tensor of (row, col) indices
        """
        B, H, W = mask.shape
        coords = torch.zeros(B, M, 2, dtype=torch.long, device=mask.device)
        for b in range(B):
            flat = mask[b].reshape(-1).nonzero(as_tuple=False).squeeze(1)  # (K,)
            if flat.numel() < M:
                # Fewer valid pixels than M — sample with replacement
                idx = flat[torch.randint(flat.numel(), (M,), device=mask.device)]
            else:
                perm = torch.randperm(flat.numel(), device=mask.device)[:M]
                idx = flat[perm]
            rows = idx // W
            cols = idx % W
            coords[b, :, 0] = rows
            coords[b, :, 1] = cols
        return coords

    @staticmethod
    def _gather_pixels(z: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Gather per-pixel embeddings from z at the given coordinates.

        Args:
            z:      (B, D, H, W)
            coords: (B, M, 2) int64 (row, col)

        Returns:
            embeddings: (B*M, D)
        """
        B, D, H, W = z.shape
        M = coords.shape[1]
        # Build a flat linear index into the H×W grid
        rows = coords[:, :, 0]   # (B, M)
        cols = coords[:, :, 1]   # (B, M)
        flat_idx = rows * W + cols  # (B, M)

        # Reshape z to (B, D, H*W) and gather
        z_flat = z.reshape(B, D, H * W)                # (B, D, H*W)
        flat_idx_exp = flat_idx.unsqueeze(1).expand(B, D, M)  # (B, D, M)
        gathered = z_flat.gather(2, flat_idx_exp)      # (B, D, M)
        gathered = gathered.permute(0, 2, 1)           # (B, M, D)
        return gathered.reshape(B * M, D)              # (B*M, D)

    @staticmethod
    def _normalise(z: torch.Tensor) -> torch.Tensor:
        """Normalise each feature dimension to zero mean and unit std."""
        mean = z.mean(dim=0, keepdim=True)
        std  = z.std(dim=0, keepdim=True).clamp(min=1e-6)
        return (z - mean) / std

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        z: torch.Tensor,
        z_prime: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute pixel-level Barlow Twins loss.

        Args:
            z:       (B, D, H, W) — projection of view v
            z_prime: (B, D, H, W) — projection of spatially-warped view v'

        Returns:
            loss: scalar tensor
        """
        B, D, H, W = z.shape

        # 1. Build valid mask from z_prime (zero-padded regions are invalid)
        mask = self._valid_mask(z_prime)              # (B, H, W)

        # 2. Sample M pixel coordinates per image
        coords = self._sample_pixels(mask, self.M)   # (B, M, 2)

        # 3. Gather pixel embeddings at sampled locations
        zm  = self._gather_pixels(z,       coords)   # (N, D), N = B*M
        zpm = self._gather_pixels(z_prime, coords)   # (N, D)

        # 4. Normalise
        zm  = self._normalise(zm)
        zpm = self._normalise(zpm)

        # 5. Cross-correlation matrix (D, D)
        N = zm.shape[0]
        C = (zm.T @ zpm) / N                         # (D, D)

        # 6. Barlow Twins loss
        on_diag  = torch.diagonal(C)
        off_diag = C.flatten()[1:].view(D - 1, D + 1)[:, :-1]  # all off-diagonal elements

        loss = (1.0 - on_diag).pow(2).sum() \
             + self.lam * off_diag.pow(2).sum()

        return loss
