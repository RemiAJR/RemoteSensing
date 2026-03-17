"""
losses/barlow_twins_pixel.py — Pixel-level Barlow Twins loss for dense SSL.

Algorithm (Pix2Rep-v2 adaptation of Barlow Twins [Zbontar et al. 2021]):

  Given two dense projection maps Z, Z' ∈ (B, D, H, W):
  1. Optionally warp Z to Z' coordinates using affine metadata from augmentation.
  2. Sample M random pixel coordinates from valid transformed regions.
  3. Gather per-pixel embeddings → Z_M, Z'_M ∈ (N, D) where N = B × M.
  4. Normalise each feature dimension to zero mean and unit std across N.
  5. Compute cross-correlation matrix C = Z_M^T · Z'_M / N  ∈ (D, D).
  6. Barlow Twins objective:
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
        if proj_dim <= 0:
            raise ValueError(f"proj_dim must be > 0, got {proj_dim}")
        if n_pixels <= 0:
            raise ValueError(f"n_pixels must be > 0, got {n_pixels}")
        self.D = proj_dim
        self.M = n_pixels
        self.lam = lambda_

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _valid_mask(z_prime: torch.Tensor) -> torch.Tensor:
        """
        Fallback validity mask from non-zero activations.
        Prefer geometry-derived masks when affine metadata is available.
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
            if flat.numel() == 0:
                raise ValueError(
                    "No valid pixels available for sampling. "
                    "Check spatial transform settings and correspondence masking."
                )
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
        std = z.std(dim=0, keepdim=True, unbiased=False).clamp(min=1e-6)
        return (z - mean) / std

    @staticmethod
    def _warp_dense_map(z: torch.Tensor, theta_vprime_to_v: torch.Tensor) -> torch.Tensor:
        """
        Warp dense map z into the coordinate frame of z_prime.

        `theta_vprime_to_v` maps output (v' frame) coordinates to source (v frame)
        coordinates, i.e. the affine_grid/grid_sample convention.
        """
        if theta_vprime_to_v.ndim != 3 or theta_vprime_to_v.shape[1:] != (2, 3):
            raise ValueError(
                "theta_vprime_to_v must have shape (B, 2, 3), "
                f"got {tuple(theta_vprime_to_v.shape)}"
            )
        if theta_vprime_to_v.shape[0] != z.shape[0]:
            raise ValueError(
                "theta_vprime_to_v batch size must match z batch size "
                f"({theta_vprime_to_v.shape[0]} != {z.shape[0]})"
            )
        grid = F.affine_grid(theta_vprime_to_v, z.shape, align_corners=False)
        return F.grid_sample(
            z, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )

    @staticmethod
    def _valid_mask_from_theta(
        theta_vprime_to_v: torch.Tensor,
        h: int,
        w: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Build a validity mask in v' coordinates from affine transform geometry only.
        """
        ones = torch.ones(theta_vprime_to_v.shape[0], 1, h, w, device=device)
        grid = F.affine_grid(theta_vprime_to_v, ones.shape, align_corners=False)
        warped = F.grid_sample(
            ones, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        return warped.squeeze(1) > 0.999

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        z: torch.Tensor,
        z_prime: torch.Tensor,
        theta_vprime_to_v: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute pixel-level Barlow Twins loss.

        Args:
            z:                 (B, D, H, W) — projection of view v
            z_prime:           (B, D, H, W) — projection of spatially-warped view v'
            theta_vprime_to_v: optional (B, 2, 3). If provided, z is warped into v'
                                coordinates before matching (dense correspondence).
            valid_mask:        optional (B, H, W) bool mask in v' coordinates.

        Returns:
            loss: scalar tensor
        """
        if z.shape != z_prime.shape:
            raise ValueError(
                f"z and z_prime must have the same shape, got {tuple(z.shape)} "
                f"and {tuple(z_prime.shape)}"
            )
        B, D, H, W = z.shape
        if D != self.D:
            raise ValueError(f"Expected proj dim {self.D}, got {D}")

        # Align z to z_prime coordinates when affine metadata is available.
        if theta_vprime_to_v is not None:
            z = self._warp_dense_map(z, theta_vprime_to_v)

        # 1. Build valid mask in z_prime coordinates.
        if valid_mask is not None:
            if valid_mask.shape != (B, H, W):
                raise ValueError(
                    "valid_mask must have shape (B, H, W), "
                    f"got {tuple(valid_mask.shape)}"
                )
            mask = valid_mask.bool()
        elif theta_vprime_to_v is not None:
            mask = self._valid_mask_from_theta(theta_vprime_to_v, H, W, z.device)
        else:
            # Fallback for backward compatibility when no transform is provided.
            mask = self._valid_mask(z_prime)

        # 2. Sample M pixel coordinates per image
        coords = self._sample_pixels(mask, self.M)    # (B, M, 2)

        # 3. Gather pixel embeddings at sampled locations
        zm = self._gather_pixels(z, coords)           # (N, D), N = B*M
        zpm = self._gather_pixels(z_prime, coords)    # (N, D)

        # 4. Normalise
        zm  = self._normalise(zm)
        zpm = self._normalise(zpm)

        # 5. Cross-correlation matrix (D, D)
        N = zm.shape[0]
        C = (zm.T @ zpm) / N                          # (D, D)

        # 6. Barlow Twins loss
        on_diag = torch.diagonal(C)
        off_diag = C[~torch.eye(D, dtype=torch.bool, device=C.device)]

        loss = (1.0 - on_diag).pow(2).sum() + self.lam * off_diag.pow(2).sum()

        return loss
