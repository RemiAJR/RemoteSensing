"""
models/unet.py — 2-D U-Net backbone for hyperspectral SSL pretraining.

Architecture
────────────
  Optional spectral-reduction layer  : Conv2d(C_hs, spectral_ch, 1) + BN + ReLU
  Encoder (4 down-blocks)            : (Conv→BN→ReLU) × 2 → MaxPool
  Bottleneck                         : (Conv→BN→ReLU) × 2
  Decoder (4 up-blocks)              : Upsample → concat skip → (Conv→BN→ReLU) × 2
  Output projection layer            : Conv2d(base_ch, embed_dim, 1)

During SSL pretraining the model returns a dense feature map of shape
(B, embed_dim, H, W).  No classification head is attached.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────── building blocks ─────────────────────────────────

def _double_conv(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class _DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = _double_conv(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor):
        feat = self.conv(x)
        return self.pool(feat), feat           # (pooled, skip)


class _UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        # in_ch is the number of channels coming from below (after upsample)
        self.conv = _double_conv(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ───────────────────────────────── U-Net ─────────────────────────────────────

class HyperspectralUNet(nn.Module):
    """
    U-Net backbone adapted for hyperspectral input.

    Args:
        n_bands:        number of input spectral bands (e.g. 239 for PRISMA)
        spectral_ch:    output channels of the spectral-reduction 1×1 conv.
                        Set to 0 to feed raw bands directly into the encoder.
        base_ch:        number of feature maps in the first encoder block.
                        Subsequent blocks double: base_ch → 2× → 4× → 8×.
        embed_dim:      number of output channels (dense embedding per pixel).
    """

    def __init__(
        self,
        n_bands: int = 239,
        spectral_ch: int = 64,
        base_ch: int = 64,
        embed_dim: int = 1024,
    ):
        super().__init__()
        self.use_spectral_reduction = spectral_ch > 0

        # ── Spectral reduction ────────────────────────────────────────────────
        if self.use_spectral_reduction:
            self.spectral_reduce = nn.Sequential(
                nn.Conv2d(n_bands, spectral_ch, 1, bias=False),
                nn.BatchNorm2d(spectral_ch),
                nn.ReLU(inplace=True),
            )
            enc_in = spectral_ch
        else:
            self.spectral_reduce = nn.Identity()
            enc_in = n_bands

        # ── Encoder ───────────────────────────────────────────────────────────
        ch = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]   # [64, 128, 256, 512]

        self.down1 = _DownBlock(enc_in, ch[0])
        self.down2 = _DownBlock(ch[0],  ch[1])
        self.down3 = _DownBlock(ch[1],  ch[2])
        self.down4 = _DownBlock(ch[2],  ch[3])

        # ── Bottleneck ────────────────────────────────────────────────────────
        self.bottleneck = _double_conv(ch[3], ch[3] * 2)        # 512 → 1024

        # ── Decoder ───────────────────────────────────────────────────────────
        # up_i: in_ch comes from below (bottleneck/previous up), skip_ch from encoder
        self.up4 = _UpBlock(ch[3] * 2, ch[3], ch[3])            # 1024+512 → 512
        self.up3 = _UpBlock(ch[3],     ch[2], ch[2])            # 512+256  → 256
        self.up2 = _UpBlock(ch[2],     ch[1], ch[1])            # 256+128  → 128
        self.up1 = _UpBlock(ch[1],     ch[0], ch[0])            # 128+64   → 64

        # ── Output projection ─────────────────────────────────────────────────
        self.out_proj = nn.Conv2d(ch[0], embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_hs, H, W)

        Returns:
            embedding: (B, embed_dim, H, W)
        """
        # Spectral reduction
        x = self.spectral_reduce(x)

        # Encoder
        x1, s1 = self.down1(x)
        x2, s2 = self.down2(x1)
        x3, s3 = self.down3(x2)
        x4, s4 = self.down4(x3)

        # Bottleneck
        b = self.bottleneck(x4)

        # Decoder
        d = self.up4(b,  s4)
        d = self.up3(d,  s3)
        d = self.up2(d,  s2)
        d = self.up1(d,  s1)

        return self.out_proj(d)                 # (B, embed_dim, H, W)
