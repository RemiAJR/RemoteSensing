from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pix2repv2.models.layers import OutConv, get_norm_layer


class SegmentationHead2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        variant: str,
        hidden_channels: int,
        norm: str,
        dropout: float,
    ):
        super().__init__()
        self.variant = variant
        self.dropout = float(dropout)
        if variant == "linear":
            self.net = OutConv(in_channels, out_channels)
        elif variant == "mlp_1x1":
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
                get_norm_layer(norm, hidden_channels, ndim=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
            )
        elif variant == "mlp_1x1_deep":
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
                get_norm_layer(norm, hidden_channels, ndim=2),
                nn.GELU(),
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=False),
                get_norm_layer(norm, hidden_channels, ndim=2),
                nn.GELU(),
                nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
            )
        elif variant == "depthwise3x3_mlp1x1":
            self.net = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    padding=1,
                    groups=in_channels,
                    bias=False,
                ),
                get_norm_layer(norm, in_channels, ndim=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
                get_norm_layer(norm, hidden_channels, ndim=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
            )
        elif variant == "conv3x3_1x1":
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
                get_norm_layer(norm, hidden_channels, ndim=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
            )
        else:
            raise ValueError(f"Unknown segmentation head variant: {variant}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout <= 0:
            return self.net(x)
        if self.variant in {"mlp_1x1", "conv3x3_1x1"}:
            x = self.net[0](x)
            x = self.net[1](x)
            x = self.net[2](x)
            x = F.dropout2d(x, p=self.dropout, training=self.training)
            x = self.net[3](x)
            return x
        if self.variant == "mlp_1x1_deep":
            x = self.net[0](x)
            x = self.net[1](x)
            x = self.net[2](x)
            x = F.dropout2d(x, p=self.dropout, training=self.training)
            x = self.net[3](x)
            x = self.net[4](x)
            x = self.net[5](x)
            x = F.dropout2d(x, p=self.dropout, training=self.training)
            x = self.net[6](x)
            return x
        if self.variant == "depthwise3x3_mlp1x1":
            x = self.net[0](x)
            x = self.net[1](x)
            x = self.net[2](x)
            x = F.dropout2d(x, p=self.dropout, training=self.training)
            x = self.net[3](x)
            x = self.net[4](x)
            x = self.net[5](x)
            x = F.dropout2d(x, p=self.dropout, training=self.training)
            x = self.net[6](x)
            return x
        return self.net(x)


class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights: torch.Tensor | None = None):
        super().__init__()
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float(), persistent=False)
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        valid = labels > 0
        if not torch.any(valid):
            return logits.sum() * 0.0
        targets = labels[valid] - 1
        flat_logits = logits.permute(0, 2, 3, 1)[valid]
        return F.cross_entropy(flat_logits, targets.long(), weight=self.class_weights)


class MaskedDiceCELoss(nn.Module):
    def __init__(
        self,
        n_classes: int,
        class_weights: torch.Tensor | None = None,
        lambda_dice: float = 1.0,
        lambda_ce: float = 1.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.n_classes = int(n_classes)
        self.lambda_dice = float(lambda_dice)
        self.lambda_ce = float(lambda_ce)
        self.eps = float(eps)
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float(), persistent=False)
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        valid = labels > 0
        if not torch.any(valid):
            return logits.sum() * 0.0

        labels_zero_based = torch.clamp(labels - 1, min=0)
        flat_logits = logits.permute(0, 2, 3, 1)[valid]
        flat_targets = labels_zero_based[valid]
        ce_loss = F.cross_entropy(
            flat_logits,
            flat_targets.long(),
            weight=self.class_weights,
        )

        probs = torch.softmax(logits, dim=1)
        targets = F.one_hot(labels_zero_based.long(), num_classes=self.n_classes).permute(0, 3, 1, 2)
        targets = targets.float()
        valid_mask = valid.unsqueeze(1)
        probs = probs * valid_mask
        targets = targets * valid_mask

        intersection = (probs * targets).sum(dim=(0, 2, 3))
        denom = probs.sum(dim=(0, 2, 3)) + targets.sum(dim=(0, 2, 3))
        valid_classes = targets.sum(dim=(0, 2, 3)) > 0
        if torch.any(valid_classes):
            dice = (2.0 * intersection[valid_classes] + self.eps) / (
                denom[valid_classes] + self.eps
            )
            dice_loss = 1.0 - dice.mean()
        else:
            dice_loss = logits.sum() * 0.0

        return self.lambda_dice * dice_loss + self.lambda_ce * ce_loss


class Pix2RepIndianPinesSegmentationModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        n_classes: int,
        freeze_backbone: bool,
        head_variant: str,
        head_hidden_dim: int,
        head_norm: str,
        head_dropout: float,
    ):
        super().__init__()
        self.backbone = backbone
        self.n_classes = int(n_classes)
        self.freeze_backbone = bool(freeze_backbone)
        self.segmentation_head = SegmentationHead2d(
            in_channels=int(self.backbone.n_feature_maps),
            out_channels=self.n_classes,
            variant=head_variant,
            hidden_channels=int(head_hidden_dim),
            norm=head_norm,
            dropout=float(head_dropout),
        )
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.freeze_backbone:
            self.backbone.eval()
            with torch.no_grad():
                features = self.backbone(images)
        else:
            features = self.backbone(images)
        return self.segmentation_head(features)


def load_pretrained_backbone(
    model: Pix2RepIndianPinesSegmentationModel,
    ckpt_path: str,
    map_location: str = "cpu",
) -> dict[str, list[str]]:
    try:
        checkpoint = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Failed to load the pretrained checkpoint. This Lightning .ckpt requires "
            "the environment to provide the modules used at save time, including "
            "`omegaconf`. Use the `rs-clean` conda env or export a backbone-only "
            "state dict first."
        ) from exc

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = dict(checkpoint["state_dict"])
    elif isinstance(checkpoint, dict) and "backbone_state_dict" in checkpoint:
        state_dict = {f"backbone.{k}": v for k, v in checkpoint["backbone_state_dict"].items()}
    elif isinstance(checkpoint, dict):
        state_dict = dict(checkpoint)
        if not any(key.startswith("backbone.") for key in state_dict):
            state_dict = {f"backbone.{k}": v for k, v in state_dict.items()}
    else:
        raise ValueError(f"Unsupported checkpoint format at {ckpt_path}")

    incompatible = model.load_state_dict(state_dict, strict=False)
    return {
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


def build_optimizer(
    model: Pix2RepIndianPinesSegmentationModel,
    freeze_backbone: bool,
    lr_head: float,
    lr_backbone: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    if freeze_backbone:
        return torch.optim.AdamW(
            model.segmentation_head.parameters(),
            lr=float(lr_head),
            weight_decay=float(weight_decay),
        )

    param_groups = [
        {
            "params": model.backbone.parameters(),
            "lr": float(lr_backbone),
            "weight_decay": float(weight_decay),
        },
        {
            "params": model.segmentation_head.parameters(),
            "lr": float(lr_head),
            "weight_decay": float(weight_decay),
        },
    ]
    return torch.optim.AdamW(param_groups)
