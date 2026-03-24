from __future__ import annotations

import lightning as L
import monai.losses
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from pix2repv2.data.mumucd_landcover import DW_CLASS_NAMES
from pix2repv2.models.layers import OutConv, get_norm_layer
from pix2repv2.models.registry import build_backbone
from pix2repv2.utils import utils


class LandCoverSegmentationHead(nn.Module):
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


class Pix2RepLandCoverSegmentationModule(L.LightningModule):
    def __init__(
        self,
        cfg: dict,
        freeze_backbone: bool,
        class_weights: torch.Tensor | None = None,
        head_variant: str = "linear",
        head_hidden_dim: int = 256,
        head_norm: str = "batch",
        head_dropout: float = 0.0,
        loss_name: str = "ce",
        ce_label_smoothing: float = 0.0,
        weight_decay: float = 1e-2,
        feature_noise_std: float = 0.0,
    ):
        super().__init__()
        flattened_cfg = utils.flatten_finetuning_cfg(cfg)
        self.save_hyperparameters(flattened_cfg)
        self.cfg = cfg
        self.freeze_backbone = freeze_backbone
        self.n_classes = int(self.cfg.finetuning.n_classes)
        self.loss_name = loss_name
        self.ce_label_smoothing = float(ce_label_smoothing)
        self.weight_decay = float(weight_decay)
        self.feature_noise_std = float(feature_noise_std)

        self.backbone = build_backbone(self.cfg)
        self.segmentation_head = LandCoverSegmentationHead(
            in_channels=self.backbone.n_feature_maps,
            out_channels=self.n_classes,
            variant=head_variant,
            hidden_channels=head_hidden_dim,
            norm=head_norm,
            dropout=float(head_dropout),
        )
        if class_weights is not None:
            self.register_buffer(
                "class_weights",
                class_weights.float(),
                persistent=True,
            )
            class_weights_tensor = self.class_weights
        else:
            class_weights_tensor = None

        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights_tensor,
            label_smoothing=self.ce_label_smoothing,
        )
        self.dice_loss = monai.losses.DiceLoss(
            include_background=True,
            to_onehot_y=True,
            reduction="mean",
            softmax=True,
        )
        if self.loss_name == "ce":
            self.loss_fn = self.ce_loss
        elif self.loss_name == "dice_ce":
            self.loss_fn = monai.losses.DiceCELoss(
                include_background=True,
                to_onehot_y=True,
                reduction="mean",
                softmax=True,
                weight=class_weights_tensor,
                lambda_dice=float(self.cfg.finetuning.lambda_dice),
                lambda_ce=float(self.cfg.finetuning.lambda_ce),
            )
        elif self.loss_name == "dice_ce_ls":
            self.loss_fn = None
        elif self.loss_name == "dice_focal":
            self.loss_fn = monai.losses.DiceFocalLoss(
                include_background=True,
                to_onehot_y=True,
                reduction="mean",
                softmax=True,
                weight=class_weights_tensor,
                lambda_dice=float(self.cfg.finetuning.lambda_dice),
                lambda_focal=1.0,
                gamma=2.0,
            )
        else:
            raise ValueError(f"Unknown loss_name: {self.loss_name}")

        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.register_buffer(
            "val_confmat",
            torch.zeros((self.n_classes, self.n_classes), dtype=torch.long),
            persistent=False,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.freeze_backbone:
            self.backbone.eval()
            with torch.no_grad():
                features = self.backbone(images)
        else:
            features = self.backbone(images)
        if self.training and self.feature_noise_std > 0:
            features = features + torch.randn_like(features) * self.feature_noise_std
        return self.segmentation_head(features)

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        images = rearrange(batch[0], "B C H W 1 -> B C H W")
        masks_with_channel = rearrange(batch[1], "B 1 H W 1 -> B 1 H W").long()
        masks = masks_with_channel.squeeze(1)
        logits = self(images)
        if self.loss_name == "dice_ce_ls":
            loss = (
                float(self.cfg.finetuning.lambda_dice) * self.dice_loss(logits, masks_with_channel)
                + float(self.cfg.finetuning.lambda_ce) * self.ce_loss(logits, masks)
            )
        elif self.loss_name in {"dice_ce", "dice_focal"}:
            loss = self.loss_fn(logits, masks_with_channel)
        else:
            loss = self.loss_fn(logits, masks)
        preds = torch.argmax(logits, dim=1)
        pixel_acc = (preds == masks).float().mean()

        self.log(
            f"{stage}_loss",
            loss,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True,
            batch_size=images.shape[0],
        )
        self.log(
            f"{stage}_pixel_acc",
            pixel_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=(stage != "train"),
            batch_size=images.shape[0],
        )

        if stage == "val":
            confmat = self._compute_confusion_matrix(preds, masks)
            self.val_confmat += confmat

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="val")

    def on_validation_epoch_start(self):
        self.val_confmat.zero_()

    def on_validation_epoch_end(self):
        confmat = self.val_confmat.float()
        true_positives = torch.diag(confmat)
        denom = confmat.sum(dim=1) + confmat.sum(dim=0) - true_positives
        valid = denom > 0
        iou = torch.zeros_like(true_positives)
        iou[valid] = true_positives[valid] / denom[valid]
        mean_iou = iou[valid].mean() if valid.any() else torch.tensor(0.0, device=self.device)
        self.log("val_mIoU", mean_iou, prog_bar=True)
        for class_name, class_iou in zip(DW_CLASS_NAMES, iou):
            self.log(f"val_iou_{class_name}", class_iou, prog_bar=False)

    def configure_optimizers(self):
        if self.freeze_backbone:
            return torch.optim.AdamW(
                self.segmentation_head.parameters(),
                lr=float(self.cfg.finetuning.linear_probing_lr_outconv),
                weight_decay=self.weight_decay,
            )

        param_groups = [
            {
                "params": self.backbone.parameters(),
                "lr": float(self.cfg.finetuning.finetuning_lr_backbone),
                "weight_decay": self.weight_decay,
            },
            {
                "params": self.segmentation_head.parameters(),
                "lr": float(self.cfg.finetuning.finetuning_lr_outconv),
                "weight_decay": self.weight_decay,
            },
        ]
        return torch.optim.AdamW(param_groups)

    def _compute_confusion_matrix(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        valid = (targets >= 0) & (targets < self.n_classes)
        flat_targets = targets[valid].view(-1)
        flat_preds = preds[valid].view(-1)
        encoded = flat_targets * self.n_classes + flat_preds
        counts = torch.bincount(encoded, minlength=self.n_classes * self.n_classes)
        return counts.reshape(self.n_classes, self.n_classes)
