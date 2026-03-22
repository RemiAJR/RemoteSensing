import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from pix2repv2.models import mlp
from pix2repv2.models.registry import build_backbone
from pix2repv2.utils import losses, utils


class Pix2RepPretrainingModule(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        flattened_cfg = utils.flatten_pretraining_cfg(cfg)
        self.save_hyperparameters(flattened_cfg)
        self.cfg = cfg
        self.uniformity_sample_size = 1024
        self.dead_dim_std_threshold = 1e-3

        self.backbone = build_backbone(self.cfg)

        channels_in = self.backbone.n_feature_maps
        channels_out = self.cfg.pretraining.n_feature_maps_mlp
        depth = self.cfg.pretraining.projection_head_depth
        assert depth >= 0 and isinstance(depth, int), (
            "projection_head_depth must be an int >= 0"
        )

        if depth == 0:
            self.projection_head = nn.Identity()
        else:
            inner_dims = list()
            if hasattr(self.cfg.pretraining, "inner_dim_1"):
                inner_dims.append(self.cfg.pretraining.inner_dim_1)
            if hasattr(self.cfg.pretraining, "inner_dim_2"):
                inner_dims.append(self.cfg.pretraining.inner_dim_2)
            if hasattr(self.cfg.pretraining, "inner_dim_3"):
                inner_dims.append(self.cfg.pretraining.inner_dim_3)
            if hasattr(self.cfg.pretraining, "inner_dim_4"):
                inner_dims.append(self.cfg.pretraining.inner_dim_4)
            inner_dims = inner_dims[:depth]
            self.projection_head = mlp.ProjectionHead(
                channels_in=channels_in,
                channels_out=channels_out,
                inner_dims=inner_dims,
                norm_type=self.cfg.finetuning.in_context_proj_norm,
            )

    def forward(self, view):
        return self.projection_head(self.backbone(view))

    def _compute_monitoring_metrics(
        self,
        dense_features_1: torch.Tensor,
        dense_features_2: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        features_1 = rearrange(dense_features_1, "b c h w -> (b h w) c")
        features_2 = rearrange(dense_features_2, "b c h w -> (b h w) c")

        norm_features_1 = F.normalize(features_1, dim=1)
        norm_features_2 = F.normalize(features_2, dim=1)

        alignment = (norm_features_1 - norm_features_2).pow(2).sum(dim=1).mean()

        combined = torch.cat([norm_features_1, norm_features_2], dim=0)
        if combined.shape[0] > self.uniformity_sample_size:
            perm = torch.randperm(combined.shape[0], device=combined.device)[
                : self.uniformity_sample_size
            ]
            combined = combined[perm]
        pairwise_sq_dists = torch.pdist(combined, p=2).pow(2)
        uniformity = torch.log(torch.exp(-2.0 * pairwise_sq_dists).mean() + 1e-12)

        feature_stds = torch.cat([features_1, features_2], dim=0).std(dim=0, unbiased=False)
        dead_dims_pct = (feature_stds < self.dead_dim_std_threshold).float().mean() * 100.0

        return {
            "alignment": alignment,
            "uniformity": uniformity,
            "dead_dims_pct": dead_dims_pct,
        }

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        view1 = rearrange(batch[0], "B C W H 1 -> B C W H")
        view2 = rearrange(batch[1], "B C W H 1 -> B C W H")
        grid = batch[2].squeeze(1)

        view2 = F.grid_sample(
            view2,
            grid,
            mode="bicubic",
            align_corners=True,
        )

        dense_features_1 = self(view1)
        dense_features_2 = self(view2)

        dense_features_1 = F.grid_sample(
            dense_features_1,
            grid,
            mode="bicubic",
            align_corners=True,
        )

        on_diag, off_diag_term = losses.barlow_terms(
            dense_features_1,
            dense_features_2,
            self.device,
        )
        loss = on_diag + self.cfg.pretraining.lambda_barlow * off_diag_term
        monitoring = self._compute_monitoring_metrics(dense_features_1, dense_features_2)

        batch_size = self.cfg.data.batch_size_pretraining
        sync_dist = stage == "val"
        self.log(
            f"{stage}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=sync_dist,
            batch_size=batch_size,
        )
        self.log(
            f"{stage}_barlow_on_diag",
            on_diag,
            on_step=False,
            on_epoch=True,
            sync_dist=sync_dist,
            batch_size=batch_size,
        )
        self.log(
            f"{stage}_barlow_off_diag",
            off_diag_term,
            on_step=False,
            on_epoch=True,
            sync_dist=sync_dist,
            batch_size=batch_size,
        )
        self.log(
            f"{stage}_alignment",
            monitoring["alignment"],
            on_step=False,
            on_epoch=True,
            sync_dist=sync_dist,
            batch_size=batch_size,
        )
        self.log(
            f"{stage}_uniformity",
            monitoring["uniformity"],
            on_step=False,
            on_epoch=True,
            sync_dist=sync_dist,
            batch_size=batch_size,
        )
        self.log(
            f"{stage}_dead_dims_pct",
            monitoring["dead_dims_pct"],
            on_step=False,
            on_epoch=True,
            sync_dist=sync_dist,
            batch_size=batch_size,
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [
                {"params": self.backbone.parameters()},
                {"params": self.projection_head.parameters()},
            ],
            lr=self.cfg.pretraining.lr_backbone,
        )

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(0.05 * total_steps)

        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-6,
            end_factor=1.0,
            total_iters=warmup_steps,
        )

        cosine_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=1e-8,
        )

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine_schedule],
            milestones=[warmup_steps],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
