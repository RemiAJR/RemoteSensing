from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from loguru import logger
from PIL import Image

from pix2repv2.data.mumucd_landcover import (
    DW_PALETTE,
    DW_CLASS_NAMES,
    MUMUCDLandCoverSample,
    build_landcover_loader,
    build_landcover_samples,
    compute_class_weights,
    discover_city_names,
    filter_samples_by_metadata,
    load_center_crop,
    MUMUCDLandCoverPatchDataset,
)
from pix2repv2.landcover_segmentation_module import Pix2RepLandCoverSegmentationModule
from pix2repv2.utils.utils import Config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["linear_probing", "finetuning"], required=True)
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--pretrained_ckpt", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--train_patches_per_image", type=int, default=12)
    parser.add_argument("--val_patches_per_image", type=int, default=4)
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--eval_cities", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--accelerator", type=str, default="cpu")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--head_type",
        choices=[
            "linear",
            "mlp_1x1",
            "mlp_1x1_deep",
            "depthwise3x3_mlp1x1",
            "conv3x3_1x1",
        ],
        default="linear",
    )
    parser.add_argument("--head_hidden_dim", type=int, default=256)
    parser.add_argument("--head_dropout", type=float, default=0.0)
    parser.add_argument(
        "--head_norm",
        choices=["batch", "group", "instance", "none"],
        default="batch",
    )
    parser.add_argument(
        "--loss_name",
        choices=["ce", "dice_ce", "dice_ce_ls", "dice_focal"],
        default="ce",
    )
    parser.add_argument("--ce_label_smoothing", type=float, default=0.0)
    parser.add_argument("--early_stopping_patience", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--feature_noise_std", type=float, default=0.0)
    parser.add_argument("--quality_filter_ckpt", type=str, default="")
    parser.add_argument("--quality_filter_keep_ratio", type=float, default=1.0)
    parser.add_argument("--quality_filter_crop_size", type=int, default=512)
    parser.add_argument(
        "--quality_filter_head_type",
        choices=[
            "linear",
            "mlp_1x1",
            "mlp_1x1_deep",
            "depthwise3x3_mlp1x1",
            "conv3x3_1x1",
        ],
        default="mlp_1x1",
    )
    parser.add_argument("--quality_filter_head_hidden_dim", type=int, default=256)
    parser.add_argument("--quality_filter_head_dropout", type=float, default=0.0)
    parser.add_argument(
        "--quality_filter_head_norm",
        choices=["batch", "group", "instance", "none"],
        default="batch",
    )
    parser.add_argument(
        "--quality_filter_loss_name",
        choices=["ce", "dice_ce", "dice_ce_ls", "dice_focal"],
        default="dice_ce",
    )
    return parser.parse_args()


def build_city_split(
    seed: int,
    val_ratio: float,
    n_eval_cities: int,
) -> tuple[list[str], list[str], list[str]]:
    all_cities = discover_city_names("/workspace/RemoteSensing/data/mumucd")
    rng = np.random.default_rng(seed)
    shuffled = list(all_cities)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_ratio)))
    val_cities = sorted(shuffled[:n_val])
    train_cities = sorted(shuffled[n_val:])
    eval_cities = val_cities[:n_eval_cities]
    return train_cities, val_cities, eval_cities


def build_cfg(args: argparse.Namespace):
    cfg = Config(
        overrides=[
            "finetuning.n_classes=9",
            f"finetuning.num_epochs={args.epochs}",
            f"data.batch_size_finetuning={args.batch_size}",
            "finetuning.linear_probing_lr_outconv=1e-3",
            "finetuning.finetuning_lr_backbone=1e-5",
            "finetuning.finetuning_lr_outconv=5e-4",
        ]
    ).cfg
    return cfg


def load_pretrained_backbone(
    model: Pix2RepLandCoverSegmentationModule,
    ckpt_path: str,
) -> dict[str, list[str]]:
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = dict(checkpoint["state_dict"])
    for key in [k for k in list(state_dict.keys()) if k.startswith("loss_fn.")]:
        state_dict.pop(key, None)
    incompatible = model.load_state_dict(state_dict, strict=False)
    return {
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


def colorize_labels(label_map: np.ndarray) -> np.ndarray:
    return DW_PALETTE[label_map]


def grayscale_preview(image_crop: np.ndarray) -> np.ndarray:
    gray = image_crop.mean(axis=2)
    gray = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)


def save_qualitative_predictions(
    model: Pix2RepLandCoverSegmentationModule,
    samples: list[MUMUCDLandCoverSample],
    out_dir: Path,
    crop_size: int,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    per_sample_metrics = []

    with torch.no_grad():
        for sample in samples:
            image_crop, label_crop = load_center_crop(sample, crop_size=crop_size)
            image_tensor = torch.from_numpy(
                np.transpose(image_crop, (2, 0, 1))[..., None].astype(np.float32)
            ).unsqueeze(0).to(model.device)
            logits = model(image_tensor[:, :, :, :, 0])
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.int64)

            preview = grayscale_preview(image_crop)
            gt_rgb = colorize_labels(label_crop)
            pred_rgb = colorize_labels(pred)
            panel = np.concatenate([preview, gt_rgb, pred_rgb], axis=1)
            panel_path = out_dir / f"{sample.city}_{sample.timepoint}_center_panel.png"
            Image.fromarray(panel).save(panel_path)

            pixel_acc = float((pred == label_crop).mean())
            n_classes = len(DW_PALETTE)
            ious = []
            for class_idx in range(n_classes):
                pred_mask = pred == class_idx
                gt_mask = label_crop == class_idx
                union = np.logical_or(pred_mask, gt_mask).sum()
                if union == 0:
                    continue
                inter = np.logical_and(pred_mask, gt_mask).sum()
                ious.append(inter / union)
            class_iou = {}
            for class_idx, class_name in enumerate(DW_CLASS_NAMES):
                pred_mask = pred == class_idx
                gt_mask = label_crop == class_idx
                union = np.logical_or(pred_mask, gt_mask).sum()
                if union == 0:
                    continue
                inter = np.logical_and(pred_mask, gt_mask).sum()
                class_iou[class_name] = float(inter / union)
            per_sample_metrics.append(
                {
                    "city": sample.city,
                    "timepoint": sample.timepoint,
                    "pixel_acc": pixel_acc,
                    "mIoU": float(np.mean(ious)) if ious else 0.0,
                    "class_iou": class_iou,
                    "panel": str(panel_path),
                }
            )

    with (out_dir / "qualitative_metrics.json").open("w") as f:
        json.dump(per_sample_metrics, f, indent=2)


def compute_detailed_metrics(
    model: Pix2RepLandCoverSegmentationModule,
    loader,
) -> dict:
    model.eval()
    confmat = torch.zeros((model.n_classes, model.n_classes), dtype=torch.long)

    with torch.no_grad():
        for batch in loader:
            images = batch[0][:, :, :, :, 0].to(model.device)
            masks = batch[1][:, 0, :, :, 0].long().to(model.device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            confmat += model._compute_confusion_matrix(preds, masks).cpu()

    confmat_float = confmat.float()
    true_positives = torch.diag(confmat_float)
    denom = confmat_float.sum(dim=1) + confmat_float.sum(dim=0) - true_positives
    valid = denom > 0
    iou = torch.zeros_like(true_positives)
    iou[valid] = true_positives[valid] / denom[valid]
    pixel_acc = (
        float(true_positives.sum() / confmat_float.sum()) if confmat_float.sum() > 0 else 0.0
    )
    return {
        "pixel_acc": pixel_acc,
        "mIoU": float(iou[valid].mean()) if valid.any() else 0.0,
        "per_class_iou": {
            class_name: float(iou[idx]) for idx, class_name in enumerate(DW_CLASS_NAMES)
        },
        "confusion_matrix": confmat.tolist(),
    }


def score_samples_against_labels(
    model: Pix2RepLandCoverSegmentationModule,
    samples: list[MUMUCDLandCoverSample],
    crop_size: int,
) -> list[dict]:
    scored = []
    model.eval()
    with torch.no_grad():
        for sample in samples:
            image_crop, label_crop = load_center_crop(sample, crop_size=crop_size)
            image_tensor = torch.from_numpy(
                np.transpose(image_crop, (2, 0, 1))[..., None].astype(np.float32)
            ).unsqueeze(0).to(model.device)
            logits = model(image_tensor[:, :, :, :, 0])
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.int64)
            pixel_acc = float((pred == label_crop).mean())
            ious = []
            for class_idx in range(len(DW_PALETTE)):
                pred_mask = pred == class_idx
                gt_mask = label_crop == class_idx
                union = np.logical_or(pred_mask, gt_mask).sum()
                if union == 0:
                    continue
                inter = np.logical_and(pred_mask, gt_mask).sum()
                ious.append(inter / union)
            scored.append(
                {
                    "city": sample.city,
                    "timepoint": sample.timepoint,
                    "pixel_acc": pixel_acc,
                    "mIoU": float(np.mean(ious)) if ious else 0.0,
                }
            )
    return scored


def main():
    args = parse_args()
    seed_everything(args.seed, workers=True)
    cfg = build_cfg(args)
    freeze_backbone = args.strategy == "linear_probing"

    train_cities, val_cities, eval_cities = build_city_split(
        seed=args.seed,
        val_ratio=args.val_ratio,
        n_eval_cities=args.eval_cities,
    )
    # before/after are two different dates for the same city; using both doubles
    # the supervised signal without leaking labels across the train/val city split.
    train_samples = build_landcover_samples(train_cities, ("before", "after"))
    val_samples = build_landcover_samples(val_cities, ("before", "after"))
    eval_samples = build_landcover_samples(eval_cities, ("after",))
    filtered_out_samples = []

    if args.quality_filter_ckpt and args.quality_filter_keep_ratio < 1.0:
        filter_model = Pix2RepLandCoverSegmentationModule(
            cfg=cfg,
            freeze_backbone=True,
            class_weights=torch.ones(9),
            head_variant=args.quality_filter_head_type,
            head_hidden_dim=args.quality_filter_head_hidden_dim,
            head_norm=args.quality_filter_head_norm,
            head_dropout=args.quality_filter_head_dropout,
            loss_name=args.quality_filter_loss_name,
            ce_label_smoothing=0.0,
            weight_decay=args.weight_decay,
            feature_noise_std=0.0,
        )
        load_pretrained_backbone(filter_model, args.quality_filter_ckpt)
        filter_model = filter_model.to(args.accelerator if args.accelerator != "gpu" else "cuda")
        scored_samples = score_samples_against_labels(
            model=filter_model,
            samples=train_samples,
            crop_size=args.quality_filter_crop_size,
        )
        keep_count = max(1, int(round(len(scored_samples) * args.quality_filter_keep_ratio)))
        scored_samples = sorted(scored_samples, key=lambda row: row["mIoU"], reverse=True)
        kept_scores = scored_samples[:keep_count]
        filtered_out_samples = scored_samples[keep_count:]
        keep_keys = {(row["city"], row["timepoint"]) for row in kept_scores}
        train_samples = filter_samples_by_metadata(train_samples, keep_keys)

    class_weights = compute_class_weights(train_samples, n_classes=9)

    train_dataset = MUMUCDLandCoverPatchDataset(
        train_samples,
        patch_size=args.patch_size,
        patches_per_image=args.train_patches_per_image,
        is_training=True,
        seed=args.seed,
    )
    val_dataset = MUMUCDLandCoverPatchDataset(
        val_samples,
        patch_size=args.patch_size,
        patches_per_image=args.val_patches_per_image,
        is_training=False,
        seed=args.seed,
    )

    train_loader = build_landcover_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = build_landcover_loader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = Pix2RepLandCoverSegmentationModule(
        cfg=cfg,
        freeze_backbone=freeze_backbone,
        class_weights=class_weights,
        head_variant=args.head_type,
        head_hidden_dim=args.head_hidden_dim,
        head_norm=args.head_norm,
        head_dropout=args.head_dropout,
        loss_name=args.loss_name,
        ce_label_smoothing=args.ce_label_smoothing,
        weight_decay=args.weight_decay,
        feature_noise_std=args.feature_noise_std,
    )
    incompatible = load_pretrained_backbone(model, args.pretrained_ckpt)
    logger.info(
        f"Loaded pretrained backbone from {args.pretrained_ckpt}. "
        f"Missing keys: {len(incompatible['missing_keys'])}, "
        f"unexpected keys: {len(incompatible['unexpected_keys'])}"
    )

    run_dir = Path("output/remote_sensing_segmentation") / args.exp_name
    ckpt_dir = run_dir / "checkpoints"
    qualitative_dir = run_dir / "qualitative"
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "city_split.json").open("w") as f:
        json.dump(
            {
                "train_cities": train_cities,
                "val_cities": val_cities,
                "eval_cities_after": eval_cities,
                "train_samples": len(train_samples),
                "val_samples": len(val_samples),
                "strategy": args.strategy,
                "pretrained_ckpt": args.pretrained_ckpt,
                "head_type": args.head_type,
                "head_hidden_dim": args.head_hidden_dim,
                "head_norm": args.head_norm,
                "head_dropout": args.head_dropout,
                "loss_name": args.loss_name,
                "ce_label_smoothing": args.ce_label_smoothing,
                "weight_decay": args.weight_decay,
                "feature_noise_std": args.feature_noise_std,
                "accelerator": args.accelerator,
                "devices": args.devices,
                "class_weights": class_weights.tolist(),
                "quality_filter_ckpt": args.quality_filter_ckpt,
                "quality_filter_keep_ratio": args.quality_filter_keep_ratio,
                "quality_filter_crop_size": args.quality_filter_crop_size,
                "quality_filter_head_type": args.quality_filter_head_type,
                "quality_filter_head_hidden_dim": args.quality_filter_head_hidden_dim,
                "quality_filter_head_dropout": args.quality_filter_head_dropout,
                "quality_filter_head_norm": args.quality_filter_head_norm,
                "quality_filter_loss_name": args.quality_filter_loss_name,
                "filtered_out_samples": filtered_out_samples,
            },
            f,
            indent=2,
        )

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="best",
        monitor="val_mIoU",
        mode="max",
        save_top_k=1,
        save_last=True,
    )
    callbacks = [checkpoint_callback]
    if args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor="val_mIoU",
                mode="max",
                patience=args.early_stopping_patience,
            )
        )

    csv_logger = CSVLogger(save_dir=str(run_dir), name="csv_logs")
    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        logger=csv_logger,
        max_epochs=args.epochs,
        callbacks=callbacks,
        log_every_n_steps=5,
        enable_progress_bar=False,
    )
    trainer.fit(model, train_loader, val_loader)
    if checkpoint_callback.best_model_path:
        best_checkpoint = torch.load(
            checkpoint_callback.best_model_path,
            map_location=model.device,
            weights_only=False,
        )
        model.load_state_dict(best_checkpoint["state_dict"], strict=False)

    detailed_metrics = compute_detailed_metrics(model=model, loader=val_loader)
    with (run_dir / "val_metrics_detailed.json").open("w") as f:
        json.dump(detailed_metrics, f, indent=2)
    save_qualitative_predictions(
        model=model,
        samples=eval_samples,
        out_dir=qualitative_dir,
        crop_size=args.crop_size,
    )

    summary = {
        "strategy": args.strategy,
        "best_model_path": checkpoint_callback.best_model_path,
        "best_val_mIoU": float(checkpoint_callback.best_model_score)
        if checkpoint_callback.best_model_score is not None
        else None,
        "head_type": args.head_type,
        "head_hidden_dim": args.head_hidden_dim,
        "head_norm": args.head_norm,
        "head_dropout": args.head_dropout,
        "loss_name": args.loss_name,
        "ce_label_smoothing": args.ce_label_smoothing,
        "early_stopping_patience": args.early_stopping_patience,
        "weight_decay": args.weight_decay,
        "feature_noise_std": args.feature_noise_std,
        "detailed_val_metrics_path": str(run_dir / "val_metrics_detailed.json"),
        "run_dir": str(run_dir),
    }
    with (run_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
