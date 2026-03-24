from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from loguru import logger
from PIL import Image

from pix2repv2.data.mumucd_landcover import DW_CLASS_NAMES, DW_PALETTE
from pix2repv2.landcover_segmentation_module import Pix2RepLandCoverSegmentationModule
from pix2repv2.utils.utils import Config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--segmentation_ckpt",
        default="/workspace/RemoteSensing/Pix2Rep-v2_Remi/output/remote_sensing_segmentation/mumucd_landcover_lp_gpu_mlp1x1_dicece_e100/checkpoints/best.ckpt",
    )
    parser.add_argument(
        "--data_root",
        default="/workspace/RemoteSensing/data/mumucd",
    )
    parser.add_argument("--city", default="paris")
    parser.add_argument("--timepoint", choices=["before", "after"], default="after")
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--head_type", choices=["linear", "mlp_1x1", "mlp_1x1_deep", "depthwise3x3_mlp1x1", "conv3x3_1x1"], default="mlp_1x1")
    parser.add_argument("--head_hidden_dim", type=int, default=256)
    parser.add_argument("--head_norm", choices=["batch", "group", "instance", "none"], default="batch")
    parser.add_argument("--head_dropout", type=float, default=0.0)
    parser.add_argument("--loss_name", choices=["ce", "dice_ce", "dice_ce_ls", "dice_focal"], default="dice_ce")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--exp_name",
        default="mumucd_full_scene_paris_after_mlp1x1_dicece_e100",
    )
    return parser.parse_args()


def build_cfg():
    return Config(
        overrides=[
            "finetuning.n_classes=9",
            "finetuning.num_epochs=100",
            "data.batch_size_finetuning=16",
            "finetuning.linear_probing_lr_outconv=1e-3",
            "finetuning.finetuning_lr_backbone=1e-5",
            "finetuning.finetuning_lr_outconv=5e-4",
        ]
    ).cfg


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _rescale_intensity(image: np.ndarray) -> np.ndarray:
    lo = np.percentile(image, 1)
    hi = np.percentile(image, 99)
    if hi - lo < 1e-6:
        return np.zeros_like(image, dtype=np.float32)
    scaled = (image - lo) / (hi - lo)
    return np.clip(scaled, 0.0, 1.0).astype(np.float32, copy=False)


def build_positions(length: int, tile_size: int, overlap: int) -> list[int]:
    if tile_size > length:
        raise ValueError(f"tile_size={tile_size} exceeds image length={length}")
    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError(f"overlap={overlap} must be smaller than tile_size={tile_size}")
    if tile_size == length:
        return [0]

    positions = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def colorize(label_map: np.ndarray) -> np.ndarray:
    return DW_PALETTE[np.clip(label_map, 0, len(DW_PALETTE) - 1)]


def compute_metrics(prediction: np.ndarray, target: np.ndarray, n_classes: int) -> dict[str, object]:
    encoded = target.reshape(-1) * n_classes + prediction.reshape(-1)
    confusion_matrix = np.bincount(encoded, minlength=n_classes * n_classes).reshape(
        n_classes,
        n_classes,
    )
    conf = confusion_matrix.astype(np.float64, copy=False)
    true_positives = np.diag(conf)
    union = conf.sum(axis=1) + conf.sum(axis=0) - true_positives
    valid = union > 0
    per_class_iou = np.zeros(n_classes, dtype=np.float64)
    per_class_iou[valid] = true_positives[valid] / union[valid]
    pixel_acc = float(true_positives.sum() / max(conf.sum(), 1.0))
    return {
        "pixel_acc": pixel_acc,
        "mIoU": float(per_class_iou[valid].mean()) if np.any(valid) else 0.0,
        "per_class_iou": {
            class_name: float(per_class_iou[idx]) for idx, class_name in enumerate(DW_CLASS_NAMES)
        },
        "confusion_matrix": confusion_matrix.astype(np.int64).tolist(),
    }


def load_model(args: argparse.Namespace, device: torch.device) -> tuple[Pix2RepLandCoverSegmentationModule, dict[str, list[str]]]:
    model = Pix2RepLandCoverSegmentationModule(
        cfg=build_cfg(),
        freeze_backbone=True,
        class_weights=torch.ones(9),
        head_variant=args.head_type,
        head_hidden_dim=args.head_hidden_dim,
        head_norm=args.head_norm,
        head_dropout=args.head_dropout,
        loss_name=args.loss_name,
        ce_label_smoothing=0.0,
        weight_decay=1e-2,
        feature_noise_std=0.0,
    )
    checkpoint = torch.load(args.segmentation_ckpt, map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(checkpoint["state_dict"], strict=False)
    model = model.to(device)
    model.eval()
    return model, {
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


def predict_full_scene(
    model: Pix2RepLandCoverSegmentationModule,
    image_path: Path,
    tile_size: int,
    overlap: int,
    device: torch.device,
) -> np.ndarray:
    image_ds = xr.open_dataset(image_path)
    height = int(image_ds.sizes["nj"])
    width = int(image_ds.sizes["ni"])
    row_positions = build_positions(height, tile_size, overlap)
    col_positions = build_positions(width, tile_size, overlap)

    logits_sum = np.zeros((model.n_classes, height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.float32)

    total_tiles = len(row_positions) * len(col_positions)
    tile_idx = 0
    with torch.inference_mode():
        for row in row_positions:
            for col in col_positions:
                tile_idx += 1
                tile = image_ds["sr"].isel(
                    nj=slice(row, row + tile_size),
                    ni=slice(col, col + tile_size),
                ).values
                tile = _rescale_intensity(np.asarray(tile, dtype=np.float32))
                image_tensor = torch.from_numpy(
                    np.transpose(tile, (2, 0, 1)).astype(np.float32, copy=False)
                ).unsqueeze(0).to(device)
                tile_logits = model(image_tensor).squeeze(0).cpu().numpy().astype(np.float32)
                logits_sum[:, row : row + tile_size, col : col + tile_size] += tile_logits
                counts[row : row + tile_size, col : col + tile_size] += 1.0
                if tile_idx == 1 or tile_idx == total_tiles or tile_idx % 4 == 0:
                    logger.info(
                        "Predicted tile {}/{} at row={}, col={}",
                        tile_idx,
                        total_tiles,
                        row,
                        col,
                    )

    averaged_logits = logits_sum / np.maximum(counts[None, :, :], 1.0)
    return np.argmax(averaged_logits, axis=0).astype(np.uint8)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    city_dir = Path(args.data_root) / args.city
    image_path = city_dir / f"{args.city}-{args.timepoint}-prs.nc"
    label_path = city_dir / f"{args.city}-{args.timepoint}-dw.nc"
    if not image_path.exists():
        raise FileNotFoundError(f"Missing image file: {image_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"Missing label file: {label_path}")

    model, incompatible = load_model(args, device=device)
    logger.info(
        "Loaded segmentation checkpoint {} on {}. Missing keys: {}, unexpected keys: {}",
        args.segmentation_ckpt,
        device,
        len(incompatible["missing_keys"]),
        len(incompatible["unexpected_keys"]),
    )

    prediction = predict_full_scene(
        model=model,
        image_path=image_path,
        tile_size=args.tile_size,
        overlap=args.overlap,
        device=device,
    )
    label_ds = xr.open_dataset(label_path)
    target = np.clip(np.asarray(label_ds["lcc"].values, dtype=np.int64), 0, model.n_classes - 1)
    metrics = compute_metrics(
        prediction=prediction.astype(np.int64),
        target=target,
        n_classes=model.n_classes,
    )

    run_dir = (
        Path("/workspace/RemoteSensing/Pix2Rep-v2_Remi/output/remote_sensing_segmentation")
        / args.exp_name
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    np.save(run_dir / "prediction.npy", prediction.astype(np.uint8))
    Image.fromarray(colorize(prediction)).save(run_dir / f"{args.city}_{args.timepoint}_prediction.png")
    Image.fromarray(colorize(target.astype(np.uint8))).save(
        run_dir / f"{args.city}_{args.timepoint}_ground_truth.png"
    )

    summary = {
        "city": args.city,
        "timepoint": args.timepoint,
        "image_path": str(image_path),
        "label_path": str(label_path),
        "segmentation_ckpt": args.segmentation_ckpt,
        "tile_size": args.tile_size,
        "overlap": args.overlap,
        "device": str(device),
        "head_type": args.head_type,
        "loss_name": args.loss_name,
        "pixel_acc": metrics["pixel_acc"],
        "mIoU": metrics["mIoU"],
        "missing_keys": incompatible["missing_keys"],
        "unexpected_keys": incompatible["unexpected_keys"],
    }
    with (run_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    with (run_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(
        "Full-scene inference complete for {}-{} | pixel_acc={:.4f} | mIoU={:.4f}",
        args.city,
        args.timepoint,
        metrics["pixel_acc"],
        metrics["mIoU"],
    )


if __name__ == "__main__":
    main()
