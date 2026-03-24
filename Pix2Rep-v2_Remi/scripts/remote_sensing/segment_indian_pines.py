from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from PIL import Image

from pix2repv2.data.indian_pines import (
    INDIAN_PINES_CLASS_NAMES,
    IndianPinesPatchDataset,
    build_loader,
    build_sliding_windows,
    build_split_masks,
    colorize_label_map,
    combine_patch_predictions,
    compute_class_weights,
    compute_confusion_matrix,
    compute_metrics_from_confusion_matrix,
    compute_pca,
    load_indian_pines,
    pad_channels,
    split_indices,
)
from pix2repv2.indian_pines_segmentation import (
    MaskedCrossEntropyLoss,
    MaskedDiceCELoss,
    Pix2RepIndianPinesSegmentationModel,
    build_optimizer,
    load_pretrained_backbone,
)
from pix2repv2.models.unet import UNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", required=True)
    parser.add_argument(
        "--pretrained_ckpt",
        default="/workspace/RemoteSensing/Pix2Rep-v2_Remi/output/remote_sensing_segmentation/backbone_checkpoints/pretrain_mumucd_unet_4gpu_fix3_best_epoch127.ckpt",
    )
    parser.add_argument(
        "--strategy",
        choices=["linear_probing", "finetuning"],
        default="linear_probing",
    )
    parser.add_argument(
        "--data_path",
        default="/workspace/RemoteSensing/HyperSIGMA/ImageClassification/data/Indian_pines_corrected.mat",
    )
    parser.add_argument(
        "--labels_path",
        default="/workspace/RemoteSensing/HyperSIGMA/ImageClassification/data/Indian_pines_gt.mat",
    )
    parser.add_argument("--data_key", default="data")
    parser.add_argument("--labels_key", default="groundT")
    parser.add_argument("--window_size", type=int, default=145)
    parser.add_argument("--overlap_size", type=int, default=0)
    parser.add_argument("--split_type", choices=["number", "ratio"], default="number")
    parser.add_argument("--train_num", type=int, default=10)
    parser.add_argument("--val_num", type=int, default=5)
    parser.add_argument("--train_ratio", type=float, default=0.05)
    parser.add_argument("--val_ratio", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3704)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--pca_components", type=int, default=0)
    parser.add_argument("--pca_whiten", action="store_true")
    parser.add_argument("--pad_to_channels", type=int, default=230)
    parser.add_argument(
        "--head_type",
        choices=[
            "linear",
            "mlp_1x1",
            "mlp_1x1_deep",
            "depthwise3x3_mlp1x1",
            "conv3x3_1x1",
        ],
        default="mlp_1x1",
    )
    parser.add_argument("--head_hidden_dim", type=int, default=256)
    parser.add_argument(
        "--head_norm",
        choices=["batch", "group", "instance", "none"],
        default="batch",
    )
    parser.add_argument("--head_dropout", type=float, default=0.0)
    parser.add_argument("--loss_name", choices=["ce", "dice_ce"], default="dice_ce")
    parser.add_argument("--lambda_dice", type=float, default=1.0)
    parser.add_argument("--lambda_ce", type=float, default=1.0)
    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--early_stopping_patience", type=int, default=100)
    parser.add_argument("--log_every", type=int, default=10)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def build_loss(
    loss_name: str,
    n_classes: int,
    class_weights: torch.Tensor | None,
    lambda_dice: float,
    lambda_ce: float,
) -> torch.nn.Module:
    if loss_name == "ce":
        return MaskedCrossEntropyLoss(class_weights=class_weights)
    if loss_name == "dice_ce":
        return MaskedDiceCELoss(
            n_classes=n_classes,
            class_weights=class_weights,
            lambda_dice=lambda_dice,
            lambda_ce=lambda_ce,
        )
    raise ValueError(f"Unsupported loss_name: {loss_name}")


def run_epoch(
    model: Pix2RepIndianPinesSegmentationModel,
    loader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_labeled = 0

    for images, labels in loader:
        images = images[:, :, :, :, 0].to(device)
        labels = labels[:, 0, :, :, 0].long().to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = loss_fn(logits, labels)

        if training:
            loss.backward()
            optimizer.step()

        preds = torch.argmax(logits, dim=1)
        valid = labels > 0
        total_loss += float(loss.item())
        if torch.any(valid):
            total_correct += int((preds[valid] == (labels[valid] - 1)).sum().item())
            total_labeled += int(valid.sum().item())

    mean_loss = total_loss / max(len(loader), 1)
    labeled_acc = total_correct / max(total_labeled, 1)
    return mean_loss, labeled_acc


def predict_scene(
    model: Pix2RepIndianPinesSegmentationModel,
    cube: np.ndarray,
    window_size: int,
    overlap_size: int,
    device: torch.device,
) -> np.ndarray:
    image_patches, _ = build_sliding_windows(
        cube=cube,
        label_mask=np.zeros(cube.shape[:2], dtype=np.int64),
        window_size=window_size,
        overlap_size=overlap_size,
    )
    dataset = IndianPinesPatchDataset(
        image_patches=image_patches,
        label_patches=np.zeros(image_patches.shape[:3], dtype=np.int64),
    )
    loader = build_loader(dataset, batch_size=1, shuffle=False, num_workers=0)

    patch_predictions: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            images = images[:, :, :, :, 0].to(device)
            logits = model(images)
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.int64) + 1
            patch_predictions.append(pred)

    return combine_patch_predictions(
        patch_predictions=np.stack(patch_predictions, axis=0),
        image_shape=cube.shape[:2],
        window_size=window_size,
        overlap_size=overlap_size,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    cube, labels = load_indian_pines(
        data_path=args.data_path,
        labels_path=args.labels_path,
        data_key=args.data_key,
        labels_key=args.labels_key,
        normalize=True,
    )
    original_channels = int(cube.shape[-1])
    if args.pca_components > 0:
        cube = compute_pca(cube, num_components=args.pca_components, whiten=args.pca_whiten)
    cube = pad_channels(cube, target_channels=args.pad_to_channels)
    n_classes = int(labels.max())

    train_index, val_index, test_index = split_indices(
        labels_flat=labels.reshape(-1),
        class_num=n_classes,
        train_num=args.train_num,
        val_num=args.val_num,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        split_type=args.split_type,
        seed=args.seed,
    )
    split = build_split_masks(
        labels=labels,
        train_index=train_index,
        val_index=val_index,
        test_index=test_index,
    )

    train_images, train_labels = build_sliding_windows(
        cube=cube,
        label_mask=split.train_mask,
        window_size=args.window_size,
        overlap_size=args.overlap_size,
    )
    val_images, val_labels = build_sliding_windows(
        cube=cube,
        label_mask=split.val_mask,
        window_size=args.window_size,
        overlap_size=args.overlap_size,
    )

    train_loader = build_loader(
        IndianPinesPatchDataset(train_images, train_labels),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = build_loader(
        IndianPinesPatchDataset(val_images, val_labels),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    freeze_backbone = args.strategy == "linear_probing"
    class_weights = compute_class_weights(split.train_mask, n_classes=n_classes)
    backbone = UNet(
        in_channels=int(cube.shape[-1]),
        out_channels_repr=1024,
        norm="batch",
        bilinear=False,
    )
    model = Pix2RepIndianPinesSegmentationModel(
        backbone=backbone,
        n_classes=n_classes,
        freeze_backbone=freeze_backbone,
        head_variant=args.head_type,
        head_hidden_dim=args.head_hidden_dim,
        head_norm=args.head_norm,
        head_dropout=args.head_dropout,
    )
    incompatible = load_pretrained_backbone(model, args.pretrained_ckpt, map_location="cpu")
    model = model.to(device)
    loss_fn = build_loss(
        loss_name=args.loss_name,
        n_classes=n_classes,
        class_weights=class_weights.to(device),
        lambda_dice=args.lambda_dice,
        lambda_ce=args.lambda_ce,
    )
    optimizer = build_optimizer(
        model=model,
        freeze_backbone=freeze_backbone,
        lr_head=args.lr_head,
        lr_backbone=args.lr_backbone,
        weight_decay=args.weight_decay,
    )

    run_dir = Path("/workspace/RemoteSensing/Pix2Rep-v2_Remi/output/remote_sensing_segmentation") / args.exp_name
    ckpt_dir = run_dir / "checkpoints"
    qualitative_dir = run_dir / "qualitative"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    qualitative_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "split.json").open("w") as f:
        json.dump(
            {
                "seed": args.seed,
                "strategy": args.strategy,
                "pretrained_ckpt": args.pretrained_ckpt,
                "original_channels": original_channels,
                "model_channels": int(cube.shape[-1]),
                "pca_components": args.pca_components,
                "pca_whiten": args.pca_whiten,
                "window_size": args.window_size,
                "overlap_size": args.overlap_size,
                "split_type": args.split_type,
                "train_num": args.train_num,
                "val_num": args.val_num,
                "train_ratio": args.train_ratio,
                "val_ratio": args.val_ratio,
                "train_indices": train_index.tolist(),
                "val_indices": val_index.tolist(),
                "test_indices": test_index.tolist(),
                "class_weights": class_weights.tolist(),
            },
            f,
            indent=2,
        )

    history: list[dict[str, float | int]] = []
    best_val_loss = float("inf")
    best_epoch = 0
    no_improve = 0
    best_model_path = ckpt_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            device=device,
            optimizer=optimizer,
        )
        val_loss, val_acc = run_epoch(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            optimizer=None,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_labeled_acc": train_acc,
                "val_loss": val_loss,
                "val_labeled_acc": val_acc,
            }
        )

        if epoch == 1 or epoch % args.log_every == 0:
            logger.info(
                "Epoch {}/{} | train_loss={:.4f} | train_acc={:.4f} | val_loss={:.4f} | val_acc={:.4f}",
                epoch,
                args.epochs,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "args": vars(args),
                },
                best_model_path,
            )
        else:
            no_improve += 1

        if args.early_stopping_patience > 0 and no_improve >= args.early_stopping_patience:
            logger.info(
                "Early stopping at epoch {} after {} epochs without val_loss improvement.",
                epoch,
                no_improve,
            )
            break

    best_checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"], strict=True)

    prediction_map = predict_scene(
        model=model,
        cube=cube,
        window_size=args.window_size,
        overlap_size=args.overlap_size,
        device=device,
    )
    confusion_matrix = compute_confusion_matrix(
        prediction_map=prediction_map,
        target_map=split.test_mask,
        n_classes=n_classes,
    )
    metrics = compute_metrics_from_confusion_matrix(confusion_matrix)

    np.save(run_dir / "prediction_map.npy", prediction_map.astype(np.int64))
    np.save(run_dir / "test_mask.npy", split.test_mask.astype(np.int64))
    Image.fromarray(colorize_label_map(prediction_map)).save(qualitative_dir / "prediction_map.png")
    Image.fromarray(colorize_label_map(split.test_mask)).save(qualitative_dir / "test_mask.png")
    Image.fromarray(colorize_label_map(labels.astype(np.int64))).save(qualitative_dir / "ground_truth_full.png")

    with (run_dir / "history.json").open("w") as f:
        json.dump(history, f, indent=2)
    with (run_dir / "test_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    summary = {
        "strategy": args.strategy,
        "freeze_backbone": freeze_backbone,
        "pretrained_ckpt": args.pretrained_ckpt,
        "best_model_path": str(best_model_path),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "test_oa": metrics["oa"],
        "test_aa": metrics["aa"],
        "test_kappa": metrics["kappa"],
        "head_type": args.head_type,
        "loss_name": args.loss_name,
        "window_size": args.window_size,
        "overlap_size": args.overlap_size,
        "n_classes": n_classes,
        "train_pixels": int((split.train_mask > 0).sum()),
        "val_pixels": int((split.val_mask > 0).sum()),
        "test_pixels": int((split.test_mask > 0).sum()),
        "missing_keys": incompatible["missing_keys"],
        "unexpected_keys": incompatible["unexpected_keys"],
    }
    with (run_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        "Finished Indian Pines linear probing | best_epoch={} | best_val_loss={:.4f} | test_OA={:.4f} | test_AA={:.4f} | test_Kappa={:.4f}",
        best_epoch,
        best_val_loss,
        metrics["oa"],
        metrics["aa"],
        metrics["kappa"],
    )


if __name__ == "__main__":
    main()
