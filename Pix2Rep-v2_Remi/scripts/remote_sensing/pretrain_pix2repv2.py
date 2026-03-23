import argparse
import time
from pathlib import Path

import torch
import wandb_osh
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from loguru import logger

from pix2repv2.data.mumucd_ssl import (
    MUMUCD_PatchSSL,
    build_patch_loader,
    create_train_val_subsets,
)
from pix2repv2.pretraining_module import Pix2RepPretrainingModule
from pix2repv2.utils.utils import Config, ModelCheckpointSymlink

wandb_osh.set_log_level("ERROR")


def _to_float(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach()
        if value.numel() == 1:
            return float(value.cpu().item())
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


class EpochTimingCallback(Callback):
    def __init__(self):
        self._epoch_start_time = None
        self._validation_start_time = None

    def on_train_epoch_start(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        self._epoch_start_time = time.perf_counter()
        epoch = trainer.current_epoch + 1
        logger.info(f"Epoch {epoch}/{trainer.max_epochs} started")

    def on_validation_epoch_start(self, trainer, pl_module):
        if trainer.sanity_checking or not trainer.is_global_zero:
            return
        self._validation_start_time = time.perf_counter()

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or not trainer.is_global_zero:
            return
        epoch = trainer.current_epoch + 1
        train_loss = _to_float(trainer.callback_metrics.get("train_loss"))
        val_loss = _to_float(trainer.callback_metrics.get("val_loss"))
        epoch_minutes = None
        val_minutes = None
        if self._epoch_start_time is not None:
            epoch_minutes = (time.perf_counter() - self._epoch_start_time) / 60.0
        if self._validation_start_time is not None:
            val_minutes = (time.perf_counter() - self._validation_start_time) / 60.0

        summary = [f"Epoch {epoch}/{trainer.max_epochs} complete"]
        if train_loss is not None:
            summary.append(f"train_loss={train_loss:.4f}")
        if val_loss is not None:
            summary.append(f"val_loss={val_loss:.4f}")
        if epoch_minutes is not None:
            summary.append(f"epoch_time={epoch_minutes:.2f} min")
        if val_minutes is not None:
            summary.append(f"val_time={val_minutes:.2f} min")
        summary.append(f"global_step={trainer.global_step}")
        logger.info(" | ".join(summary))


class StepProgressLoggerCallback(Callback):
    def __init__(self, every_n_steps: int):
        self.every_n_steps = every_n_steps

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not trainer.is_global_zero or self.every_n_steps <= 0:
            return
        if trainer.global_step == 0 or trainer.global_step % self.every_n_steps != 0:
            return

        loss = _to_float(trainer.callback_metrics.get("train_loss_step"))
        lr = None
        if trainer.optimizers:
            lr = trainer.optimizers[0].param_groups[0]["lr"]

        epoch = trainer.current_epoch + 1
        num_batches = trainer.num_training_batches
        batch_position = batch_idx + 1
        message = f"Epoch {epoch}/{trainer.max_epochs} | batch {batch_position}/{num_batches} | global_step={trainer.global_step}"
        if loss is not None:
            message += f" | train_loss_step={loss:.4f}"
        if lr is not None:
            message += f" | lr={lr:.2e}"
        logger.info(message)


def resolve_resume_checkpoint(checkpoint_dir: str, resume_from: str | None) -> str | None:
    if resume_from is None:
        return None

    checkpoint_dir = Path(checkpoint_dir)
    named_paths = {
        "auto": [
            checkpoint_dir / "latest_step.ckpt",
            checkpoint_dir / "last.ckpt",
        ],
        "latest_step": [checkpoint_dir / "latest_step.ckpt"],
        "last": [checkpoint_dir / "last.ckpt"],
    }

    if resume_from in named_paths:
        candidates = named_paths[resume_from]
    else:
        candidates = [Path(resume_from)]

    for candidate in candidates:
        if candidate.exists():
            logger.info(f"Resuming training from checkpoint: {candidate}")
            return str(candidate)

    if resume_from == "auto":
        logger.info("No existing checkpoint found for --resume_from=auto. Starting from scratch.")
        return None

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"No checkpoint found for --resume_from={resume_from}. Searched: {searched}")


def run_training(
    cfg: dict,
    train_loader_ssl: torch.utils.data.DataLoader,
    val_loader_ssl: torch.utils.data.DataLoader,
    experiment_name: str,
    distributed: bool,
    resume_from: str | None,
):
    wandb_logger = WandbLogger(
        project="Pix2Repv2-remote-sensing-pretraining",
        name=experiment_name,
        save_dir="output",
    )

    checkpoint_dir = (
        f"output/lightning_logs/pretraining/mumucd/{experiment_name}/checkpoints/"
    )
    last_ckpt_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        save_last="link",
        save_top_k=0,
        monitor=None,
    )

    best_ckpt_callback = ModelCheckpointSymlink(
        dirpath=checkpoint_dir,
        save_top_k=3,
        save_weights_only=True,
        monitor="val_loss",
        mode="min",
    )
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        EpochTimingCallback(),
        StepProgressLoggerCallback(cfg.pretraining.log_every_n_steps),
        last_ckpt_callback,
        best_ckpt_callback,
    ]
    if cfg.pretraining.step_checkpoint_every_n_train_steps > 0:
        callbacks.append(
            ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename="latest_step",
                monitor=None,
                every_n_train_steps=cfg.pretraining.step_checkpoint_every_n_train_steps,
                save_top_k=1,
                save_weights_only=False,
                save_on_train_epoch_end=False,
                enable_version_counter=False,
            )
        )

    if distributed:
        n_gpus = torch.cuda.device_count()
        logger.info(f"Distributed training: enabled ({n_gpus} GPUs)")
        strategy = "ddp"
        devices = n_gpus
        num_nodes = 1
        sync_batchnorm = (
            hasattr(cfg.pretraining.backbone.params, "norm")
            and cfg.pretraining.backbone.params.norm != "batch"
        )
    else:
        logger.info("Distributed training: disabled")
        strategy = "auto"
        devices = 1
        num_nodes = 1
        sync_batchnorm = False

    ckpt_path = resolve_resume_checkpoint(checkpoint_dir, resume_from)

    model = Pix2RepPretrainingModule(cfg)
    trainer = Trainer(
        accelerator="cuda",
        strategy=strategy,
        devices=devices,
        num_nodes=num_nodes,
        logger=wandb_logger,
        max_epochs=cfg.pretraining.num_epochs,
        sync_batchnorm=sync_batchnorm,
        accumulate_grad_batches=1,
        num_sanity_val_steps=0,
        limit_val_batches=10,
        log_every_n_steps=cfg.pretraining.log_every_n_steps,
        use_distributed_sampler=True,
        callbacks=callbacks,
    )
    # Lightning forwards this to torch.load when restoring a training checkpoint.
    # Our existing checkpoints contain OmegaConf objects and defaultdict state that
    # are not compatible with PyTorch 2.6's weights_only=True default.
    trainer.fit(
        model,
        train_loader_ssl,
        val_loader_ssl,
        ckpt_path=ckpt_path,
        weights_only=False,
    )

    if num_nodes > 1 and trainer.strategy.launcher is not None:
        trainer.strategy.barrier("end_of_training")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pretrain Pix2Rep-v2 on the MUMUCD remote-sensing dataset."
    )
    parser.add_argument(
        "--pretraining_dataset",
        type=str,
        default="MUMUCD",
        help="Remote-sensing fork currently supports only MUMUCD.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        required=True,
        help="Experiment name for checkpoints and W&B runs.",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        default=False,
        help="Whether to use distributed training (DDP).",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Checkpoint to resume from: auto, last, latest_step, or an explicit path.",
    )
    parser.add_argument("--overrides", nargs="*", default=[])
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.pretraining_dataset != "MUMUCD":
        raise ValueError(
            "This remote-sensing fork supports only --pretraining_dataset MUMUCD."
        )

    cfg = Config(overrides=args.overrides).cfg
    seed_everything(cfg.data.random_seed, workers=True)

    mumucd_ssl_patch_ds = MUMUCD_PatchSSL(
        cfg=cfg,
        apply_augmentations=True,
    )
    train_subset, val_subset = create_train_val_subsets(
        patch_dataset=mumucd_ssl_patch_ds,
        cfg=cfg,
    )
    cfg.data.train_dataset_len = len(train_subset)
    logger.info(f"Train dataset: {len(train_subset)} samples, Val: {len(val_subset)} samples")

    train_loader_ssl = build_patch_loader(
        patch_dataset=train_subset,
        batch_size=cfg.data.batch_size_pretraining,
        shuffle=False,
    )
    val_loader_ssl = build_patch_loader(
        patch_dataset=val_subset,
        batch_size=cfg.data.batch_size_pretraining,
        shuffle=False,
    )

    run_training(
        cfg=cfg,
        train_loader_ssl=train_loader_ssl,
        val_loader_ssl=val_loader_ssl,
        experiment_name=args.exp_name,
        distributed=args.distributed,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()
