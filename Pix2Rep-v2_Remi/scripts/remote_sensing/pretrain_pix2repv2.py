import argparse

import torch
import wandb_osh
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from loguru import logger
from wandb_osh.lightning_hooks import TriggerWandbSyncLightningCallback

from pix2repv2.data.mumucd_ssl import (
    MUMUCD_PatchSSL,
    build_patch_loader,
    create_train_val_subsets,
)
from pix2repv2.pretraining_module import Pix2RepPretrainingModule
from pix2repv2.utils.utils import Config, ModelCheckpointSymlink

wandb_osh.set_log_level("ERROR")


def run_training(
    cfg: dict,
    train_loader_ssl: torch.utils.data.DataLoader,
    val_loader_ssl: torch.utils.data.DataLoader,
    experiment_name: str,
    distributed: bool,
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

    if distributed:
        logger.info("Distributed training: enabled")
        strategy = "ddp"
        devices = 4
        num_nodes = 2
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
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            TriggerWandbSyncLightningCallback(),
            last_ckpt_callback,
            best_ckpt_callback,
        ],
    )
    trainer.fit(model, train_loader_ssl, val_loader_ssl)

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
    logger.info("MUMUCD PRISMA dataloaders construction")
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
    )


if __name__ == "__main__":
    main()
