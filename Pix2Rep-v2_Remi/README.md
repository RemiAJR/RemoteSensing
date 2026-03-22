## Pix2Rep-v2_Remi: Remote-Sensing Adaptation for MUMUCD

This fork is the remote-sensing adaptation of Pix2Rep-v2. The actively supported
pretraining path in this repository targets the MUMUCD PRISMA hyperspectral dataset.

## Pretraining on MUMUCD

Run pretraining with:

```bash
python -m scripts.remote_sensing.pretrain_pix2repv2 --pretraining_dataset MUMUCD --exp_name {your_experiment_name}
```

The legacy compatibility entrypoint still works:

```bash
python -m scripts.cardiac.pretrain_pix2repv2 --pretraining_dataset MUMUCD --exp_name {your_experiment_name}
```

To override Hydra config values:

```bash
python -m scripts.remote_sensing.pretrain_pix2repv2 --pretraining_dataset MUMUCD --exp_name {your_experiment_name} --overrides data.batch_size_pretraining={your_batch_size} pretraining.lr_backbone={your_lr}
```

## Data Layout

`MUMUCD_PatchSSL` expects extracted files matching:

```text
/workspace/RemoteSensing/data/mumucd/<city>/<city>-before-prs.nc
```

If no matching files are present, dataset initialization will fail immediately.

## Environment

Install the dependencies declared in `pyproject.toml` before launching training.
This fork requires packages such as `torch`, `lightning`, `numpy`, `xarray`,
`torchio`, and `wandb`.
