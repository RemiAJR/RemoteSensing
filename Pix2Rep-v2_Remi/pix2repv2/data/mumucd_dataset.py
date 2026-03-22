import glob
import os
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
import torchio as tio
import xarray as xr
from einops import rearrange
from loguru import logger
from torch.utils.data import DataLoader, Dataset, random_split

from pix2repv2.utils import utils
from pix2repv2.utils.augmentations import RandomRescaleIntensity


@lru_cache(maxsize=2)
def _load_prisma_sr(path: str) -> np.ndarray:
    """Load PRISMA surface reflectance with LRU cache (per-worker process).
    Returns numpy array of shape (H, W, C=230), dtype float32.
    """
    ds = xr.open_dataset(path)
    sr = ds["sr"].values  # (H=1536, W=1536, C=230), float32
    ds.close()
    return sr


class MUMUCD_PatchSSL(Dataset):
    """Patch-based SSL dataset for MUMUCD PRISMA hyperspectral images.

    Each city provides one 'before' PRISMA image of shape (1536, 1536, 230).
    We generate `patches_per_image` random patches per city per epoch.
    Each patch is a random crop resized to (128, 128) with 230 spectral channels.
    """

    def __init__(
        self,
        cfg: dict,
        data_folder_path: str = "/workspace/RemoteSensing/data/mumucd",
        patches_per_image: int = 100,
        patch_size: int = 128,
        apply_augmentations: bool = True,
    ):
        self.cfg = cfg
        self.data_folder_path = data_folder_path
        self.patches_per_image = patches_per_image
        self.patch_size = patch_size
        self.apply_augmentations = apply_augmentations

        # Discover all before-prs.nc files across city subdirectories
        self.image_paths: list[str] = sorted(
            glob.glob(os.path.join(data_folder_path, "*", "*-before-prs.nc"))
        )
        assert len(self.image_paths) > 0, (
            f"No *-before-prs.nc files found in {data_folder_path}/*/. "
            "Make sure city subdirectories are extracted."
        )
        logger.info(
            f"MUMUCD PRISMA: found {len(self.image_paths)} cities, "
            f"{patches_per_image} patches each → {len(self)} samples/epoch"
        )

        self.intensity_rescaler = RandomRescaleIntensity(
            Imin_range=(0.0, 0.3),
            Imax_range=(0.7, 1.0),
            percentiles=(1, 99),
        )

        # SSL photometric augmentations (no RandomInvert for hyperspectral)
        cfg_transform = self.cfg.pretraining.pretraining_transform
        self.pretraining_transform = tio.Compose(
            [
                tio.RandomBiasField(
                    coefficients=cfg_transform.random_field_coef,
                    p=cfg_transform.random_field_p,
                ),
                RandomRescaleIntensity(
                    Imin_range=(0.0, 0.3),
                    Imax_range=(0.7, 1.0),
                    percentiles=(1, 99),
                ),
                tio.RandomBlur(
                    std=cfg_transform.random_blur_std,
                    p=cfg_transform.random_blur_p,
                ),
                tio.RandomGamma(
                    log_gamma=cfg_transform.log_gamma,
                    p=cfg_transform.random_gamma_p,
                ),
                tio.RandomNoise(
                    std=0.005,  # very light noise for hyperspectral data
                    p=cfg_transform.random_noise_p,
                ),
            ]
        )

    def __len__(self):
        return len(self.image_paths) * self.patches_per_image

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Map flat index → city
        city_idx = idx // self.patches_per_image
        path = self.image_paths[city_idx]

        # Load PRISMA image (cached per-worker via lru_cache)
        sr = _load_prisma_sr(path)  # (H=1536, W=1536, C=230)
        img_h, img_w, n_bands = sr.shape

        # Random fixed-size 128x128 crop
        ps = self.patch_size
        row = np.random.randint(0, img_h - ps + 1)
        col = np.random.randint(0, img_w - ps + 1)
        patch_np = sr[row : row + ps, col : col + ps, :]  # (128, 128, 230)

        # Rearrange to TorchIO format: (C=230, H=128, W=128, D=1)
        patch_tensor = torch.tensor(
            rearrange(patch_np, "H W C -> C H W 1"),
            dtype=torch.float32,
        )
        subject = tio.Subject(image=tio.ScalarImage(tensor=patch_tensor))

        # SSL photometric augmentations → two different views
        if self.apply_augmentations:
            view1 = self.pretraining_transform(subject["image"].data)
            view2 = self.pretraining_transform(subject["image"].data)
        else:
            view1 = self.intensity_rescaler(subject["image"].data)
            view2 = self.intensity_rescaler(subject["image"].data)
        assert view1.shape == view2.shape  # (C=230, W=128, H=128, D=1)

        # Rigid geometric transforms (no elastic for hyperspectral)
        affine_matrix = utils.generate_single_affine_spatial_transform(
            is_rotated=self.cfg.pretraining.is_rotated,
            is_cropped=self.cfg.pretraining.is_cropped,
            is_flipped=self.cfg.pretraining.is_flipped,
            is_translation=self.cfg.pretraining.is_translation,
            max_angle=self.cfg.pretraining.max_angle,
            max_crop=self.cfg.pretraining.max_crop,
        )
        grid = F.affine_grid(
            affine_matrix.unsqueeze(0),  # (1, 2, 3)
            rearrange(view1, "C W H D -> D C W H").shape,
            align_corners=True,
        )  # (1, W=128, H=128, 2)

        return view1, view2, grid


class ACDCPatchSupervised(Dataset):
    """
    Patch-based supervised ACDC dataset.
    A fixed-size 2D patch is extracted on-the-fly from each *labeled* 2D slice of the ACDC dataset,
    with 50% background patches and 50% heart patches (LabelSampler). Patches are augmented
    through photometric augmentations during finetuning (potentially different from pretraining).
    All images are resampled to a common pixel spacing prior to patch extraction.
    """

    def __init__(
        self,
        cfg: dict,
        data_folder_path: str = os.environ.get("ACDC_DATASET_FOLDER"),
        is_resampled: bool = True,
        target_spacing: float = 1.0,
        apply_augmentations: bool = True,
        apply_elastic_transform: bool = True,
        # List of patients for finetuning in different data regimes (i.e., len(selected_patients) = |X_tr|).
        # If None, all training patients are used.
        selected_patients: list[str] | None = None,
    ):
        self.cfg = cfg
        self.data_folder_path = data_folder_path
        self.is_resampled = is_resampled
        self.target_spacing = target_spacing  # target in plane spacing (s=sw=sh) in mm
        self.apply_augmentations = apply_augmentations
        self.apply_elastic_transform = apply_elastic_transform
        self.selected_patients = selected_patients
        self.slice_indices: list[tuple[str, str, int, str]] = list()

        all_patient_dirs = sorted(
            glob.glob(os.path.join(data_folder_path, "training", "patient*"))
        )

        # If a subset is specified, only keep their patient dirs
        if self.selected_patients is not None:
            selected_set = set(self.selected_patients)
            patient_dirs = [
                p_dir
                for p_dir in all_patient_dirs
                if os.path.basename(p_dir) in selected_set
            ]
        else:
            patient_dirs = all_patient_dirs

        for p_dir in patient_dirs:
            pid = os.path.basename(p_dir)
            info_path = os.path.join(p_dir, "Info.cfg")
            ed_frame_number, es_frame_number = utils._parse_acdc_info_cfg(info_path)
            frames = [ed_frame_number, es_frame_number]

            for frame in frames:
                img_path = os.path.join(p_dir, f"{pid}_frame{frame:02d}.nii.gz")
                gt_path = os.path.join(p_dir, f"{pid}_frame{frame:02d}_gt.nii.gz")

                if not (os.path.exists(img_path) and os.path.exists(gt_path)):
                    logger.warning(
                        f"Image or GT path does not exist for patient {pid}. Skipping..."
                    )
                    continue

                img = nib.load(img_path, mmap=True)
                W, H, Z = img.shape

                self.slice_indices.extend(
                    [(img_path, gt_path, z, pid) for z in range(Z)]
                )

        self.intensity_rescaler = RandomRescaleIntensity(
            Imin_range=(0.0, 0.3),
            Imax_range=(0.7, 1.0),
            percentiles=(1, 99),
        )

        # Photometric augmentations for supervised stage
        cfg_transform = self.cfg.finetuning.supervised_transform
        self.supervised_transform = tio.Compose(
            [
                tio.RandomBiasField(
                    coefficients=cfg_transform.random_field_coef,
                    p=cfg_transform.random_field_p,
                ),
                # out_min_max=(Imin, Imax) to reduce the gap between rescaling at the
                # patch-level (pretraining/finetuning) and rescaling at the slice-level (inference/eval)
                RandomRescaleIntensity(
                    Imin_range=(0.0, 0.3),
                    Imax_range=(0.7, 1.0),
                    percentiles=(1, 99),
                ),
                tio.RandomBlur(
                    std=cfg_transform.random_blur_std,
                    p=cfg_transform.random_blur_p,
                ),
                tio.RandomGamma(
                    log_gamma=cfg_transform.log_gamma,
                    p=cfg_transform.random_gamma_p,
                ),
                tio.RandomNoise(
                    std=cfg_transform.random_noise_std,
                    p=cfg_transform.random_noise_p,
                ),
                RandomInvert(p=cfg_transform.random_invert_p),
            ]
        )

    def __len__(self):
        return len(self.slice_indices)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, gt_path, d, pid = self.slice_indices[idx]
        # Lazy loading of 2d slice and GT mask from '3D' volume
        img_nii = nib.load(img_path, mmap=True)
        gt_nii = nib.load(gt_path, mmap=True)

        orig_spacing = img_nii.header.get_zooms()  # (sw, sh, sd)
        # RandomScale between (0.9, 1.1mm) to augment scale variability
        s_out = np.random.uniform(self.target_spacing - 0.1, self.target_spacing + 0.1)
        target_spacing = (s_out, s_out, orig_spacing[-1])

        img_slice = img_nii.dataobj[:, :, d]
        gt_slice = gt_nii.dataobj[:, :, d]

        img_tensor = torch.tensor(
            rearrange(img_slice, "W H -> 1 W H 1"),
            dtype=torch.float32,
        )
        gt_tensor = torch.tensor(
            rearrange(gt_slice, "W H -> 1 W H 1"),
            dtype=torch.int32,
        )

        subject = tio.Subject(
            image=tio.ScalarImage(tensor=img_tensor),
            label=tio.LabelMap(tensor=gt_tensor),
        )

        if self.is_resampled:
            img_resampled, actual_spacing_out = utils.resample(
                img=subject["image"].data,
                spacing_in=orig_spacing,
                spacing_out=target_spacing,
                mode="bicubic",
            )  # shape: (C, W, H, D)

            # Note: grid_sample() does not support type int. Need to cast the label map
            # to type 'float' prior to resampling and back to type 'int' after.
            label_resampled, _ = utils.resample(
                img=subject["label"].data.float(),
                spacing_in=orig_spacing,
                spacing_out=target_spacing,
                mode="nearest",
            )  # shape: (C, W, H, D)

            subject["image"].data = img_resampled
            subject["label"].data = label_resampled.int()

        # During finetuning, sample half background patches and half heart patches
        sampler = tio.data.LabelSampler(
            patch_size=(128, 128, 1),
            label_name="label",
            label_probabilities={
                0: 0.5,  # BACKGROUND
                1: 0.2,  # RV
                2: 0.1,  # MYO
                3: 0.2,  # LV
            },  # same prob for LV and RV
        )
        patch = next(
            sampler._generate_patches(subject, num_patches=1)
        )  # shape: (C=1, W=128, H=128, D=1)

        if self.apply_augmentations:
            if self.apply_elastic_transform:
                affine_matrix = torch.tensor([[1.0, 0, 0], [0, 1.0, 0]])
                with torch.no_grad():
                    grid = utils.generate_random_affine_elastic_grid(
                        p=self.cfg.finetuning.supervised_transform.elastic_transform_p,
                        # recall that patch has shape (C, W, H, D)
                        H=patch["image"].data.shape[2],
                        W=patch["image"].data.shape[1],
                        affine=affine_matrix,
                    )  # shape: (1, W, H, 2)

                # Apply elastic transform, then photometric augmentations
                patch_to_elastic = rearrange(patch["image"].data, "C W H D -> D C W H")
                patch_to_elastic = F.grid_sample(
                    patch_to_elastic,
                    grid,
                    mode="bicubic",
                    align_corners=True,
                )  # shape: (D=1, C=1, W=128, H=128)
                patch_to_photometric = rearrange(patch_to_elastic, "D C W H -> C W H D")
                img_data = self.supervised_transform(patch_to_photometric)

                gt_to_elastic = rearrange(patch["label"].data, "C W H D -> D C W H")
                gt_to_elastic = F.grid_sample(
                    gt_to_elastic.float(),
                    grid,
                    mode="nearest",
                    align_corners=True,
                ).int()  # shape: (D=1, C=1, W=128, H=128)
                gt_data = rearrange(gt_to_elastic, "D C W H -> C W H D")

            else:
                img_data = self.supervised_transform(patch["image"].data)
                gt_data = patch["label"].data

        else:
            img_data = self.intensity_rescaler(patch["image"].data)
            gt_data = patch["label"].data

        assert img_data.shape == gt_data.shape  # same spatial size

        return img_data, gt_data, pid, actual_spacing_out


class ACDCSupervisedEval(Dataset):
    """
    Dataset class used for evaluating the model on '3D' ACDC volumes.
    It considers the D-dim (depth) as the batch dimension and perform inference on each 2d slice
    independently. The evaluation will be performed using the 3D Dice Score, considering all slices
    of a given patient at a specific instant (ED or ES).
    The slices are resampled to a common pixel spacing prior to inference to match finetuning
    strategy. The prediction will be later resampled back to the original spacing to compute the
    Dice score in the initial ground truth space.
    """

    def __init__(
        self,
        cfg: dict,
        data_folder_path: str = os.environ.get("ACDC_DATASET_FOLDER"),
        is_resampled: bool = True,
        target_spacing: float = 1.0,  # in plane spacing (sw, sh) in mm
        split: str = "testing",
        selected_patients: list[str] | None = None,
    ):
        self.cfg = cfg
        self.data_folder_path = data_folder_path
        self.is_resampled = is_resampled
        self.target_spacing = target_spacing  # in plane spacing (w, h, d) in mm
        self.split = split
        self.selected_patients = selected_patients
        self.slice_indices: list[tuple[str, str, int, str]] = list()

        all_patient_dirs = sorted(
            glob.glob(os.path.join(data_folder_path, split, "patient*"))
        )

        # If a subset is specified, only keep their patient dirs
        if self.selected_patients is not None:
            selected_set = set(self.selected_patients)
            patient_dirs = [
                p_dir
                for p_dir in all_patient_dirs
                if os.path.basename(p_dir) in selected_set
            ]
        else:
            patient_dirs = all_patient_dirs

        for p_dir in patient_dirs:
            pid = os.path.basename(p_dir)
            info_path = os.path.join(p_dir, "Info.cfg")
            ed_frame_number, es_frame_number = utils._parse_acdc_info_cfg(info_path)
            frames = [ed_frame_number, es_frame_number]

            for frame in frames:
                img_path = os.path.join(p_dir, f"{pid}_frame{frame:02d}.nii.gz")
                gt_path = os.path.join(p_dir, f"{pid}_frame{frame:02d}_gt.nii.gz")

                if not (os.path.exists(img_path) and os.path.exists(gt_path)):
                    logger.warning(
                        f"Image or GT path does not exist for patient {pid}. Skipping..."
                    )
                    continue

                self.slice_indices.extend([(img_path, gt_path, pid)])

        self.intensity_rescaler = tio.RescaleIntensity(
            out_min_max=(0, 1),
            percentiles=(1, 99),
        )

    def __len__(self):
        # len = 100 since 50 test patients with two '3D' volume (ED/ES) each
        return len(self.slice_indices)

    def __getitem__(
        self, idx
    ) -> tuple[torch.Tensor, torch.Tensor, str, tuple[float, float, float]]:
        img_path, gt_path, pid = self.slice_indices[idx]
        # Loading of '3D' volume at a specific instant (ED or ES)
        img_nifti = nib.load(img_path)
        gt_nifti = nib.load(gt_path)

        orig_spacing = img_nifti.header.get_zooms()  # (sw, sh, sd)
        target_spacing = (self.target_spacing, self.target_spacing, orig_spacing[-1])

        img = img_nifti.get_fdata()
        gt = gt_nifti.get_fdata()

        img_tensor = torch.tensor(img, dtype=torch.float32)  # shape: (W, H, D)
        gt_tensor = torch.tensor(gt, dtype=torch.int32)

        subject = tio.Subject(
            image=tio.ScalarImage(tensor=img_tensor.unsqueeze(0)),
            label=tio.LabelMap(tensor=gt_tensor.unsqueeze(0)),
        )
        actual_spacing_out = orig_spacing
        if self.is_resampled:
            img_resampled, actual_spacing_out = utils.resample(
                img=subject["image"].data,
                spacing_in=orig_spacing,
                spacing_out=target_spacing,
                mode="bicubic",
            )  # shape: (C=1, W, H, D)
            subject["image"].data = img_resampled

        img_data = self.intensity_rescaler(subject["image"].data)
        gt_data = subject["label"].data  # shape: (C=1, W, H, D)

        return img_data, gt_data, pid, orig_spacing, actual_spacing_out


class ACDCSupervised4D(Dataset):
    """
    Dataset used for zero-shot 3D+t segmentation propagation on ACDC dataset.
    The dataset returns the full 4D ACDC volume along with the ED and ES ground truth masks.
    The first frame of the returned 4D volume (mr_volume_4d) corresponds to the ED instant and
    the last frame corresponds to the ES instant.

    -   is_resampled boolean argument controls whether the 4D volumes are resampled to a common
    pixel spacing prior to forward pass. If resampled, the predictions will be later resampled
    back to the original spacing.

    The ed_frame_number and es_frame_number are also returned to identify the frames
    corresponding to the ED and ES ground truth masks.
    """

    def __init__(
        self,
        data_folder_path: str = os.environ.get("ACDC_DATASET_FOLDER"),
        is_resampled: bool = True,
        target_spacing: float = 1.0,
        split: str = "training",
    ):
        self.data_folder_path = data_folder_path
        self.is_resampled = is_resampled
        self.target_spacing = target_spacing  # in plane spacing (w, h, d) in mm
        self.slice_indices: list[tuple[str, str, str, str, tuple[int, int]]] = list()

        all_patient_dirs = sorted(
            glob.glob(os.path.join(data_folder_path, split, "patient*"))
        )

        for p_dir in all_patient_dirs:
            pid = os.path.basename(p_dir)
            info_path = os.path.join(p_dir, "Info.cfg")
            ed_frame_number, es_frame_number = utils._parse_acdc_info_cfg(info_path)

            img_path = os.path.join(p_dir, f"{pid}_4d.nii.gz")
            ed_gt_path = os.path.join(
                p_dir,
                f"{pid}_frame{ed_frame_number:02d}_gt.nii.gz",
            )
            es_gt_path = os.path.join(
                p_dir,
                f"{pid}_frame{es_frame_number:02d}_gt.nii.gz",
            )

            if not (os.path.exists(img_path) and os.path.exists(ed_gt_path)):
                logger.warning(
                    f"Image or GT path does not exist for patient {pid}. Skipping..."
                )
                continue

            self.slice_indices.extend(
                [
                    (
                        img_path,
                        ed_gt_path,
                        es_gt_path,
                        pid,
                        (ed_frame_number, es_frame_number),
                    )
                ]
            )

        self.intensity_rescaler = tio.RescaleIntensity(
            out_min_max=(0, 1),
            percentiles=(1, 99),
        )

    def __len__(self):
        # len = 100 since 100 training patients
        return len(self.slice_indices)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, str]:
        (
            img_path,
            ed_gt_path,
            es_gt_path,
            pid,
            (ed_frame_number, es_frame_number),
        ) = self.slice_indices[idx]

        # Loading of '4D' volume with ground truth seg at ED
        img_nifti = nib.load(img_path)  # shape: (W, H, D, T)
        gt_ed_nifti = nib.load(ed_gt_path)  # shape: (W, H, D)
        gt_es_nifti = nib.load(es_gt_path)

        sw, sh, sd, st = img_nifti.header.get_zooms()  # (sw, sh, sd, st)
        orig_spacing = (sw, sh, sd)

        img = img_nifti.get_fdata()[:, :, :, ed_frame_number - 1 : es_frame_number]
        gt_ed = gt_ed_nifti.get_fdata()
        gt_es = gt_es_nifti.get_fdata()

        img_tensor = torch.tensor(img, dtype=torch.float32)  # shape: (W, H, D, T)
        gt_ed_tensor = torch.tensor(gt_ed, dtype=torch.int32)  # shape: (W, H, D)
        gt_es_tensor = torch.tensor(gt_es, dtype=torch.int32)

        if self.is_resampled:
            target_spacing = (self.target_spacing, self.target_spacing, sd)
            resampled_volumes = []

            for t in range(img_tensor.shape[-1]):
                mr_volume = img_tensor[:, :, :, t]
                # Resampling each 3D volume independently to target spacing.
                mr_volume_res, actual_spacing_out = utils.resample(
                    img=mr_volume.unsqueeze(0),
                    spacing_in=orig_spacing,  # only sw, sh is taken into account
                    spacing_out=target_spacing,
                    mode="bicubic",
                )  # shape: (C=1, W, H, D)

                subject = tio.Subject(image=tio.ScalarImage(tensor=mr_volume_res))
                mr_volume_res = self.intensity_rescaler(subject["image"].data)

                resampled_volumes.append(mr_volume_res)

            # Resampled 4D MR volume
            mr_volume_4d = torch.stack(
                resampled_volumes, dim=-1
            )  # shape: (C=1, W, H, D, T)
        else:
            subject = tio.Subject(image=tio.ScalarImage(tensor=img_tensor))
            mr_volume_4d = self.intensity_rescaler(subject["image"].data)
            mr_volume_4d = mr_volume_4d.unsqueeze(0)  # shape: (C=1, W, H, D, T)
            actual_spacing_out = orig_spacing

        return (
            mr_volume_4d,
            gt_ed_tensor,
            gt_es_tensor,
            (ed_frame_number, es_frame_number),
            pid,
            orig_spacing,
            actual_spacing_out,
        )


class ValidationWrapper(torch.utils.data.Dataset):
    """Wrapper to disable photometric augmentations for validation set."""

    def __init__(self, subset):
        self.subset = subset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        ds = self.subset.dataset

        # temporarily disable augmentation
        old = ds.apply_augmentations
        ds.apply_augmentations = False
        item = ds[self.subset.indices[idx]]
        ds.apply_augmentations = old

        return item


def select_nested_patient_ids(
    cfg: dict,
    data_folder_path: str = os.environ.get("ACDC_DATASET_FOLDER"),
) -> list[str]:
    """Returns a list of patient_ids to include in the dataset during finetuning.
    The function is implemented in a way to create a reproducible subset of
    labeled training slices for finetuning (depending on num_patients).
    The heuristic has the following properties:
        - Nested splits across data regimes (|X_tr| = 1 ⊂ |X_tr| = 2 ⊂ |X_tr| = 5 ...)
        - Balanced splits across pathologies (e.g. |X_tr| = 5 contains patients from all 5 pathologies in ACDC dataset)
        - Consistent splits across runs given cfg.data.random_seed

    Args:
        data_folder_path (str)
        cfg (dict): Hydra config dict.

    Returns:
        list[str]: List of patient ids to train on during finetuning.
    """
    patients_info = {"training": {}, "testing": {}}

    for train_or_test_folder in sorted(os.listdir(data_folder_path)):
        train_or_test_folder_path = os.path.join(data_folder_path, train_or_test_folder)
        if not os.path.isdir(train_or_test_folder_path):
            continue

        for patient_folder in sorted(os.listdir(train_or_test_folder_path)):
            patient_folder_path = os.path.join(
                train_or_test_folder_path, patient_folder
            )
            if not os.path.isdir(patient_folder_path):
                continue

            infos = {}
            patient_id = patient_folder.lstrip("patient")
            with open(os.path.join(patient_folder_path, "Info.cfg")) as f:
                for line in f:
                    label, value = line.split(":")
                    infos[label] = value.rstrip("\n").lstrip(" ")
            patients_info[train_or_test_folder][patient_id] = infos

    training_infos = patients_info["training"]

    base_groups = ["MINF", "NOR", "RV", "DCM", "HCM"]  # 5 pathologies in ACDC dataset
    patients_groups_ids: dict[str, list[str]] = {g: [] for g in base_groups}
    for patient_id, info in sorted(training_infos.items()):
        group = info["Group"]
        if group in patients_groups_ids:
            patients_groups_ids[group].append(patient_id)

    rng = random.Random(cfg.data.random_seed)
    rotation_offset = rng.randrange(len(base_groups))
    groups_order = base_groups[rotation_offset:] + base_groups[:rotation_offset]

    per_group_perm: dict[str, list[int]] = dict()
    for group in base_groups:
        n_p = len(patients_groups_ids[group])
        idx = list(range(n_p))
        rng.shuffle(idx)
        per_group_perm[group] = idx

    num_patients = int(cfg.finetuning.num_patients)
    num_groups = len(base_groups)
    subjects_per_group, r = divmod(num_patients, num_groups)

    selected_per_group: dict[str, list[int]] = {
        group: per_group_perm[group][
            : min(subjects_per_group, len(per_group_perm[group]))
        ]
        for group in base_groups
    }

    for group in groups_order[:r]:
        cur_len = len(selected_per_group[group])
        if cur_len < len(per_group_perm[group]):
            selected_per_group[group].append(per_group_perm[group][cur_len])

    selected_patient_ids: list[str] = list()
    for group in base_groups:
        group_patient_ids = patients_groups_ids[group]
        for patient_idx in selected_per_group[group]:
            selected_patient_ids.append(group_patient_ids[patient_idx])

    # convert ids ('001') to ACDC patient format ('patient001')
    selected_patients = [f"patient{int(pid):03d}" for pid in selected_patient_ids]

    return selected_patients


def create_train_val_subsets(
    patch_dataset: torch.utils.data.Dataset,
    cfg: dict = None,
) -> tuple[torch.utils.data.Subset, torch.utils.data.Subset]:
    """Create training/validation split for torch Dataset that does not
    have a predefined split, by taking val_ratio of the total (training) slices for validation.

    Args:
        patch_dataset (torch.utils.data.Dataset): Dataset to split.
        cfg (dict):

    Returns:
        tuple[torch.utils.data.Subset, torch.utils.data.Subset]: training and validation subsets.
    """
    total_slices = len(patch_dataset)
    val_slices = max(1, int(round(cfg.data.val_ratio * total_slices)))
    train_slices = total_slices - val_slices

    generator = torch.Generator().manual_seed(cfg.data.random_seed)
    train_subset, val_subset = random_split(
        dataset=patch_dataset,
        lengths=[train_slices, val_slices],
        generator=generator,
    )
    # wrap only validation to disable augmentations
    val_subset = ValidationWrapper(val_subset)

    return train_subset, val_subset


def build_patch_loader(
    patch_dataset: torch.utils.data.Dataset
    | torch.utils.data.Subset
    | torch.utils.data.ConcatDataset,
    batch_size: int,
    shuffle: bool,
) -> torch.utils.data.DataLoader:
    """Build a Dataloader for patch-based SSL pre-training and/or finetuning.

    Args:
        patch_dataset:
            Patch-based dataset from which batches should be drawn.
        batch_size (int): Batch size may changed between pretraining and finetuning.
        shuffle (bool): Whether to shuffle the dataset at each epoch.

    Returns:
        torch.utils.data.DataLoader: DataLoader for the given patch dataset.
    """
    num_workers = min(os.cpu_count() or 1, 2)
    logger.info(f"Using num_workers = {num_workers}")

    loader = DataLoader(
        patch_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    return loader
