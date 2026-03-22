from collections import defaultdict
from typing import Union

import numpy as np
import torch
import torchio as tio
_CONNECTOME_AVAILABLE = False
try:
    from connectome import Input, Transform, inverse, optional, positional
    _CONNECTOME_AVAILABLE = True
except Exception:
    pass
from torchio.data.subject import Subject
from torchio.transforms import Blur, IntensityTransform
from torchio.transforms.augmentation import RandomTransform


class RandomInvert(tio.Transform):
    """Reimplementation of torchvision RandomInvert augmentation under the torchIO format.
    Image intensities are clamped to prevent negative values after inversion.
    """

    def __init__(self, p: float, min_val: float = 0.0, max_val: float = 1.0):
        super().__init__(p=p)
        self.min_val = min_val
        self.max_val = max_val

    def apply_transform(self, subject):
        for image in subject.get_images(intensity_only=True):
            x = image.data
            x = x.clamp(self.min_val, self.max_val)
            image.set_data(self.max_val - x)
        return subject


class RandomRescaleIntensity(tio.Transform):
    def __init__(
        self,
        Imin_range=(0.0, 0.3),
        Imax_range=(0.7, 1.0),
        percentiles=(1, 99),
    ):
        super().__init__()
        self.Imin_range = Imin_range
        self.Imax_range = Imax_range
        self.percentiles = percentiles

    def apply_transform(self, subject):
        Imin = np.random.uniform(*self.Imin_range)
        Imax = np.random.uniform(*self.Imax_range)
        if Imin >= Imax:
            return subject

        for image in subject.get_images(intensity_only=True):
            data = image.data.numpy()
            lo = np.percentile(data, self.percentiles[0])
            hi = np.percentile(data, self.percentiles[1])
            if hi - lo < 1e-8:
                continue
            data = (data - lo) / (hi - lo)
            data = data * (Imax - Imin) + Imin
            np.clip(data, Imin, Imax, out=data)
            image.set_data(torch.from_numpy(data))
        return subject


class HUWindow(IntensityTransform):
    """Apply HU windowing to an image."""

    def __init__(
        self,
        min_hu,
        max_hu,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.min_hu = min_hu
        self.max_hu = max_hu

    def apply_transform(self, subject):
        images = self.get_images_dict(subject)
        if not images:
            return subject

        for name, image in images.items():
            # Unpack per-image parameters
            min_hu = self.min_hu[name]
            max_hu = self.max_hu[name]

            if min_hu >= max_hu:
                raise ValueError(f"Invalid HU window: [{min_hu}, {max_hu}]")

            data = image.data
            data = (data - min_hu) / (max_hu - min_hu)
            data = torch.clamp(data, 0, 1)
            image.set_data(data)

        return subject


class RandomHUWindow(RandomTransform, IntensityTransform):
    """Random HU windowing."""

    def __init__(
        self,
        min_window_hu: tuple[float, float],
        max_window_hu: tuple[float, float],
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.min_window_hu = self._parse_range(min_window_hu, "min_window_hu")
        self.max_window_hu = self._parse_range(max_window_hu, "max_window_hu")

    def get_params(
        self,
        min_window_hu: tuple[float, float],
        max_window_hu: tuple[float, float],
    ) -> tuple[float, float]:
        min_hu = self.sample_uniform(max_window_hu[0], min_window_hu[0])
        max_hu = self.sample_uniform(min_window_hu[1], max_window_hu[1])

        assert min_hu < max_hu, f"Invalid HU window: [{min_hu}, {max_hu}]"

        return min_hu, max_hu

    def apply_transform(self, subject):
        images_dict = self.get_images_dict(subject)
        if not images_dict:
            return subject

        min_hu, max_hu = self.get_params(
            self.min_window_hu,
            self.max_window_hu,
        )

        arguments = defaultdict(dict)
        for name in images_dict:
            arguments["min_hu"][name] = min_hu
            arguments["max_hu"][name] = max_hu

        transform = HUWindow(**self.add_base_args(arguments))
        transformed = transform(subject)

        return transformed


class RandomBlurOrSharpen(RandomTransform, IntensityTransform):
    r"""Randomly blur or sharpen an image.

    Sharpening is implemented as an unsharp mask using TorchIO's Blur transform:
        output = image + alpha * (image - blurred_image)
    """

    def __init__(
        self,
        p: float,
        std: Union[float, tuple[float, float]] = (0.25, 1.5, 0.25, 1.5, 0.0, 0.0),
        alpha: tuple[float, float] = (0.2, 0.99),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.p = p
        self.std_ranges = self.parse_params(std, None, "std", min_constraint=0)
        self.alpha_range = self._parse_range(alpha, "alpha")

    def apply_transform(self, subject: Subject) -> Subject:
        images_dict = self.get_images_dict(subject)
        if not images_dict:
            return subject

        if self.sample_uniform(0, 1) > self.p:
            return subject

        # Decide once per subject: blur OR sharpen with equal chance
        do_blur = self.sample_uniform(0, 1) < 0.5

        arguments: dict[str, dict] = defaultdict(dict)
        alpha_dict: dict[str, float] = {}

        for name in images_dict:
            std = self.get_params(self.std_ranges)
            arguments["std"][name] = std

            if not do_blur:
                alpha_dict[name] = self.sample_uniform(*self.alpha_range)

        if do_blur:
            transform = Blur(**self.add_base_args(arguments))
            transformed = transform(subject)
        else:
            # --- Unsharp mask ---
            blurred = Blur(**self.add_base_args(arguments))(subject)
            assert isinstance(blurred, Subject)

            for name, image in images_dict.items():
                # original = subject[name].data
                original = image.data
                blurred_data = blurred[name].data
                # alpha = arguments["alpha"][name]
                alpha = alpha_dict[name]

                image.set_data(original + alpha * (original - blurred_data))

            transformed = subject

        assert isinstance(transformed, Subject)
        return transformed

    def get_params(self, std_ranges):
        sx, sy, sz = self.sample_uniform_sextet(std_ranges)
        return sx, sy, sz


try:
    class FlipAxesToCanonical(Transform):
        __inherit__ = True

        @positional
        def image(x, flipped_axes):
            if not flipped_axes:
                return x
            return np.flip(x, flipped_axes).copy()

        @optional
        @positional
        def mask(x, flipped_axes):
            if not flipped_axes or x is None:
                return x
            return np.flip(x, flipped_axes).copy()

        def flipped_axes(flipped_axes):
            return ()

        @inverse
        def sgm(sgm, flipped_axes: Input):
            if not flipped_axes:
                return sgm
            return np.flip(sgm, flipped_axes).copy()
except Exception:
    pass  # connectome metaclass incompatible with Python 3.13+
