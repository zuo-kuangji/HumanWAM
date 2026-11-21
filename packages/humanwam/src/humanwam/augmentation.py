"""Shared HB training augmentation in RGB and raw joint coordinates.

The caller selects this profile and supplies a stable sample key. Observations
and visual targets share scene transforms. Only observations receive sensor
degradation. This API never receives action labels or model state features.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from numbers import Integral

import numpy as np

UNIFIED_V1 = "humanwam_unified_augmentation_v1"
UNIFIED_V2 = "humanwam_unified_augmentation_v2"
_CLEAN_IMAGE_PROBABILITIES = {UNIFIED_V1: 0.20, UNIFIED_V2: 0.05}
_HB_PROFILES = {"unified": UNIFIED_V2, "unified_v2": UNIFIED_V2, "unified_v1": UNIFIED_V1}


@dataclass(frozen=True)
class SampleKey:
    """Episode and frame are the dataset's IDs, independent of worker or rank."""

    episode: int
    frame: int
    epoch: int = 0

    def __post_init__(self):
        for value in (self.episode, self.frame, self.epoch):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError("Episode, frame and epoch must be nonnegative integers")


class UnifiedAugmentation:
    """Replayable, opt-in HumanWAM visual and raw proprioception recipe."""

    profile = UNIFIED_V2

    def __init__(self, seed=42, *, profile=UNIFIED_V2):
        if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
            raise ValueError("Augmentation seed must be a nonnegative integer")
        if not isinstance(profile, str) or profile not in _CLEAN_IMAGE_PROBABILITIES:
            raise ValueError(f"Unknown unified augmentation profile: {profile!r}")
        self.seed = int(seed)
        self.profile = profile

    @classmethod
    def from_hb_profile(cls, profile, *, seed=42):
        """Resolve the HB launcher alias to a specific shared recipe."""
        if not isinstance(profile, str) or profile not in _HB_PROFILES:
            raise ValueError(f"Unknown HB_AUG_PROFILE: {profile!r}; choose unified_v1 or unified_v2")
        return cls(seed=seed, profile=_HB_PROFILES[profile])

    def _rng(self, sample, stream, index=0):
        if not isinstance(sample, SampleKey):
            raise TypeError("Use an explicit SampleKey for shared augmentation")
        return np.random.default_rng([
            self.seed, int(sample.episode), int(sample.frame), int(sample.epoch), stream, index,
        ])

    def sample_rgb(self, sample, *, cameras):
        """Sample resolution-independent scene and sensor parameters."""
        if isinstance(cameras, bool) or not isinstance(cameras, Integral) or cameras < 1:
            raise ValueError("cameras must be a positive integer")
        rng = self._rng(sample, 0)
        if rng.random() < _CLEAN_IMAGE_PROBABILITIES[self.profile]:
            return None
        strong = rng.choice(("affine", "overexposure", "noise", "jpeg")) if rng.random() < 0.15 else None
        parameters = {
            "strong": strong,
            "exposure_ev": rng.uniform(-0.22, 0.22),
            "overexposure_gain": rng.uniform(1.25, 1.8) if strong == "overexposure" else 1.0,
            "temperature": rng.uniform(-0.066, 0.066),
            "contrast": rng.uniform(0.88, 1.12),
            "saturation": rng.uniform(0.76, 1.24),
            "hue": rng.uniform(-0.015, 0.015),
            "gamma": rng.uniform(0.89, 1.11),
            "cameras": [],
        }
        for camera in range(cameras):
            view = self._rng(sample, 1, camera)
            angle = float(view.uniform(-1.5, 1.5))
            scale = 1.0 + abs(angle) * 0.015 + float(view.uniform(0, 0.02))
            translation = (0.0, 0.0)
            if strong == "affine":
                angle = float(view.uniform(-3.0, 3.0))
                scale = float(view.uniform(0.96, 1.04))
                translation = tuple(view.uniform(-0.04, 0.04, size=2))
            shadows = []
            if view.random() < 0.30:
                for _ in range(int(view.integers(1, 3))):
                    cx, cy = view.uniform(0.1, 0.9, size=2)
                    shadows.append((cx, cy, view.uniform(0.15, 0.35), view.uniform(0.70, 1.30)))
            gains = [1.0, 1.0, 1.0]
            if view.random() < 0.30:
                gains[int(view.integers(3))] = view.uniform(0.9, 1.1)
            sensor = self._rng(sample, 2, camera)
            parameters["cameras"].append({
                "angle": angle,
                "scale": scale,
                "translation": translation,
                "shadows": shadows,
                "gains": gains,
                "blur": sensor.uniform(0.25, 0.65) if sensor.random() < 0.15 else 0.0,
                "motion_blur": (sensor.uniform(3, 7), sensor.uniform(0, math.pi))
                    if sensor.random() < 0.15 else None,
                "sharpen": sensor.uniform(0.2, 0.6) if sensor.random() < 0.15 else 0.0,
                "defocus": sensor.uniform(1, 2) if sensor.random() < 0.10 else 0.0,
                "noise": 8 / 255 if strong == "noise" else (4 / 255 if sensor.random() < 0.5 else 0.0),
                "jpeg": int(sensor.integers(50, 81)) if strong == "jpeg" else (
                    int(sensor.integers(70, 96)) if sensor.random() < 0.35 else None
                ),
            })
        return parameters

    @staticmethod
    def _scene(images, scene, view):
        import torch
        import torch.nn.functional as F
        from torchvision.transforms import functional as TF

        height, width = images.shape[-2:]
        theta = math.radians(view["angle"])
        cos, sin = math.cos(theta) / view["scale"], math.sin(theta) / view["scale"]
        tx, ty = view["translation"]
        affine = images.new_tensor([
            [cos, -sin * height / width, 2 * tx], [sin * width / height, cos, 2 * ty],
        ])
        grid = F.affine_grid(affine[None].expand(len(images), -1, -1), images.shape, align_corners=False)
        images = F.grid_sample(images, grid, mode="bilinear", padding_mode="border", align_corners=False)
        if view["shadows"]:
            y = torch.linspace(0, 1, height, dtype=images.dtype)[:, None]
            x = torch.linspace(0, 1, width, dtype=images.dtype)[None, :]
            light = torch.ones_like(images[0, 0])
            for cx, cy, size, gain in view["shadows"]:
                light += (gain - 1) * torch.exp(-((x - cx).square() + (y - cy).square()) / (2 * size**2))
            images = (images * light.clamp_min(0)).clamp(0, 1)
        temp = scene["temperature"]
        gains = images.new_tensor(view["gains"]) * images.new_tensor([1 + temp, 1, 1 - temp])
        images = (images * gains[None, :, None, None] * (2 ** scene["exposure_ev"])).clamp(0, 1)
        images = ((images - 0.5) * scene["contrast"] + 0.5).clamp(0, 1)
        images = TF.adjust_saturation(images, scene["saturation"])
        images = TF.adjust_hue(images, scene["hue"]).clamp(0, 1).pow(scene["gamma"])
        return (images * scene["overexposure_gain"]).clamp(0, 1)

    @staticmethod
    def _filter(images, kernel):
        import torch.nn.functional as F

        pad = kernel.shape[-1] // 2
        weights = (kernel / kernel.sum()).to(images)[None, None].expand(3, 1, -1, -1)
        return F.conv2d(F.pad(images, (pad,) * 4, mode="replicate"), weights, groups=3)

    def _sensor(self, images, view, sample, camera):
        import torch
        from PIL import Image
        from torchvision.transforms import functional as TF

        scale = images.shape[-2] / 224.0
        if view["blur"]:
            sigma = view["blur"] * scale
            size = 2 * max(1, math.ceil(2 * sigma)) + 1
            # Keep torchvision's reflection padding smaller than the input.
            size = min(size, 2 * (min(images.shape[-2:]) - 1) + 1)
            images = TF.gaussian_blur(images, [size, size], [sigma, sigma])
        if view["motion_blur"] is not None:
            length, direction = view["motion_blur"]
            length = max(1.0, length * scale)
            radius = math.ceil(length / 2)
            kernel = torch.zeros(2 * radius + 1, 2 * radius + 1)
            positions = torch.linspace(-length / 2, length / 2, 8 * radius + 1)
            x = (positions * math.cos(direction)).round().long() + radius
            y = (positions * math.sin(direction)).round().long() + radius
            kernel[y, x] = 1
            images = self._filter(images, kernel)
        if view["defocus"]:
            radius = view["defocus"] * scale
            offsets = torch.arange(-math.ceil(radius), math.ceil(radius) + 1)
            disk = (offsets[:, None].square() + offsets[None, :].square() <= (radius + 0.5) ** 2)
            images = self._filter(images, disk.float())
        if view["sharpen"]:
            smooth = TF.gaussian_blur(images, [5, 5], [1.0, 1.0])
            images = (images + view["sharpen"] * (images - smooth)).clamp(0, 1)
        if view["noise"]:
            seed = int(self._rng(sample, 3, camera).integers(2**63 - 1))
            generator = torch.Generator().manual_seed(seed)
            # Each observed frame has its own noise array.
            images = (images + torch.randn(images.shape, generator=generator) * view["noise"]).clamp(0, 1)
        if view["jpeg"] is not None:
            restored = []
            for frame in images:
                pixels = frame.clamp(0, 1).mul(255).round().to(torch.uint8).permute(1, 2, 0).numpy()
                with io.BytesIO() as stream:
                    Image.fromarray(pixels).save(stream, format="JPEG", quality=view["jpeg"], subsampling=0)
                    stream.seek(0)
                    with Image.open(stream) as compressed:
                        restored.append(TF.pil_to_tensor(compressed.convert("RGB")).float().div(255))
            images = torch.stack(restored)
        return images.clamp(0, 1)

    def apply_rgb(self, clip, *, sample, observation_mask):
        """Apply scene transforms to [T,V,3,H,W], sensor effects to masked frames."""
        import torch

        if (
            not torch.is_tensor(clip) or not clip.is_floating_point() or clip.device.type != "cpu"
            or clip.ndim != 5 or clip.shape[2] != 3 or min(clip.shape[:2]) < 1
            or min(clip.shape[-2:]) < 3 or not bool(torch.isfinite(clip).all())
            or bool((clip < 0).any()) or bool((clip > 1).any())
        ):
            raise ValueError("Expected finite CPU float RGB [T,V,3,H,W] in [0,1], H and W at least 3")
        mask = torch.as_tensor(observation_mask)
        if mask.dtype != torch.bool or mask.device.type != "cpu" or mask.shape != (len(clip),):
            raise ValueError("observation_mask must be a CPU boolean vector of length T")
        scene = self.sample_rgb(sample, cameras=clip.shape[1])
        if scene is None:
            return clip.clone()
        views = []
        for camera, parameters in enumerate(scene["cameras"]):
            images = self._scene(clip[:, camera].float(), scene, parameters)
            if bool(mask.any()):
                images[mask] = self._sensor(images[mask], parameters, sample, camera)
            views.append(images)
        return torch.stack(views, dim=1).to(clip.dtype)

    def apply_state(self, state, *, body_dims, sample):
        """Perturb leading body joint angles in radians before native normalization."""
        value = np.asarray(state)
        if (
            not np.issubdtype(value.dtype, np.floating) or value.ndim < 1
            or not isinstance(body_dims, Integral) or body_dims not in (29, 31)
            or value.shape[-1] < body_dims or not np.isfinite(value).all()
        ):
            raise ValueError("Expected finite raw robot state with 29 (G1) or 31 (H2) leading body angles")
        rng = self._rng(sample, 4)
        result = value.copy()
        body = value[..., :body_dims].astype(np.float64)
        bias = rng.uniform(-math.radians(2), math.radians(2), size=body_dims)
        sigma = math.radians(0.25)
        noise = np.clip(rng.normal(0, sigma, size=body.shape), -3 * sigma, 3 * sigma)
        result[..., :body_dims] = body + bias + noise
        return result
