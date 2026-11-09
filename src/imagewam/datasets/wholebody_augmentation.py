"""Paired train-time RGB augmentation for a single whole-body camera clip.

All frames use the same sampled parameters. Operations never mix time steps,
so the current-frame transformation cannot depend on a future target image.
The dataset owns the generator/epoch policy and calls this on decoded copies.
This module never reads actions, proprioception, text or FM source latents.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from imagewam.augmentation_config import normalize_visual_config


DEFAULT_VISUAL_AUGMENTATION = normalize_visual_config()


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_plain(item) for item in value]
    return value


def _event(probability, generator):
    if probability <= 0:
        return False
    if probability >= 1:
        return True
    return bool(torch.rand((), generator=generator) < probability)


def _uniform(bounds, generator):
    lo, hi = bounds
    return lo if lo == hi else lo + (hi - lo) * float(torch.rand((), generator=generator))


def _hue(rgb, amount):
    """HSV hue rotation in cycles, with achromatic pixels left achromatic."""
    if amount == 0:
        return rgb
    maximum, index = rgb.max(dim=1)
    minimum = rgb.min(dim=1).values
    delta = maximum - minimum
    red, green, blue = rgb.unbind(dim=1)
    safe = delta.clamp(min=1e-10)
    h = torch.where(index == 0, (green - blue) / safe,
                    torch.where(index == 1, (blue - red) / safe + 2, (red - green) / safe + 4))
    h = torch.where(delta > 0, h / 6, torch.zeros_like(h))
    h = torch.remainder(h + amount, 1) * 6
    saturation = torch.where(maximum > 0, delta / maximum.clamp(min=1e-10), torch.zeros_like(delta))
    i = torch.floor(h).to(torch.int64)
    f = h - i
    p, q, t = maximum * (1 - saturation), maximum * (1 - saturation * f), maximum * (1 - saturation * (1 - f))
    r = torch.where(i == 0, maximum, torch.where(i == 1, q, torch.where(i == 2, p,
        torch.where(i == 3, p, torch.where(i == 4, t, maximum)))))
    g = torch.where(i == 0, t, torch.where(i == 1, maximum, torch.where(i == 2, maximum,
        torch.where(i == 3, q, torch.where(i == 4, p, p)))))
    b = torch.where(i == 0, p, torch.where(i == 1, p, torch.where(i == 2, t,
        torch.where(i == 3, maximum, torch.where(i == 4, maximum, q)))))
    return torch.stack((r, g, b), dim=1)


class WholeBodyVisualAugmenter:
    """Transform a float RGB [T,3,H,W] clip with caller-owned CPU RNG.

    `apply` always returns new storage, including when disabled. Use a separate
    sample per camera, or reuse `sample_parameters` across cameras explicitly.
    Dataset/worker seeding must advance across calls or include the epoch.
    """

    def __init__(self, config=None):
        self.config = normalize_visual_config(config)
        self.camera_mode = self.config["camera_mode"]

    def sample_parameters(self, generator, *, apply_probability=True):
        if not isinstance(generator, torch.Generator) or generator.device.type != "cpu":
            raise ValueError("Visual augmentation requires a caller-owned CPU torch.Generator")
        result = {key: None for key in ("exposure", "overexposure", "color", "jitter", "blur")}
        if not self.config["enabled"]:
            return result
        if apply_probability and not _event(self.config.get("probability", 1.0), generator):
            return result
        for key in result:
            block = self.config[key]
            if not _event(block["probability"], generator):
                continue
            if key == "exposure":
                result[key] = {k: _uniform(block[k], generator) for k in ("gain", "gamma")}
            elif key == "overexposure":
                result[key] = {"gain": _uniform(block["gain"], generator)}
            elif key == "color":
                result[key] = {k: _uniform(block[k], generator) for k in ("contrast", "saturation")}
                result[key]["hue"] = _uniform((-block["hue"], block["hue"]), generator)
            elif key == "jitter":
                result[key] = {k: _uniform((-block["translation"], block["translation"]), generator)
                               for k in ("translation_x", "translation_y")}
                result[key].update(rotation_degrees=_uniform((-block["rotation_degrees"], block["rotation_degrees"]), generator),
                                   scale=_uniform(block["scale"], generator))
            else:
                result[key] = {"sigma": _uniform(block["sigma"], generator), "kernel_size": block["kernel_size"]}
        return result

    def apply(self, frames, *, generator, parameters=None, apply_probability=True):
        if (not torch.is_tensor(frames) or not frames.is_floating_point() or frames.ndim != 4
                or frames.shape[0] < 1 or frames.shape[1] != 3 or min(frames.shape[-2:]) < 2
                or not bool(torch.isfinite(frames).all())
                or bool((frames < 0).any()) or bool((frames > 1).any())):
            raise ValueError("Paired visual augmentation expects finite float RGB [T,3,H,W] in[0,1]")
        if not self.config["enabled"]:
            return frames.clone()
        params = (self.sample_parameters(generator, apply_probability=apply_probability)
                  if parameters is None else _plain(parameters))
        if not isinstance(params, dict) or set(params) != {"exposure", "overexposure", "color", "jitter", "blur"}:
            raise ValueError("Explicit paired visual parameters have an invalid schema")
        if all(value is None for value in params.values()):
            return frames.clone()
        image = frames.float().clone()
        exposure = params["exposure"]
        if exposure is not None:
            image = (image * exposure["gain"]).clamp(0, 1).pow(exposure["gamma"])
        overexposure = params["overexposure"]
        if overexposure is not None:
            image = (image * overexposure["gain"]).clamp(0, 1)
        color = params["color"]
        if color is not None:
            gray = (image * image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)).sum(dim=1, keepdim=True)
            # Each frame's own mean prevents future-image leakage into the current prefix.
            center = gray.mean(dim=(-2, -1), keepdim=True)
            image = (center + color["contrast"] * (image - center)).clamp(0, 1)
            gray = (image * image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)).sum(dim=1, keepdim=True)
            image = (gray + color["saturation"] * (image - gray)).clamp(0, 1)
            image = _hue(image, color["hue"]).clamp(0, 1)
        jitter = params["jitter"]
        if jitter is not None and (jitter["rotation_degrees"] != 0 or jitter["scale"] != 1
                                   or jitter["translation_x"] != 0 or jitter["translation_y"] != 0):
            height, width = image.shape[-2:]
            angle = math.radians(jitter["rotation_degrees"])
            c, s = math.cos(angle) / jitter["scale"], math.sin(angle) / jitter["scale"]
            theta = image.new_tensor([[c, -s * height / width, 2 * jitter["translation_x"]],
                                      [s * width / height, c, 2 * jitter["translation_y"]]])
            grid = F.affine_grid(theta[None].expand(len(image), -1, -1), image.shape, align_corners=False)
            image = F.grid_sample(image, grid, mode="bilinear", padding_mode="border", align_corners=False)
        blur = params["blur"]
        if blur is not None:
            size, sigma = blur["kernel_size"], blur["sigma"]
            coords = torch.arange(size, dtype=image.dtype, device=image.device) - size // 2
            kernel = torch.exp(-0.5 * (coords / sigma).square())
            kernel = kernel / kernel.sum()
            weights = (kernel[:, None] * kernel[None, :])[None, None].expand(3, 1, size, size)
            image = F.conv2d(F.pad(image, (size//2,)*4, mode="replicate"), weights, groups=3)
        return image.clamp(0, 1).to(dtype=frames.dtype)
