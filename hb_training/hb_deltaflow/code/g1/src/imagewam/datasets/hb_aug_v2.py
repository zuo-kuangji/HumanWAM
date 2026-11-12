"""Humanoid Benchmark data augmentation v2 (project CLAUDE.md), model-agnostic torch implementation.

augment(x, rng, obs_mask=None) takes x: float [T, V, 3, H, W] in [0, 1] (T frames, V cameras) and returns the
augmented tensor. Each call is one training sample: one of the 15 effects is chosen uniformly and its magnitude
is scaled by 1.5 around identity; the rest stay at x1. Colour is shared across cameras and time; geometry, light
patches and white balance are drawn per camera and fixed over time; sensor/optics effects (noise, gaussian blur,
JPEG, motion blur, sharpen, defocus) apply only to observation frames (obs_mask True).
Pixel-size parameters are defined at 224 px and scaled by H / 224.
perturb_joints(q, rng) applies the proprio spec: per-joint bias +-2 deg plus N(0, 0.25 deg) clipped at 3 sigma.
"""
import io
import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

EFFECTS = ("exposure", "contrast", "saturation", "hue", "gamma", "temperature", "rotation_crop", "noise",
           "gaussian_blur", "jpeg", "motion_blur", "sharpen", "defocus", "light_shadow", "wb_channel")
CENTER = {"exposure": 0.0, "contrast": 1.0, "saturation": 1.0, "hue": 0.0, "gamma": 1.0, "temperature": 0.0}
RANGE = {"exposure": (-0.22, 0.22), "contrast": (0.88, 1.12), "saturation": (0.76, 1.24), "hue": (-0.015, 0.015),
         "gamma": (0.89, 1.11), "temperature": (-0.066, 0.066)}
VERSION = "hb-aug-v2-20261109"


def sample(rng, views, height):
    boost = EFFECTS[int(rng.integers(len(EFFECTS)))]

    def k(name):
        return 1.5 if boost == name else 1.0

    px = height / 224.0
    p = {"boost": boost}
    for name, (a, b) in RANGE.items():
        c = CENTER[name]
        p[name] = float(rng.uniform(c + k(name) * (a - c), c + k(name) * (b - c)))
    p["geometry"] = []
    for _ in range(views):
        angle = float(rng.uniform(-1.5 * k("rotation_crop"), 1.5 * k("rotation_crop")))
        p["geometry"].append((angle, 1.0 + abs(angle) * 0.015 + float(rng.uniform(0, 0.02 * k("rotation_crop")))))
    p["noise"] = 4.0 / 255.0 * k("noise") if rng.random() < 0.5 else 0.0
    p["gaussian_blur"] = float(rng.uniform(0.25, 0.65)) * k("gaussian_blur") if rng.random() < 0.15 else 0.0
    if rng.random() < 0.35:
        lo = int(round(100 - k("jpeg") * 30))
        hi = int(round(100 - k("jpeg") * 5))
        p["jpeg"] = int(rng.integers(lo, hi + 1))
    else:
        p["jpeg"] = None
    p["motion_blur"] = ((float(rng.uniform(3, 7)) * k("motion_blur") * px, float(rng.uniform(0, 180)))
                        if rng.random() < 0.15 else None)
    p["sharpen"] = float(rng.uniform(0.2, 0.6)) * k("sharpen") if rng.random() < 0.15 else 0.0
    p["defocus"] = float(rng.uniform(1, 2)) * k("defocus") * px if rng.random() < 0.10 else 0.0
    if rng.random() < 0.30:
        g = 0.30 * k("light_shadow")
        p["light_shadow"] = [[(float(rng.uniform(0.1, 0.9)), float(rng.uniform(0.1, 0.9)),
                               float(rng.uniform(0.15, 0.35)), float(rng.uniform(1 - g, 1 + g)))
                              for _ in range(int(rng.integers(1, 3)))] for _ in range(views)]
    else:
        p["light_shadow"] = None
    if rng.random() < 0.30:
        g = 0.10 * k("wb_channel")
        gains = []
        for _ in range(views):
            row = [1.0, 1.0, 1.0]
            row[int(rng.integers(0, 3))] = float(rng.uniform(1 - g, 1 + g))
            gains.append(row)
        p["wb_channel"] = gains
    else:
        p["wb_channel"] = None
    return p


def _conv(x, kernel):
    n, c, h, w = x.shape
    k = kernel.to(x)[None, None].repeat(c, 1, 1, 1)
    pad = kernel.shape[-1] // 2
    return F.conv2d(F.pad(x, (pad, pad, pad, pad), mode="reflect"), k, groups=c)


def motion_kernel(length, angle_deg):
    size = max(3, int(round(length)) | 1)
    k = torch.zeros(size, size)
    c = size // 2
    th = math.radians(angle_deg)
    for s in np.linspace(-(length - 1) / 2, (length - 1) / 2, 8 * size):
        k[int(round(c - s * math.sin(th))), int(round(c + s * math.cos(th)))] = 1.0
    return k / k.sum()


def disk_kernel(r):
    size = 2 * int(math.ceil(r)) + 1
    c = size // 2
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    k = (((yy - c) ** 2 + (xx - c) ** 2) <= (r + 0.5) ** 2).float()
    return k / k.sum()


def apply(x, p, obs_mask=None, noise_seed=0):
    t, v, c, h, w = x.shape
    x = x.clone().clamp(0, 1)
    for vi, (angle, scale) in enumerate(p["geometry"]):
        x[:, vi] = TF.affine(x[:, vi], angle=angle, translate=[0, 0], scale=scale, shear=[0, 0],
                             interpolation=InterpolationMode.BILINEAR)
    if p["light_shadow"] is not None:
        yy, xx = torch.meshgrid(torch.linspace(0, 1, h), torch.linspace(0, 1, w), indexing="ij")
        for vi, blobs in enumerate(p["light_shadow"]):
            m = torch.ones(h, w)
            for cx, cy, s, g in blobs:
                m = m + (g - 1.0) * torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s))
            x[:, vi] = (x[:, vi] * m.clamp(min=0).to(x)).clamp(0, 1)
    if p["wb_channel"] is not None:
        for vi, g in enumerate(p["wb_channel"]):
            x[:, vi] = (x[:, vi] * torch.tensor(g, dtype=x.dtype).view(1, 3, 1, 1)).clamp(0, 1)
    y = x.reshape(t * v, c, h, w)
    y = (y * float(2.0 ** p["exposure"])).clamp(0, 1)
    d = p["temperature"]
    y = (y * y.new_tensor([1.0 + d, 1.0, 1.0 - d]).view(1, 3, 1, 1)).clamp(0, 1)
    y = ((y - 0.5) * p["contrast"] + 0.5).clamp(0, 1)
    y = TF.adjust_saturation(y, p["saturation"])
    y = TF.adjust_hue(y, p["hue"])
    y = y.clamp(0, 1).pow(p["gamma"])
    x = y.reshape(t, v, c, h, w)
    mask = torch.ones(t, dtype=torch.bool) if obs_mask is None else torch.as_tensor(obs_mask, dtype=torch.bool)
    if bool(mask.any()):
        o = x[mask].reshape(-1, c, h, w)
        if p["gaussian_blur"] > 0:
            o = TF.gaussian_blur(o, [3, 3], [p["gaussian_blur"]] * 2)
        if p["motion_blur"] is not None:
            o = _conv(o, motion_kernel(*p["motion_blur"]))
        if p["defocus"] > 0:
            o = _conv(o, disk_kernel(p["defocus"]))
        if p["sharpen"] > 0:
            o = o + p["sharpen"] * (o - TF.gaussian_blur(o, [5, 5], [1.0, 1.0]))
        o = o.clamp(0, 1)
        if p["noise"] > 0:
            gen = torch.Generator().manual_seed(int(noise_seed))
            o = (o + torch.randn(o.shape, generator=gen, dtype=o.dtype) * p["noise"]).clamp(0, 1)
        if p["jpeg"] is not None:
            for i in range(o.shape[0]):
                buf = io.BytesIO()
                TF.to_pil_image(o[i]).save(buf, format="JPEG", quality=int(p["jpeg"]))
                buf.seek(0)
                with Image.open(buf) as im:
                    o[i] = TF.pil_to_tensor(im.convert("RGB")).to(o.dtype).div(255.0)
        x[mask] = o.reshape(-1, v, c, h, w)
    return x.clamp(0, 1)


def augment(x, rng, obs_mask=None):
    p = sample(rng, x.shape[1], x.shape[-2])
    return apply(x, p, obs_mask, noise_seed=int(rng.integers(0, 2 ** 62))), p


def perturb_joints(q, rng):
    q = np.asarray(q, dtype=np.float64)
    d = q.shape[-1]
    bias = rng.uniform(-math.radians(2.0), math.radians(2.0), size=d)
    sigma = math.radians(0.25)
    noise = np.clip(rng.normal(0.0, sigma, size=q.shape), -3 * sigma, 3 * sigma)
    return (q + bias + noise).astype(np.float32)
