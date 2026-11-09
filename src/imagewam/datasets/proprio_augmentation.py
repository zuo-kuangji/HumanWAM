"""Training-only current-state perturbations and worker-local augmentation RNG.

The measured targets, action history, dataset statistics and inference are never
modified. Gravity is rotated as one physical direction before normalization.
"""

from __future__ import annotations

import hashlib
import math
import os

import numpy as np
import torch

from imagewam.augmentation_config import (
    SCHEMA,
    normalize_augmentation,
    normalize_proprio_config,
)


class State44Augmenter:
    """Perturb only the current normalized body29/hand12 observation.

    Noise and per-sample bias use normalized coordinates and one sample event.
    Dropout never alters this raw state; the model masks its encoded condition.
    The same perturbed
    observation is subsequently repeated across the proprio time axis, so that
    axis never becomes a synthetic future-state sequence.
    """

    def __init__(self, config=None):
        self.config = normalize_proprio_config(
            {"enabled": False} if config is None else config
        )
        self.enabled = self.config["enabled"]
        self.values = {
            key: value
            for key, value in self.config.items()
            if key not in {"enabled", "state_dropout", "noise_probability"}
        }
        self.noise_probability = self.config["noise_probability"]

    def apply(
        self, normalized_current_state, *, raw_current_state, normalizer, generator
    ):
        value = np.asarray(normalized_current_state)
        if (
            value.shape != (44,)
            or value.dtype.kind not in "fiu"
            or not np.isfinite(value).all()
        ):
            raise ValueError(
                "Current proprio augmentation requires a finite normalized state44"
            )
        result = value.astype(np.float32, copy=True)
        if not self.enabled:
            return result
        if not isinstance(generator, torch.Generator) or generator.device.type != "cpu":
            raise ValueError("Use the dataset's explicit CPU augmentation generator")
        probability = self.noise_probability
        if probability == 0 or (
            probability < 1 and float(torch.rand((), generator=generator)) >= probability
        ):
            return result
        for region, start, end, clip in (("body", 0, 29, 5.0), ("hand", 29, 41, 1.0)):
            for kind in ("noise", "bias"):
                std = self.values[f"{region}_{kind}_std"]
                if std:
                    result[start:end] += (
                        torch.randn(end - start, generator=generator).numpy() * std
                    )
            result[start:end] = np.clip(result[start:end], -clip, clip)
        degrees = self.values["gravity_rotation_std_deg"]
        if degrees:
            raw = np.asarray(raw_current_state, dtype=np.float32)
            if raw.shape != (44,) or not np.isfinite(raw).all():
                raise ValueError(
                    "Gravity rotation requires the true raw current state44"
                )
            gravity = raw[41:44].astype(np.float64)
            norm = np.linalg.norm(gravity)
            if norm > 1e-8:
                axis = torch.randn(3, generator=generator, dtype=torch.float64).numpy()
                axis_norm = np.linalg.norm(axis)
                if axis_norm > 1e-12:
                    axis /= axis_norm
                    # Bound rare Gaussian tails; there is no independent XYZ noise.
                    angle = float(torch.randn((), generator=generator))
                    angle = np.clip(angle, -3.0, 3.0) * math.radians(degrees)
                    rotated = (
                        gravity * math.cos(angle)
                        + np.cross(axis, gravity) * math.sin(angle)
                        + axis * np.dot(axis, gravity) * (1 - math.cos(angle))
                    )
                    rotated *= norm / np.linalg.norm(rotated)
                    raw_copy = raw.copy()
                    raw_copy[41:44] = rotated
                    normalized = normalizer.forward(raw_copy, "state")
                    # A quantile clamp can distort a physical direction. Skip
                    # that optional rotation rather than inject a non-unit vector.
                    restored = normalizer.backward(normalized, "state")[41:44]
                    if np.allclose(restored, rotated, rtol=1e-5, atol=1e-6):
                        result[41:44] = normalized[41:44]
        if not np.isfinite(result).all():
            raise ValueError("Current-state augmentation produced invalid values")
        return result


class TrainingAugmentation:
    """Share epoch with persistent workers; advance a private generator per sample."""

    def __init__(self, config=None, *, training=True):
        self.config = normalize_augmentation(config)
        self.enabled = self.config["enabled"]
        if self.enabled and self.config["schema"] != SCHEMA:
            raise ValueError("Legacy augmentation v1 is provenance-only; use the v2 training profile")
        if self.enabled and training is not True:
            raise ValueError("Augmentation may only be enabled on the training dataset")
        self.seed = self.config["seed"]
        self.proprio = State44Augmenter(self.config["proprio"] if self.enabled else None)
        self.visual = None
        visual = self.config["visual"]
        if self.enabled and visual["enabled"]:
            from imagewam.datasets.wholebody_augmentation import (
                WholeBodyVisualAugmenter,
            )

            self.visual = WholeBodyVisualAugmenter(visual)
        self._epoch = torch.zeros(1, dtype=torch.int64)
        if self.enabled:
            self._epoch.share_memory_()
        self._generator = None
        self._generator_key = None

    def set_epoch(self, epoch):
        if type(epoch) is not int or epoch < 0:
            raise ValueError("Augmentation epoch must be a nonnegative integer")
        self._epoch[0] = epoch

    def generator(self):
        if not self.enabled:
            return None
        worker = torch.utils.data.get_worker_info()
        worker_id = -1 if worker is None else worker.id
        rank = os.environ.get("RANK", "0")
        key = (
            self.seed,
            int(torch.initial_seed()),
            rank,
            worker_id,
            int(self._epoch[0]),
        )
        if self._generator_key != key:
            digest = hashlib.sha256(repr(key).encode()).digest()
            self._generator = torch.Generator(device="cpu")
            self._generator.manual_seed(int.from_bytes(digest[:8], "little") % (2**63))
            self._generator_key = key
        return self._generator

    def visual_parameters(self, *, generator):
        if self.enabled and self.visual is not None:
            probability = self.config["visual"]["probability"]
            if probability == 0 or (
                probability < 1 and float(torch.rand((), generator=generator)) >= probability
            ):
                # One event controls both cameras; explicit no-op parameters
                # also prevent a second event draw during apply_video.
                return {key: None for key in ("exposure", "overexposure", "color", "jitter", "blur")}
            if self.visual.camera_mode == "shared":
                return self.visual.sample_parameters(generator, apply_probability=False)
        return None

    def sample_state_keep(self, generator):
        """Draw one condition keep flag, reused for every FM/IC path of a sample."""
        if not self.enabled or not self.proprio.enabled:
            return torch.tensor(True, dtype=torch.bool)
        probability = self.config["proprio"]["state_dropout"]["probability"]
        if probability == 0:
            return torch.tensor(True, dtype=torch.bool)
        if probability == 1:
            return torch.tensor(False, dtype=torch.bool)
        if not isinstance(generator, torch.Generator) or generator.device.type != "cpu":
            raise ValueError("Use the dataset's explicit CPU augmentation generator")
        return torch.rand((), generator=generator) >= probability

    def apply_video(self, frames, *, generator, parameters=None):
        # Explicit clone protects decoded/cache tensors even if the augmentation
        # implementation later changes to use in-place operations internally.
        copied = frames.clone()
        if self.enabled and self.visual is not None:
            return self.visual.apply(copied, generator=generator, parameters=parameters,
                                     apply_probability=False)
        return copied

    def apply_current_state(
        self, normalized_current_state, *, raw_current_state, normalizer, generator
    ):
        if not self.enabled:
            return np.asarray(normalized_current_state, dtype=np.float32).copy()
        return self.proprio.apply(
            normalized_current_state,
            raw_current_state=raw_current_state,
            normalizer=normalizer,
            generator=generator,
        )
