"""Opt-in adapter for HumanWAM's shared RGB/raw-state augmentation recipe."""

from __future__ import annotations

import numpy as np
import torch

from imagewam.augmentation_config import UNIFIED_SCHEMAS, normalize_augmentation


class UnifiedTrainingAugmentation:
    """Key one complete camera clip by episode/frame/epoch, before normalization.

    No model-level state dropout is implied by this data profile. The engine is
    imported only when the shared profile is explicitly enabled.
    """

    unified = True

    def __init__(self, config, *, training=True):
        self.config = normalize_augmentation(config)
        if self.config["schema"] not in UNIFIED_SCHEMAS:
            raise ValueError("Use the explicit HumanWAM unified augmentation profile")
        self.enabled = self.config["enabled"]
        if self.enabled and training is not True:
            raise ValueError("Augmentation may only be enabled on the training dataset")
        self._epoch = torch.zeros(1, dtype=torch.int64)
        self._engine = None
        self._sample_class = None
        if self.enabled:
            from humanwam.augmentation import SampleKey, UnifiedAugmentation

            self._engine = UnifiedAugmentation(seed=self.config["seed"], profile=self.config["schema"])
            if self._engine.profile != self.config["schema"]:
                raise ValueError("The installed shared augmentation engine has a different profile")
            self._sample_class = SampleKey
            self._epoch.share_memory_()

    def set_epoch(self, epoch):
        if type(epoch) is not int or epoch < 0:
            raise ValueError("Augmentation epoch must be a nonnegative integer")
        self._epoch[0] = epoch

    def apply_observation(self, frames, raw_state, *, episode, frame):
        # Copies protect the decoder/LRU tensors and the immutable numeric data.
        clip = frames.clone()
        state = np.array(raw_state, dtype=np.float32, copy=True)
        if not self.enabled:
            return clip, state
        sample = self._sample_class(episode=int(episode), frame=int(frame), epoch=int(self._epoch[0]))
        observation_mask = torch.zeros(len(clip), dtype=torch.bool)
        observation_mask[0] = True
        return (
            self._engine.apply_rgb(clip, sample=sample, observation_mask=observation_mask),
            self._engine.apply_state(state, body_dims=29, sample=sample),
        )
