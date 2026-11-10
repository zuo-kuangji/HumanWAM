"""ACT (Psi0 baselines/act = LeRobot ACT, ResNet18) launch config for the Humanoid Benchmark tasks.

Registered by hb_act_train.py as psi.config.train.hb_act_config; the pinned upstream scripts/train.py loads it
unchanged and Trainer.instantiate picks psi.trainers.act_g1.ActG1Trainer (--train.name=act-g1).
Native recipe = baselines/act/train_act_g1_simple.sh (G1 simulation): chunk 16, VAE (kl 10, latent 32), dim 512,
4 encoder / 1 decoder layers, ResNet18 ImageNet backbone, images resized to 256x480 (NEAREST) and center-cropped to
224x224, ImageNet normalisation, bounds-normalised actions and states, AdamW lr 1e-4 (betas 0.95/0.999, wd 1e-6),
cosine schedule, warmup 2.5 % (1000/40000), bf16 DDP.
HB changes (CLAUDE.md v2, identical inputs for every model):
* repack = hb_psi0_config.HbSonicRepackTransform (the same Psi0 view as Psi-Zero: G1 observation.hb44 /
  action.hb76, H2 observation.hb78 / action.hb75, cameras cam0 + chest, raw state WITHOUT joint velocity),
  with the HB v2 proprio augmentation (15 % delay 1..2 frames, +-2 deg per-trajectory bias, 0.25 deg noise).
* field  = native SonicActionStateTransform (bounds), as Psi-Zero (same stats_psi0.json of the view).
* model  = HbACTModelTransform: native geometry (resize 256x480 NEAREST, center crop 224x224) and normalisation;
  native ColorJitter (img_aug) OFF, replaced by hb_aug_v2.augment on both cameras jointly at 224x224.
  ACT natively encodes every camera with the shared ResNet18 and concatenates the feature maps (n_cameras = 2).
"""
import os
import sys
from typing import Any

import numpy as np
import torch
from pydantic import model_validator
from torchvision.transforms import v2
from typing_extensions import Self

_HERE = os.path.dirname(os.path.abspath(__file__))
_PSI0 = os.path.join(os.path.dirname(_HERE), "hb_psi0")
if _PSI0 not in sys.path:
    sys.path.insert(0, _PSI0)
import hb_psi0_config as HP  # noqa: E402  HbSonicRepackTransform, hb_rng, set_epoch, A = hb_aug_v2

from psi.config import transform as pt  # noqa: E402
from psi.config import transform_psi0_sonic as ps  # noqa: E402
from psi.config.config import LaunchConfig  # noqa: E402
from psi.config.data_lerobot import LerobotDataConfig  # noqa: E402
from psi.config.model_act import ACTModelConfig  # noqa: E402
from psi.config.transform import DataTransform  # noqa: E402

RAW_STATE_DIMS = (44, 78)  # G1 / H2 benchmark state input (no joint velocity)


class HbACTModelTransform(pt.ACTModelTransform):
    hb_aug: bool = True

    def __call__(self, data: dict[str, Any], no_aug: bool = False, **kwargs) -> dict[str, Any]:
        assert not self.img_aug, "ACT native ColorJitter must stay off (HB v2 replaces it)"
        geom = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True), self.resize(), self.center_crop()])
        x = torch.stack([geom(img) for img in data["observations"]])  # (V, 3, 224, 224) in [0, 1], native geometry
        if self.hb_aug and not no_aug:
            y, p = HP.A.augment(x[None], HP.hb_rng())  # [T=1, V, 3, H, W]: colour shared, geometry/light per camera
            x = y[0].clamp(0.0, 1.0)
            HP._log_aug(p, y.shape)
        images = self.normalize()(x)
        mask = np.asarray(data["actions_mask"]).astype(bool)
        if not mask.all():
            raise ValueError("HB views never pad actions (actions_mask must be all True)")
        return {
            "observation.images": images,  # (n_cameras, C, H, W)
            "observation.state": data["states"],  # (1, Ds)
            "action": data["actions"],  # (chunk, Da)
            "action_is_pad": np.zeros(np.asarray(data["actions"]).shape[0], dtype=bool),
        }


class HbACTDataTransform(DataTransform):
    repack: HP.HbSonicRepackTransform
    field: ps.SonicActionStateTransform
    model: HbACTModelTransform


class HbACTDataConfig(LerobotDataConfig):
    transform: HbACTDataTransform  # must be re-annotated for tyro (see hb_psi0_config.HbDataConfig)
    num_workers: int | None = None
    prefetch_factor: int | None = None


class DynamicLaunchConfig(LaunchConfig):
    data: HbACTDataConfig
    model: ACTModelConfig

    @model_validator(mode="after")
    def check_hb_dims(self, __context: Any) -> Self:
        repack, field, model = self.data.transform.repack, self.data.transform.field, self.model
        assert repack.pad_action_dim == field.pad_action_dim == model.action_dim, \
            (repack.pad_action_dim, field.pad_action_dim, model.action_dim)
        assert repack.pad_state_dim == field.pad_state_dim == model.state_dim, \
            (repack.pad_state_dim, field.pad_state_dim, model.state_dim)
        assert model.state_dim in RAW_STATE_DIMS, f"ACT state must be the raw 44 / 78 dims (no velocity), got {model.state_dim}"
        assert model.chunk_size == model.n_action_steps == repack.action_chunk_size, \
            (model.chunk_size, model.n_action_steps, repack.action_chunk_size)
        assert field.action_min is not None and len(field.action_min) == model.action_dim, "action stats width"
        assert field.normalize_state and field.state_min is not None and len(field.state_min) == model.state_dim, \
            "state stats width"
        assert not self.data.transform.model.img_aug, "disable --data.transform.model.img-aug (HB v2 replaces it)"
        assert field.state_noise_std == 0.0 and repack.state_temporal_jitter == 0
        assert 0 <= repack.hb_body_dims <= model.state_dim
        return self
