"""Opt-in shared ImageWAM data path, without modifying legacy v2/v3 recipes."""

import copy
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from test_wholebody_training_augmentation import synthetic_dataset

from imagewam.augmentation_config import (
    SCHEMA,
    UNIFIED_SCHEMA,
    UNIFIED_V2_SCHEMA,
    UNIFIED_V3_SCHEMA,
    augmentation_sha256,
    default_augmentation,
    normalize_augmentation,
)
from imagewam.datasets.unified_augmentation import UnifiedTrainingAugmentation

ROOT = Path(__file__).resolve().parents[1]
PROFILE = {"schema": UNIFIED_SCHEMA, "enabled": True, "seed": 42}
PROFILE_V2 = {"schema": UNIFIED_V2_SCHEMA, "enabled": True, "seed": 42}
PROFILE_V3 = {"schema": UNIFIED_V3_SCHEMA, "enabled": True, "seed": 42}
PROFILES = (PROFILE, PROFILE_V2, PROFILE_V3)
VERSIONED_PROFILES = tuple(enumerate(PROFILES, start=1))


class ScaledNormalizer:
    def forward(self, values, field):
        # A non-identity body scale distinguishes noise in raw radians from
        # accidentally adding the same numbers after normalization.
        result = np.array(values, dtype=np.float32, copy=True)
        if field == "state":
            result[..., :29] = result[..., :29] * 7 + 0.25
            result[..., 29:41] = result[..., 29:41] * 2 - 1
        return result


@pytest.mark.parametrize("version,canonical", VERSIONED_PROFILES)
def test_profile_is_explicit_canonical_and_does_not_change_default(version, canonical):
    from omegaconf import OmegaConf

    profile = OmegaConf.load(ROOT / f"configs/wholebody/augmentation_unified_v{version}.yaml")
    assert normalize_augmentation(profile) == canonical
    assert normalize_augmentation({"schema": canonical["schema"]}) == canonical
    assert normalize_augmentation({}) == default_augmentation()
    assert default_augmentation()["schema"] == SCHEMA
    assert augmentation_sha256(PROFILE) != augmentation_sha256(default_augmentation())
    for extra in ({"visual": {}}, {"proprio": {}}, {"state_dropout": 0.1}, {"seed": True}):
        with pytest.raises(ValueError):
            normalize_augmentation({**canonical, **extra})


def test_historical_unified_hashes_are_not_reinterpreted_as_v3():
    assert augmentation_sha256(PROFILE) == "7d8dc40ae425150cf6fe53a56fdeb27bdfa874fd0131b91a21ede7e49bdeb5ed"
    assert augmentation_sha256({**PROFILE, "enabled": False}) == "678a9b0371bcfee19436d3e0106a7e798a53af6148a4507dcc9e7c16559b2916"
    assert augmentation_sha256(PROFILE_V2) == "68982760eef15fc6aaeb243eb580d60f4b2219f7164fd8bd321ac456c93efcb2"
    assert augmentation_sha256({**PROFILE_V2, "enabled": False}) == "a05ac720910ce5f6273c643ba488b65fbe069af76dc09f1d418160bf67e0c460"
    assert len({augmentation_sha256(profile) for profile in PROFILES}) == len(PROFILES)


@pytest.mark.parametrize("schema", [[], {}, None])
def test_non_string_schema_still_raises_value_error(schema):
    with pytest.raises(ValueError, match="schema"):
        normalize_augmentation({"schema": schema})


@pytest.mark.parametrize("route", ["motion", "joint"])
@pytest.mark.parametrize("profile", PROFILES)
def test_both_routes_apply_one_complete_clip_and_raw_state_before_normalization(route, profile, monkeypatch):
    import humanwam.augmentation as shared

    calls = []

    class RecordingEngine:
        profile = UNIFIED_SCHEMA

        def __init__(self, seed, profile):
            assert seed == 42
            self.profile = profile

        def apply_rgb(self, clip, *, sample, observation_mask):
            calls.append(("rgb", sample, observation_mask.clone(), clip.clone()))
            output = clip.clone()
            output[0] += 0.1  # Stand-in observation-only sensor degradation.
            return output

        def apply_state(self, state, *, body_dims, sample):
            calls.append(("state", sample, body_dims, state.copy()))
            result = state.copy()
            result[:body_dims] += 0.02
            return result

    monkeypatch.setattr(shared, "UnifiedAugmentation", RecordingEngine)
    dataset = synthetic_dataset(route, profile)
    dataset.normalizer = ScaledNormalizer()
    original = copy.deepcopy(dataset.episodes)
    baseline = synthetic_dataset(route, None)
    baseline.normalizer = ScaledNormalizer()
    plain = baseline[42]
    result = dataset[42]
    rgb, state = calls
    assert rgb[0] == "rgb" and state[0] == "state"
    assert rgb[1] == shared.SampleKey(episode=0, frame=42, epoch=0) == state[1]
    assert rgb[3].shape == (2, 2, 3, 224, 224)
    torch.testing.assert_close(rgb[2], torch.tensor([True, False]))
    assert state[2] == 29
    np.testing.assert_array_equal(state[3], original[0][0][42])
    torch.testing.assert_close(result["proprio"][:, :29], plain["proprio"][:, :29] + 0.14)
    torch.testing.assert_close(result["proprio"][:, 29:], plain["proprio"][:, 29:], rtol=0, atol=0)
    torch.testing.assert_close(result["video"][:, 0], plain["video"][:, 0] + 0.2)
    torch.testing.assert_close(result["video"][:, 1], plain["video"][:, 1], rtol=0, atol=0)
    assert result["state_condition_keep"].item() is True
    for name in ("action", "past_action", "action_is_pad", "past_action_is_pad", "image_is_pad"):
        torch.testing.assert_close(result[name], plain[name], rtol=0, atol=0)
    for actual, expected in zip(dataset.episodes[0], original[0]):
        np.testing.assert_array_equal(actual, expected)
    for (_, _, frame), image in dataset.frame_cache.items():
        torch.testing.assert_close(image, torch.full_like(image, 0.2 + frame / 500), rtol=0, atol=0)


@pytest.mark.parametrize("route", ["motion", "joint"])
@pytest.mark.parametrize("profile", PROFILES)
def test_real_engine_is_keyed_by_sample_epoch_not_iteration_order_and_keeps_labels_clean(route, profile):
    dataset = synthetic_dataset(route, profile)
    assert dataset.augmentation._engine.profile == profile["schema"]
    baseline = synthetic_dataset(route, None)[42]
    first = dataset[42]
    dataset[18]
    repeated = dataset[42]
    torch.testing.assert_close(first["video"], repeated["video"], rtol=0, atol=0)
    torch.testing.assert_close(first["proprio"], repeated["proprio"], rtol=0, atol=0)
    dataset.set_epoch(1)
    another_epoch = dataset[42]
    assert not torch.equal(first["proprio"], another_epoch["proprio"])
    assert not torch.equal(first["video"], another_epoch["video"])
    for item in (first, another_epoch):
        assert item["state_condition_keep"].item() is True
        torch.testing.assert_close(item["proprio"], item["proprio"][:1].expand_as(item["proprio"]), rtol=0, atol=0)
        torch.testing.assert_close(item["proprio"][:, 29:], baseline["proprio"][:, 29:], rtol=0, atol=0)
        for field in ("action", "past_action"):
            torch.testing.assert_close(item[field], baseline[field], rtol=0, atol=0)


@pytest.mark.parametrize("route", ["motion", "joint"])
@pytest.mark.parametrize("profile", PROFILES)
def test_changing_future_rgb_does_not_change_current_augmented_observation(route, profile):
    original = synthetic_dataset(route, profile)
    changed = synthetic_dataset(route, profile)
    anchor = 42
    decode = changed.decoder

    def replaced(path, indices):
        return {frame: pixels if frame == anchor else torch.ones_like(pixels)
                for frame, pixels in decode(path, indices).items()}

    changed.decoder = replaced
    first, second = original[anchor], changed[anchor]
    torch.testing.assert_close(first["video"][:, 0], second["video"][:, 0], rtol=0, atol=0)
    torch.testing.assert_close(first["proprio"], second["proprio"], rtol=0, atol=0)


def test_profile_import_is_optional_for_legacy_disabled_and_cpu_config_validation():
    code = """
import importlib.abc
import sys
class RejectHumanWAM(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'humanwam' or fullname.startswith('humanwam.'):
            raise AssertionError('Legacy path imported shared augmentation')
sys.meta_path.insert(0, RejectHumanWAM())
from imagewam.augmentation_config import normalize_augmentation
assert normalize_augmentation({'schema':'humanwam_unified_augmentation_v1'})['seed'] == 42
assert normalize_augmentation({'schema':'humanwam_unified_augmentation_v3'})['seed'] == 42
from imagewam.datasets.wholebody import WholeBodyLeRobotDataset
for config in (None, {'schema':'wholebody_training_augmentation_v2'},
               {'schema':'wholebody_training_augmentation_v3'},
               {'schema':'humanwam_unified_augmentation_v1','enabled':False},
               {'schema':'humanwam_unified_augmentation_v2','enabled':False},
               {'schema':'humanwam_unified_augmentation_v3','enabled':False}):
    dataset = object.__new__(WholeBodyLeRobotDataset)
    dataset._configure_augmentation(config, split='train')
assert 'humanwam' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True,
                   env={**os.environ, "PYTHONPATH": str(ROOT / "src")})


@pytest.mark.parametrize("profile", PROFILES)
def test_unified_profile_is_training_only_and_explicitly_disabled_is_noop(profile):
    with pytest.raises(ValueError, match="training"):
        UnifiedTrainingAugmentation(profile, training=False)
    augmentation = UnifiedTrainingAugmentation({**profile, "enabled": False}, training=False)
    images, state = torch.ones(2, 2, 3, 4, 4), np.zeros(44, np.float32)
    actual, raw = augmentation.apply_observation(images, state, episode=0, frame=0)
    torch.testing.assert_close(images, actual, rtol=0, atol=0)
    np.testing.assert_array_equal(state, raw)
    assert images.data_ptr() != actual.data_ptr() and not np.shares_memory(state, raw)


@pytest.mark.parametrize("profile", PROFILES)
def test_adapter_matches_explicit_versioned_engine_instead_of_new_default(profile):
    from humanwam.augmentation import SampleKey, UnifiedAugmentation

    frames = torch.linspace(0, 1, 2 * 2 * 3 * 8 * 8).reshape(2, 2, 3, 8, 8)
    state = np.zeros(44, dtype=np.float32)
    state[29:41] = 0.5
    state[43] = -1
    adapter = UnifiedTrainingAugmentation(profile)
    adapter.set_epoch(3)
    actual_rgb, actual_state = adapter.apply_observation(frames, state, episode=4, frame=42)
    # Historical references explicitly select their engine rather than silently
    # following UnifiedAugmentation's newest default profile.
    reference = UnifiedAugmentation(seed=profile["seed"], profile=profile["schema"])
    sample = SampleKey(episode=4, frame=42, epoch=3)
    expected_rgb = reference.apply_rgb(
        frames, sample=sample, observation_mask=torch.tensor([True, False]),
    )
    expected_state = reference.apply_state(state, body_dims=29, sample=sample)
    torch.testing.assert_close(actual_rgb, expected_rgb, rtol=0, atol=0)
    np.testing.assert_array_equal(actual_state, expected_state)


@pytest.mark.parametrize("version,canonical", VERSIONED_PROFILES)
def test_builder_and_training_contract_accept_unified_without_state_gate(tmp_path, version, canonical):
    from test_wholebody_training_cli import builder, fixture_config, trainer

    args = type("Args", (), {"augmentation_config": ROOT / f"configs/wholebody/augmentation_unified_v{version}.yaml",
                             "no_augmentation": False})()
    actual, digest, source = builder.resolve_training_augmentation(args)
    assert actual == canonical and source.endswith(f"augmentation_unified_v{version}.yaml")
    cfg = fixture_config(tmp_path)
    cfg["data"]["train"]["augmentation"] = canonical
    cfg["wholebody_contract"].update(training_augmentation=canonical, training_augmentation_sha256=digest)
    assert trainer.training_augmentation_contract(cfg)["config"] == canonical


def test_legacy_and_disabled_paths_do_not_import_the_shared_package():
    code = """
import importlib.abc, sys
class RejectShared(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'humanwam' or fullname.startswith('humanwam.'):
            raise AssertionError('Shared dependency imported outside opt-in: ' + fullname)
        return None
sys.meta_path.insert(0, RejectShared())
from imagewam.augmentation_config import default_augmentation, UNIFIED_SCHEMAS
from imagewam.datasets.wholebody import WholeBodyLeRobotDataset
for profile in (None, default_augmentation(),
                *({'schema': schema, 'enabled': False} for schema in UNIFIED_SCHEMAS)):
    dataset = object.__new__(WholeBodyLeRobotDataset)
    dataset._configure_augmentation(profile, split='train')
assert not any(k == 'humanwam' or k.startswith('humanwam.') for k in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True,
                   env={**os.environ, "PYTHONPATH": str(ROOT / "src")})


class UnifiedWorkerProbe(torch.utils.data.Dataset):
    def __init__(self, profile):
        self.augmentation = UnifiedTrainingAugmentation(profile)

    def __len__(self):
        return 2

    def __getitem__(self, index):
        _, raw = self.augmentation.apply_observation(
            torch.full((2, 2, 3, 8, 8), 0.3), np.zeros(44, np.float32),
            episode=3, frame=index,
        )
        return torch.from_numpy(raw)


@pytest.mark.parametrize("profile", PROFILES)
def test_sample_identity_survives_worker_count_and_epoch_reaches_persistent_workers(profile):
    local = UnifiedWorkerProbe(profile)
    expected0 = torch.stack([local[i] for i in range(2)])
    local.augmentation.set_epoch(1)
    expected1 = torch.stack([local[i] for i in range(2)])
    worker_dataset = UnifiedWorkerProbe(profile)
    loader = torch.utils.data.DataLoader(worker_dataset, batch_size=2, num_workers=1,
                                        persistent_workers=True)
    try:
        actual0 = next(iter(loader))
        worker_dataset.augmentation.set_epoch(1)
        actual1 = next(iter(loader))
    finally:
        if loader._iterator is not None:
            loader._iterator._shutdown_workers()
    torch.testing.assert_close(actual0, expected0, rtol=0, atol=0)
    torch.testing.assert_close(actual1, expected1, rtol=0, atol=0)
    assert not torch.equal(actual0, actual1)
