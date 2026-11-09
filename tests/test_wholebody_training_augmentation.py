"""Synthetic CPU checks for training-only augmentation and immutable labels/caches."""

from collections import OrderedDict
from copy import deepcopy
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch

from imagewam.datasets.proprio_augmentation import (
    State44Augmenter,
    TrainingAugmentation,
)
from imagewam.datasets.wholebody import WholeBodyLeRobotDataset
from imagewam.datasets.wholebody_joint_reference import (
    JointReferenceLeRobotDataset,
    timing_configuration,
)


class SimpleNormalizer:
    def forward(self, values, field):
        result = np.asarray(values, dtype=np.float32).copy()
        if field == "state":
            result[..., 29:41] = result[..., 29:41] * 2 - 1
        return result

    def backward(self, values, field):
        result = np.asarray(values, dtype=np.float32).copy()
        if field == "state":
            result[..., 29:41] = (result[..., 29:41] + 1) / 2
        return result


def raw_state():
    state = np.zeros(44, dtype=np.float32)
    state[29:41] = 0.5
    state[43] = -1
    return state


def enabled_config(*, proprio=True, visual=True):
    return {
        "enabled": True,
        "seed": 76,
        "proprio": {"enabled": proprio},
        "visual": {
            "enabled": visual,
            "exposure": {"probability": 1.0, "gain": [0.8, 1.3]},
            "color": {"probability": 0},
            "jitter": {"probability": 0},
            "blur": {"probability": 0},
        },
    }


def synthetic_dataset(route, config):
    cls = WholeBodyLeRobotDataset if route == "motion" else JointReferenceLeRobotDataset
    ds = object.__new__(cls)
    ds._configure_augmentation(config, split="train")
    ds.root = __import__("pathlib").Path("/unused-synthetic-source")
    ds.normalizer = SimpleNormalizer()
    ds.episode_ids, ds.ends = [0], np.array([120])
    states = np.repeat(raw_state()[None], 120, axis=0)
    states[:, 0] = np.arange(120) / 100
    dimension = 76 if route == "motion" else 47
    actions = np.repeat(
        np.arange(120, dtype=np.float32)[:, None] / 200, dimension, axis=1
    )
    if route == "motion":
        actions[:, 64:] = states[:, 29:41]
        ds.action_horizon = ds.past_action_size = 16
        ds.episodes = {0: (states, actions)}
    else:
        actions[:, 29:41] = states[:, 29:41]
        actions[:, 41:47] = [1, 0, 0, 0, 1, 0]
        roots = np.repeat(np.array([[1, 0, 0, 0]], dtype=np.float32), 120, axis=0)
        ds.action_horizon, ds.past_action_size = 40, 16
        ds.action_dim, ds.recipe_version = 47, "v2"
        ds.timing = timing_configuration(
            40, 16, 2, 32, recipe_version="v2", execution_horizon=16
        )
        ds.episodes = {0: (states, actions, np.ones(120, dtype=bool), roots, roots)}
    ds.prompt = "Synthetic task"
    ds.text_hidden_states, ds.text_attention_mask = (
        torch.ones(1),
        torch.ones(1, dtype=torch.bool),
    )
    ds.video_cache_frames, ds.frame_cache = 32, OrderedDict()
    ds.decoder = lambda path, indices: {
        i: torch.full((3, 224, 224), 0.2 + i / 500) for i in indices
    }
    return ds


def test_current_state_noise_is_reproducible_bounded_and_preserves_physical_gravity_norm():
    normalizer = SimpleNormalizer()
    raw = raw_state()
    normalized = normalizer.forward(raw, "state")
    saved = normalized.copy()
    state_aug = State44Augmenter({"enabled": True, "gravity_rotation_std_deg": 1.0})
    first = state_aug.apply(
        normalized,
        raw_current_state=raw,
        normalizer=normalizer,
        generator=torch.Generator().manual_seed(2),
    )
    same = state_aug.apply(
        normalized,
        raw_current_state=raw,
        normalizer=normalizer,
        generator=torch.Generator().manual_seed(2),
    )
    np.testing.assert_array_equal(first, same)
    np.testing.assert_array_equal(normalized, saved)
    np.testing.assert_array_equal(raw, raw_state())
    assert not np.array_equal(first[:29], normalized[:29])
    assert not np.array_equal(first[29:41], normalized[29:41])
    assert np.all(np.abs(first[:29]) <= 5) and np.all(np.abs(first[29:41]) <= 1)
    restored = normalizer.backward(first, "state")[41:44]
    np.testing.assert_allclose(np.linalg.norm(restored), 1, atol=1e-7)
    assert not np.array_equal(restored, raw[41:44])


def test_disabled_augmentation_is_an_exact_copy_and_does_not_consume_rng():
    normalizer = SimpleNormalizer()
    value = normalizer.forward(raw_state(), "state")
    generator = torch.Generator().manual_seed(4)
    before = generator.get_state().clone()
    result = State44Augmenter({"enabled": False}).apply(
        value, raw_current_state=None, normalizer=normalizer, generator=generator
    )
    np.testing.assert_array_equal(result, value)
    assert not np.shares_memory(result, value)
    torch.testing.assert_close(before, generator.get_state())
    disabled = TrainingAugmentation()
    assert disabled.generator() is None
    frames = torch.rand(2, 3, 4, 4)
    copied = disabled.apply_video(frames, generator=None)
    torch.testing.assert_close(frames, copied)
    assert frames.data_ptr() != copied.data_ptr()


@pytest.mark.parametrize("dimension_probability", [0.05, 1.0])
def test_sample_dropout_hides_one_to_three_dimensions_without_touching_gravity_or_scaling_survivors(
    dimension_probability,
):
    normalizer = SimpleNormalizer()
    raw = raw_state()
    raw[:29], raw[29:41] = 2.0, 0.9
    value = normalizer.forward(raw, "state")
    augment = State44Augmenter(
        {
            "enabled": True,
            "body_noise_std": 0.0,
            "hand_noise_std": 0.0,
            "state_dropout": {
                "probability": 1.0,
                "dimension_probability": dimension_probability,
                "max_dropped_dimensions": 3,
            },
        }
    )
    result = augment.apply(
        value,
        raw_current_state=raw,
        normalizer=normalizer,
        generator=torch.Generator().manual_seed(9),
    )
    dropped = result[:41] == 0
    assert 1 <= int(dropped.sum()) <= 3
    np.testing.assert_array_equal(result[:41][~dropped], value[:41][~dropped])
    np.testing.assert_array_equal(result[41:], value[41:])
    # Body2.0 survives unchanged, rather than being clipped to the hand bounds.
    assert np.any(result[:29] == 2.0)
    np.testing.assert_array_equal(value[:29], 2.0)


def test_default_proprio_recipe_does_not_rotate_gravity_and_dropout_can_be_disabled():
    normalizer = SimpleNormalizer()
    raw = raw_state()
    value = normalizer.forward(raw, "state")
    augment = State44Augmenter({"enabled": True, "state_dropout": {"probability": 0.0}})
    result = augment.apply(
        value,
        raw_current_state=raw,
        normalizer=normalizer,
        generator=torch.Generator().manual_seed(6),
    )
    np.testing.assert_array_equal(result[41:], value[41:])
    assert augment.values["body_bias_std"] == 0
    assert augment.values["hand_bias_std"] == 0
    assert augment.values["hand_noise_std"] == 0.005


def test_zero_dimension_probability_disables_mask_without_forced_fallback_or_rng_draw():
    normalizer = SimpleNormalizer()
    value = np.full(44, 0.25, dtype=np.float32)
    raw = normalizer.backward(value, "state")
    augment = State44Augmenter({
        "enabled": True, "body_noise_std": 0, "hand_noise_std": 0,
        "state_dropout": {"probability": 1, "dimension_probability": 0},
    })
    generator = torch.Generator().manual_seed(29)
    before = generator.get_state().clone()
    result = augment.apply(value, raw_current_state=raw, normalizer=normalizer, generator=generator)
    np.testing.assert_array_equal(result, value)
    torch.testing.assert_close(before, generator.get_state())


@pytest.mark.parametrize(
    "config",
    [
        {"enabled": "true"},
        {"enabled": True, "body_noise_std": 0.2},
        {"enabled": True, "gravity_rotation_std_deg": float("nan")},
        {"enabled": True, "perturb_actions": True},
    ],
)
def test_invalid_or_action_augmentation_configuration_is_rejected(config):
    with pytest.raises(ValueError):
        State44Augmenter(config)
    with pytest.raises(ValueError):
        TrainingAugmentation({"enabled": True}, training=False)


def test_private_rng_advances_by_call_and_changes_by_epoch_worker_rank_without_global_rng():
    config = enabled_config(visual=False)
    before = torch.get_rng_state().clone()
    first, same = TrainingAugmentation(config), TrainingAugmentation(config)
    draw = torch.rand(8, generator=first.generator())
    torch.testing.assert_close(draw, torch.rand(8, generator=same.generator()))
    assert not torch.equal(draw, torch.rand(8, generator=first.generator()))
    first.set_epoch(1)
    epoch1 = torch.rand(8, generator=first.generator())
    assert not torch.equal(draw, epoch1)
    same.set_epoch(1)
    torch.testing.assert_close(epoch1, torch.rand(8, generator=same.generator()))
    with mock.patch(
        "torch.utils.data.get_worker_info", return_value=SimpleNamespace(id=1)
    ):
        worker = torch.rand(8, generator=first.generator())
    assert not torch.equal(worker, epoch1)
    with mock.patch.dict("os.environ", {"RANK": "1"}):
        rank = torch.rand(8, generator=first.generator())
    assert not torch.equal(rank, epoch1)
    torch.testing.assert_close(before, torch.get_rng_state())
    assert first._epoch.is_shared()


@pytest.mark.parametrize("route", ["motion", "joint"])
def test_dataset_augmentation_only_changes_current_observation_and_preserves_sources_and_cache(
    route,
):
    plain = synthetic_dataset(route, None)
    augmented = synthetic_dataset(route, enabled_config())
    source = deepcopy(augmented.episodes)
    baseline = plain[42]
    with mock.patch.object(
        augmented.augmentation.proprio,
        "apply",
        wraps=augmented.augmentation.proprio.apply,
    ) as apply:
        first = augmented[42]
        # The resulting one current-state observation is reused by every FM/IC
        # path consuming this sample, rather than sampling a future sequence.
        assert apply.call_count == 1
    cached = {key: value.clone() for key, value in augmented.frame_cache.items()}
    later = augmented[42]
    for field in (
        "action",
        "past_action",
        "action_is_pad",
        "past_action_is_pad",
        "image_is_pad",
    ):
        torch.testing.assert_close(first[field], baseline[field], rtol=0, atol=0)
        torch.testing.assert_close(later[field], baseline[field], rtol=0, atol=0)
    assert not torch.equal(first["video"], baseline["video"])
    assert not torch.equal(first["video"], later["video"])
    assert not torch.equal(first["proprio"], baseline["proprio"])
    torch.testing.assert_close(
        first["proprio"],
        first["proprio"][:1].expand_as(first["proprio"]),
        rtol=0,
        atol=0,
    )
    for key, expected in cached.items():
        torch.testing.assert_close(augmented.frame_cache[key], expected, rtol=0, atol=0)
    for actual, expected in zip(augmented.episodes[0], source[0]):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("route", ["motion", "joint"])
def test_dataset_proprio_flag_keeps_state_clean_during_visual_only_training(route):
    plain = synthetic_dataset(route, None)[42]
    visual = synthetic_dataset(route, enabled_config(proprio=False))[42]
    torch.testing.assert_close(visual["proprio"], plain["proprio"], rtol=0, atol=0)
    torch.testing.assert_close(visual["action"], plain["action"], rtol=0, atol=0)
    assert not torch.equal(visual["video"], plain["video"])


@pytest.mark.parametrize("route", ["motion", "joint"])
def test_shared_camera_mode_reuses_parameters_across_both_current_future_pairs(route):
    shared_config = enabled_config(proprio=False)
    shared_config["visual"].update(
        camera_mode="shared", overexposure={"probability": 0.0}
    )
    shared = synthetic_dataset(route, shared_config)[42]["video"]
    torch.testing.assert_close(shared[..., :224], shared[..., 224:], rtol=0, atol=0)
    independent_config = deepcopy(shared_config)
    independent_config["visual"]["camera_mode"] = "independent"
    independent = synthetic_dataset(route, independent_config)[42]["video"]
    assert not torch.equal(independent[..., :224], independent[..., 224:])


class WorkerAugmentationProbe(torch.utils.data.Dataset):
    def __init__(self):
        self.augmentation = TrainingAugmentation(enabled_config(visual=False))

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return torch.rand(4, generator=self.augmentation.generator())


def test_epoch_update_reaches_persistent_worker_and_repeats_from_the_same_loader_seed():
    def run():
        dataset = WorkerAugmentationProbe()
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=1,
            persistent_workers=True,
            generator=torch.Generator().manual_seed(22),
        )
        try:
            epoch0 = torch.cat(list(loader))
            dataset.augmentation.set_epoch(1)
            epoch1 = torch.cat(list(loader))
            return epoch0, epoch1
        finally:
            if loader._iterator is not None:
                loader._iterator._shutdown_workers()

    a0, a1 = run()
    b0, b1 = run()
    torch.testing.assert_close(a0, b0)
    torch.testing.assert_close(a1, b1)
    assert not torch.equal(a0, a1)
