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
        "proprio": {"enabled": proprio, "noise_probability": 1},
        "visual": {
            "enabled": visual,
            "probability": 1,
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
    state_aug = State44Augmenter({"enabled": True, "noise_probability": 1, "gravity_rotation_std_deg": 1.0})
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


def test_state_dropout_never_zeros_raw_state_and_noise_zero_preserves_rng():
    normalizer = SimpleNormalizer()
    raw = raw_state()
    raw[:29], raw[29:41] = 2.0, 0.9
    value = normalizer.forward(raw, "state")
    augment = State44Augmenter(
        {
            "enabled": True,
            "noise_probability": 0.0,
            "state_dropout": {"probability": 1.0},
        }
    )
    generator = torch.Generator().manual_seed(9)
    before = generator.get_state().clone()
    result = augment.apply(
        value,
        raw_current_state=raw,
        normalizer=normalizer,
        generator=generator,
    )
    np.testing.assert_array_equal(result, value)
    torch.testing.assert_close(before, generator.get_state())
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


def test_noise_probability_one_has_no_extra_bernoulli_draw():
    normalizer = SimpleNormalizer()
    value = np.full(44, 0.25, dtype=np.float32)
    raw = normalizer.backward(value, "state")
    augment = State44Augmenter({
        "enabled": True, "noise_probability": 1,
    })
    generator = torch.Generator().manual_seed(29)
    reference = torch.Generator().manual_seed(29)
    expected = value.copy()
    expected[:29] += torch.randn(29, generator=reference).numpy() * 0.01
    expected[29:41] += torch.randn(12, generator=reference).numpy() * 0.005
    result = augment.apply(value, raw_current_state=raw, normalizer=normalizer, generator=generator)
    np.testing.assert_array_equal(result, expected)
    torch.testing.assert_close(reference.get_state(), generator.get_state())


def test_half_probability_noise_event_has_expected_frequency_and_is_reproducible():
    normalizer = SimpleNormalizer()
    raw = raw_state()
    value = normalizer.forward(raw, "state")
    augment = State44Augmenter({"noise_probability": 0.5})
    def draw(seed):
        generator = torch.Generator().manual_seed(seed)
        return np.stack([augment.apply(value, raw_current_state=raw,
            normalizer=normalizer, generator=generator) for _ in range(1000)])
    first = draw(71)
    np.testing.assert_array_equal(first, draw(71))
    changed = np.any(first != value, axis=1)
    assert 430 <= changed.sum() <= 570
    np.testing.assert_array_equal(first[:, 41:44], np.repeat(value[None, 41:44], 1000, axis=0))


@pytest.mark.parametrize("probability,expected", [(0, True), (1, False)])
def test_condition_keep_extremes_are_scalar_bool_without_random_draw(probability, expected):
    config = enabled_config(visual=False)
    config["proprio"]["state_dropout"] = {"probability": probability}
    controller = TrainingAugmentation(config)
    generator = torch.Generator().manual_seed(11)
    before = generator.get_state().clone()
    keep = controller.sample_state_keep(generator)
    assert keep.shape == () and keep.dtype is torch.bool and keep.item() is expected
    torch.testing.assert_close(before, generator.get_state())


def test_condition_keep_probability_is_reproducible_and_no_inverted_dropout_scaling():
    config = enabled_config(visual=False)
    def draw(seed):
        controller = TrainingAugmentation(config)
        generator = torch.Generator().manual_seed(seed)
        return torch.stack([controller.sample_state_keep(generator) for _ in range(5000)])
    first = draw(5)
    torch.testing.assert_close(first, draw(5))
    assert first.dtype is torch.bool and 0.87 < first.float().mean() < 0.93
    for controller in (TrainingAugmentation(), TrainingAugmentation(enabled_config(proprio=False))):
        assert controller.sample_state_keep(None).item() is True


def test_enabled_legacy_v1_is_provenance_only_and_cannot_train():
    from imagewam.augmentation_config import LEGACY_SCHEMA
    with pytest.raises(ValueError, match="provenance-only"):
        TrainingAugmentation({"schema": LEGACY_SCHEMA})
    disabled = TrainingAugmentation({"schema": LEGACY_SCHEMA, "enabled": False})
    assert disabled.sample_state_keep(None).item() is True


@pytest.mark.parametrize("camera_mode", ["independent", "shared"])
def test_global_visual_zero_event_is_noop_for_both_cameras_without_rng(camera_mode):
    config = enabled_config(proprio=False)
    config["visual"].update(probability=0, camera_mode=camera_mode)
    controller = TrainingAugmentation(config)
    generator = torch.Generator().manual_seed(15)
    before = generator.get_state().clone()
    parameters = controller.visual_parameters(generator=generator)
    frames = torch.full((2, 3, 8, 8), 0.4)
    for _ in range(2):
        result = controller.apply_video(frames, generator=generator, parameters=parameters)
        torch.testing.assert_close(result, frames, rtol=0, atol=0)
    torch.testing.assert_close(before, generator.get_state())


def test_global_visual_event_is_once_per_sample_not_per_camera():
    config = enabled_config(proprio=False)
    config["visual"]["probability"] = 0.5
    controller = TrainingAugmentation(config)
    generator = torch.Generator().manual_seed(19)
    reference = torch.Generator().manual_seed(19)
    expected = torch.rand(1000, generator=reference) < 0.5
    actual = torch.tensor([controller.visual_parameters(generator=generator) is None
                           for _ in range(1000)])
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(reference.get_state(), generator.get_state())
    assert 430 <= actual.sum() <= 570


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
    ) as apply, mock.patch.object(
        augmented.augmentation, "sample_state_keep",
        wraps=augmented.augmentation.sample_state_keep,
    ) as sample_keep:
        first = augmented[42]
        # The resulting one current-state observation is reused by every FM/IC
        # path consuming this sample, rather than sampling a future sequence.
        assert apply.call_count == 1
        assert sample_keep.call_count == 1
        assert first["state_condition_keep"].shape == ()
        assert first["state_condition_keep"].dtype is torch.bool
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
    assert plain["state_condition_keep"].item() is True
    assert visual["state_condition_keep"].item() is True


@pytest.mark.parametrize("route", ["motion", "joint"])
def test_whole_condition_dropout_returns_flag_without_changing_states_or_supervision(route):
    config = enabled_config(visual=False)
    config["proprio"].update(noise_probability=0, state_dropout={"probability": 1})
    plain = synthetic_dataset(route, None)[42]
    dropped = synthetic_dataset(route, config)[42]
    assert dropped["state_condition_keep"].item() is False
    for field in ("proprio", "video", "action", "past_action", "action_is_pad"):
        torch.testing.assert_close(dropped[field], plain[field], rtol=0, atol=0)


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
        generator = self.augmentation.generator()
        return torch.cat((torch.rand(4, generator=generator),
            self.augmentation.sample_state_keep(generator).float().reshape(1)))


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
