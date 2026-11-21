import math

import numpy as np
import pytest

from humanwam.augmentation import UNIFIED_V1, UNIFIED_V2, SampleKey, UnifiedAugmentation


def test_v2_only_changes_clean_rgb_gate_and_v1_remains_replayable():
    old = UnifiedAugmentation.from_hb_profile("unified_v1")
    new = UnifiedAugmentation.from_hb_profile("unified_v2")
    assert old.profile == UNIFIED_V1
    assert new.profile == UnifiedAugmentation().profile == UNIFIED_V2
    assert new.profile == UnifiedAugmentation.from_hb_profile("unified").profile
    strong_kinds = ("affine", "overexposure", "noise", "jpeg")
    observed_strong = set()
    for frame in range(250):
        key = SampleKey(7, frame, 3)
        draws = np.random.default_rng([42, 7, frame, 3, 0, 0])
        draw = draws.random()
        previous = old.sample_rgb(key, cameras=2)
        current = new.sample_rgb(key, cameras=2)
        assert (previous is None) == (draw < 0.20)
        assert (current is None) == (draw < 0.05)
        if previous is not None:
            assert current == previous
        if current is not None:
            expected = draws.choice(strong_kinds) if draws.random() < 0.15 else None
            assert current["strong"] == expected
            observed_strong.add(expected)
    assert observed_strong == {*strong_kinds, None}
    state = np.zeros((2, 44), dtype=np.float32)
    np.testing.assert_array_equal(
        new.apply_state(state, body_dims=29, sample=key),
        old.apply_state(state, body_dims=29, sample=key),
    )


def test_raw_body_disturbance_keeps_other_channels_and_replays():
    augmentation = UnifiedAugmentation(seed=42)
    sample = SampleKey(4, 16, 2)
    for width, body in ((44, 29), (78, 31)):
        state = np.arange(2 * width, dtype=np.float32).reshape(2, width) / 100
        saved = state.copy()
        output = augmentation.apply_state(state, body_dims=body, sample=sample)
        np.testing.assert_array_equal(state, saved)
        np.testing.assert_array_equal(output[:, body:], saved[:, body:])
        np.testing.assert_array_equal(
            output, augmentation.apply_state(saved, body_dims=body, sample=sample)
        )
        assert np.max(np.abs(output[:, :body] - saved[:, :body])) <= math.radians(2.75)
        next_epoch = augmentation.apply_state(saved, body_dims=body, sample=SampleKey(4, 16, 3))
        assert not np.array_equal(output, next_epoch)


def _rgb_runtime():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    return torch


def _key_with(augmentation, predicate):
    for frame in range(200):
        key = SampleKey(7, frame, 3)
        if predicate(augmentation.sample_rgb(key, cameras=2)):
            return key
    pytest.fail("Expected shared augmentation sampling branch was not reached")


def test_shared_scene_transform_and_observation_only_sensor_noise():
    torch = _rgb_runtime()
    aug = UnifiedAugmentation(seed=42)
    key = _key_with(aug, lambda p: p is not None and all(c["noise"] for c in p["cameras"]))
    frame = torch.rand((2, 3, 32, 48), generator=torch.Generator().manual_seed(19))
    clip = frame[None].repeat(4, 1, 1, 1, 1)
    pristine = clip.clone()
    observed = aug.apply_rgb(clip, sample=key, observation_mask=[True, True, False, False])
    scene = aug.apply_rgb(clip, sample=key, observation_mask=[False] * 4)
    torch.testing.assert_close(clip, pristine, rtol=0, atol=0)
    torch.testing.assert_close(scene[0], scene[2], rtol=0, atol=0)
    torch.testing.assert_close(observed[2:], scene[2:], rtol=0, atol=0)
    assert not torch.equal(observed[0], scene[0])
    assert not torch.equal(observed[0], observed[1])
    assert observed.dtype == clip.dtype and observed.min() >= 0 and observed.max() <= 1
    changed_future = clip.clone()
    changed_future[2:] = 1 - changed_future[2:]
    revisited = aug.apply_rgb(changed_future, sample=key, observation_mask=[True, True, False, False])
    torch.testing.assert_close(observed[:2], revisited[:2], rtol=0, atol=0)


def test_clean_image_branch_and_strong_affine_keep_frame_pairing():
    torch = _rgb_runtime()
    aug = UnifiedAugmentation()
    clip = torch.linspace(0, 1, 32 * 48).reshape(1, 1, 1, 32, 48).repeat(2, 2, 3, 1, 1)
    clean = _key_with(aug, lambda p: p is None)
    kept = aug.apply_rgb(clip, sample=clean, observation_mask=[True, False])
    assert kept.data_ptr() != clip.data_ptr()
    torch.testing.assert_close(kept, clip, rtol=0, atol=0)

    affine = _key_with(aug, lambda p: p is not None and p["strong"] == "affine")
    moving = aug.apply_rgb(clip, sample=affine, observation_mask=[False, False])
    torch.testing.assert_close(moving[0], moving[1], rtol=0, atol=0)
    assert not torch.equal(moving[0, 0], moving[0, 1])
    with pytest.raises(ValueError, match="observation_mask"):
        aug.apply_rgb(clip, sample=affine, observation_mask=[1, 0])
