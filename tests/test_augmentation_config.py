"""Portable augmentation configuration contracts, without Torch or model imports."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from imagewam.augmentation_config import (
    SCHEMA, PREVIOUS_SCHEMA, LEGACY_SCHEMA, default_augmentation, normalize_augmentation,
    normalize_visual_config, normalize_proprio_config, augmentation_sha256,
)


REPO = Path(__file__).resolve().parents[1]


def test_missing_config_preserves_legacy_while_explicit_profile_enables_training():
    assert default_augmentation()["schema"] == SCHEMA
    assert normalize_augmentation(None) == default_augmentation(enabled=False)
    assert normalize_augmentation({}) == default_augmentation(enabled=True)
    assert normalize_augmentation({"enabled": False})["enabled"] is False
    assert augmentation_sha256(None) == augmentation_sha256(default_augmentation(enabled=False))
    assert augmentation_sha256(None) != augmentation_sha256({})


def test_profile_yaml_and_hydra_nested_config_match_canonical_defaults():
    OmegaConf = pytest.importorskip("omegaconf").OmegaConf
    profile = OmegaConf.load(REPO / "configs/wholebody/augmentation_v3.yaml")
    assert normalize_augmentation(profile) == default_augmentation()
    assert augmentation_sha256(profile) == augmentation_sha256(default_augmentation())
    assert normalize_visual_config(profile.visual) == default_augmentation()["visual"]
    assert normalize_proprio_config(profile.proprio) == default_augmentation()["proprio"]


def test_updated_color_and_blur_defaults_keep_the_three_event_probabilities():
    config = default_augmentation()
    assert config["visual"]["color"] == {
        "probability": 0.7, "contrast": [0.85, 1.15],
        "saturation": [0.8, 1.2], "hue": 5 / 360,
    }
    assert config["visual"]["blur"] == {
        "probability": 0.12, "sigma": [0.4, 1.2], "kernel_size": 9,
    }
    assert config["visual"]["probability"] == 0.5
    assert config["proprio"]["noise_probability"] == 0.25
    assert config["proprio"]["state_dropout"] == {"probability": 0.1}
    assert config["visual"]["pixel_noise"] == {"probability": 0.25, "std": [4 / 255, 8 / 255]}
    assert config["visual"]["jpeg"] == {"probability": 0.25, "quality": [50, 80]}
    assert config["visual"]["jitter"] == {
        "probability": 0.4, "translation": 0.04,
        "rotation_degrees": 3.0, "scale": [0.96, 1.04],
    }


def test_previous_complete_v2_profile_keeps_explicit_ranges_and_its_original_sha():
    previous = normalize_augmentation({"schema": PREVIOUS_SCHEMA})
    previous["visual"]["color"].update(
        contrast=[0.9, 1.1], saturation=[0.85, 1.15], hue=0.01,
    )
    previous["visual"]["blur"].update(sigma=[0.2, 0.8], kernel_size=5)
    saved = copy.deepcopy(previous)
    assert normalize_augmentation(previous) == saved
    assert augmentation_sha256(previous) == "a0d5efb8d3659022014f921fe1fe113a690192b5fd78a39a505c03cbe4f01aca"
    assert previous == saved
    assert augmentation_sha256(previous) != augmentation_sha256(default_augmentation())


def test_previous_v2_profile_keeps_its_complete_canonical_config_and_hash():
    OmegaConf = pytest.importorskip("omegaconf").OmegaConf
    profile = OmegaConf.load(REPO / "configs/wholebody/augmentation_v2.yaml")
    previous = normalize_augmentation(profile)
    assert previous["schema"] == PREVIOUS_SCHEMA
    assert "pixel_noise" not in previous["visual"] and "jpeg" not in previous["visual"]
    assert previous["visual"]["jitter"]["translation"] == 0.02
    assert augmentation_sha256(profile) == "bfe4f655f81d282e15fcb0e6e6aaf6899a75a482c3b5724adbdbe88facb62d7e"
    previous["enabled"] = False
    assert augmentation_sha256(previous) == "2b3d6059c95c5e03759d4994479db5aa395da4288ae41ec2dafec6b7aec4f8ba"
    for key in ("pixel_noise", "jpeg"):
        with pytest.raises(ValueError, match="Unknown"):
            normalize_augmentation({"schema": PREVIOUS_SCHEMA, "visual": {key: {"probability": 0.25}}})


def test_runtime_visual_normalization_does_not_enable_new_ops_on_previous_or_partial_inputs():
    previous = normalize_augmentation({"schema": PREVIOUS_SCHEMA})["visual"]
    for supplied in (previous, {}, {"color": {"hue": 0}}):
        runtime = normalize_visual_config(supplied)
        assert runtime["pixel_noise"]["probability"] == 0
        assert runtime["jpeg"]["probability"] == 0
        assert runtime["jitter"]["translation"] == 0.02
    runtime = normalize_visual_config(previous)
    assert {key: value for key, value in runtime.items() if key not in {"pixel_noise", "jpeg"}} == previous
    for supplied_key, absent_key in (("pixel_noise", "jpeg"), ("jpeg", "pixel_noise")):
        runtime = normalize_visual_config({supplied_key: {"probability": 0.6}})
        assert runtime[supplied_key]["probability"] == 0.6
        assert runtime[absent_key]["probability"] == 0
    assert normalize_visual_config(None) == default_augmentation()["visual"]
    assert normalize_augmentation({"visual": {}})["visual"] == default_augmentation()["visual"]


def test_legacy_v1_profile_preserves_its_canonical_hash_for_checkpoint_provenance():
    OmegaConf = pytest.importorskip("omegaconf").OmegaConf
    profile = OmegaConf.load(REPO / "configs/wholebody/augmentation_v1.yaml")
    normalized = normalize_augmentation(profile)
    assert normalized["schema"] == LEGACY_SCHEMA
    assert normalized["proprio"]["state_dropout"]["max_dropped_dimensions"] == 3
    assert "noise_probability" not in normalized["proprio"]
    assert "probability" not in normalized["visual"]
    assert augmentation_sha256(profile) == "87ecf74008836ffa3580294254248ef9b0ef79e844a752ae843c30929e29645d"
    normalized["enabled"] = False
    assert augmentation_sha256(normalized) == "cebe5c431fb1d64fd95d361fb386f809d007ab97e26b7d731110e1a68b283263"
    with pytest.raises(ValueError):
        normalize_augmentation({"schema": LEGACY_SCHEMA, "proprio": {"noise_probability": 0.5}})
    with pytest.raises(ValueError):
        normalize_augmentation({"schema": LEGACY_SCHEMA, "visual": {"probability": 0.5}})


def test_config_and_normalized_result_are_fully_independent_deep_copies():
    supplied = default_augmentation()
    saved = copy.deepcopy(supplied)
    normalized = normalize_augmentation(supplied)
    normalized["visual"]["exposure"]["gain"][0] = 0.11
    normalized["proprio"]["state_dropout"]["probability"] = 1
    normalized["visual"]["jpeg"]["quality"][0] = 1
    assert supplied == saved
    assert default_augmentation() == saved
    supplied["visual"]["color"]["saturation"][0] = 0.12
    assert normalized["visual"]["color"]["saturation"] == saved["visual"]["color"]["saturation"]


def test_hash_is_order_independent_binds_full_profile_and_has_defined_json_encoding():
    config = default_augmentation()
    reverse = dict(reversed(list(config.items())))
    reverse["visual"] = dict(reversed(list(config["visual"].items())))
    assert augmentation_sha256(reverse) == augmentation_sha256(config)
    canonical = json.dumps(normalize_augmentation(config), sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert augmentation_sha256(config) == hashlib.sha256(canonical.encode()).hexdigest()
    for changed in ({"seed": 43}, {"visual": {"camera_mode": "shared"}},
                    {"proprio": {"hand_noise_std": 0.01}}, {"enabled": False},
                    {"visual": {"pixel_noise": {"std": [0.01, 0.02]}}},
                    {"visual": {"jpeg": {"quality": [60, 90]}}}):
        assert augmentation_sha256(changed) != augmentation_sha256(config)


def test_equivalent_numeric_spellings_have_the_same_canonical_hash():
    first = {"visual": {"blur": {"probability": 0}}}
    second = {"visual": {"blur": {"probability": 0.0}}}
    assert augmentation_sha256(first) == augmentation_sha256(second)


def test_top_level_and_per_branch_disabled_flags_remain_explicit():
    config = normalize_augmentation({"enabled": False, "visual": {"enabled": True}})
    assert config["enabled"] is False and config["visual"]["enabled"] is True
    config = normalize_augmentation({"visual": {"enabled": False}, "proprio": {"enabled": False}})
    assert config["enabled"] is True
    assert config["visual"]["enabled"] is config["proprio"]["enabled"] is False


@pytest.mark.parametrize("invalid", [False, True, 1, "default", [], ["enabled"]])
def test_non_mapping_profiles_are_rejected(invalid):
    with pytest.raises(ValueError): normalize_augmentation(invalid)


@pytest.mark.parametrize("invalid", [True, -1, 2**63, 42.0, "42", None])
def test_seed_type_and_signed64_bounds_are_strict(invalid):
    with pytest.raises(ValueError): normalize_augmentation({"seed": invalid})


@pytest.mark.parametrize("invalid", [
    {"schema": "other"}, {"enabled": 1}, {"visual": None}, {"proprio": []},
    {"visual": {"enabled": "true"}}, {"proprio": {"enabled": 0}},
    {"visual": {"camera_mode": "all"}}, {"visual": {"exposure": "off"}},
    {"visual": {"exposure": {"probability": True}}},
])
def test_schema_boolean_branch_and_block_types_are_checked(invalid):
    with pytest.raises(ValueError): normalize_augmentation(invalid)


@pytest.mark.parametrize("invalid", [
    {"typo": True}, {"visual": {"random_flip": True}},
    {"visual": {"exposure": {"gian": [0.8, 1.2]}}},
    {"proprio": {"perturb_actions": True}},
    {"proprio": {"state_dropout": {"maximum_dims": 3}}},
    {"proprio": {"state_dropout": {"max_dropped_dimensions": 3}}},
    {"proprio": {"state_dropout": {"dimension_probability": 0.05}}},
    {1: True, "typo": False}, {"visual": {1: True, "typo": False}},
])
def test_unknown_and_non_string_fields_fail_with_clear_value_errors(invalid):
    with pytest.raises(ValueError, match="Unknown|keys must be strings"):
        normalize_augmentation(invalid)


@pytest.mark.parametrize("invalid", [
    {"visual": {"overexposure": {"probability": 1.01}}},
    {"visual": {"probability": 1.01}},
    {"visual": {"probability": True}},
    {"visual": {"probability": float("nan")}},
    {"visual": {"blur": {"probability": -0.1}}},
    {"visual": {"color": {"hue": 0.51}}},
    {"visual": {"jitter": {"translation": 0.26}}},
    {"visual": {"jitter": {"rotation_degrees": 46}}},
    {"visual": {"exposure": {"gamma": [0, 1]}}},
    {"visual": {"overexposure": {"gain": [1.8, 1.2]}}},
    {"visual": {"jitter": {"scale": [1]}}},
    {"visual": {"blur": {"sigma": [float("nan"), 1]}}},
    {"visual": {"color": {"contrast": [0.9, float("inf")]}}},
    {"visual": {"blur": {"kernel_size": 4}}},
    {"visual": {"blur": {"kernel_size": 33}}},
    {"visual": {"blur": {"kernel_size": True}}},
    {"visual": {"pixel_noise": {"probability": True}}},
    {"visual": {"pixel_noise": {"probability": 1.01}}},
    {"visual": {"pixel_noise": {"std": [-0.01, 0.02]}}},
    {"visual": {"pixel_noise": {"std": [0.02, 0.11]}}},
    {"visual": {"pixel_noise": {"std": [0.03, 0.02]}}},
    {"visual": {"pixel_noise": {"std": [0.01]}}},
    {"visual": {"pixel_noise": {"std": [float("nan"), 0.02]}}},
    {"visual": {"pixel_noise": {"std": [0.01, float("inf")]}}},
    {"visual": {"jpeg": {"probability": -0.01}}},
    {"visual": {"jpeg": {"quality": [0, 80]}}},
    {"visual": {"jpeg": {"quality": [50, 101]}}},
    {"visual": {"jpeg": {"quality": [80, 50]}}},
    {"visual": {"jpeg": {"quality": [50.0, 80]}}},
    {"visual": {"jpeg": {"quality": [True, 80]}}},
    {"visual": {"jpeg": {"quality": [50]}}},
    {"proprio": {"body_noise_std": -0.01}},
    {"proprio": {"body_bias_std": 0.11}},
    {"proprio": {"hand_noise_std": 0.11}},
    {"proprio": {"gravity_rotation_std_deg": 6}},
    {"proprio": {"state_dropout": {"probability": 1.1}}},
    {"proprio": {"noise_probability": -0.01}},
    {"proprio": {"noise_probability": 1.01}},
    {"proprio": {"noise_probability": True}},
    {"proprio": {"noise_probability": float("nan")}},
])
def test_numeric_bounds_and_range_shape_are_validated(invalid):
    with pytest.raises(ValueError): normalize_augmentation(invalid)


def test_zero_augmentation_strengths_and_zero_event_probabilities_are_valid():
    config = normalize_augmentation({"visual": {"color": {"saturation": [0, 0], "hue": 0}},
        "proprio": {"body_noise_std": 0, "hand_noise_std": 0,
                    "noise_probability": 0, "state_dropout": {"probability": 0}}})
    assert config["visual"]["color"]["saturation"] == [0.0, 0.0]
    assert config["proprio"]["noise_probability"] == 0.0
    assert config["proprio"]["state_dropout"]["probability"] == 0.0
    assert default_augmentation()["proprio"]["noise_probability"] == 0.25
    assert default_augmentation()["proprio"]["state_dropout"] == {"probability": 0.1}
    assert default_augmentation()["visual"]["probability"] == 0.5


@pytest.mark.parametrize("quality", [[1, 1], [100, 100], [1, 100]])
def test_jpeg_integer_quality_endpoints_and_zero_pixel_noise_are_supported(quality):
    config = normalize_augmentation({"visual": {
        "jpeg": {"quality": quality}, "pixel_noise": {"std": [0, 0]},
    }})
    assert config["visual"]["jpeg"]["quality"] == quality
    assert config["visual"]["pixel_noise"]["std"] == [0.0, 0.0]


def test_import_validation_and_hash_do_not_import_tensor_or_image_runtimes():
    code = """import sys
from imagewam.augmentation_config import normalize_augmentation, augmentation_sha256
assert normalize_augmentation(None)['enabled'] is False
assert len(augmentation_sha256({'enabled':True})) == 64
assert not {'torch','torchvision','numpy','PIL','hydra'}.intersection(sys.modules)
print('ok')
"""
    result = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True,
                            env=dict(os.environ, PYTHONPATH=str(REPO / "src")))
    assert result.stdout.strip() == "ok"
