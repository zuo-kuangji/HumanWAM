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
    SCHEMA, default_augmentation, normalize_augmentation,
    normalize_visual_config, normalize_proprio_config, augmentation_sha256,
)


REPO = Path(__file__).resolve().parents[1]


def test_missing_config_preserves_legacy_while_explicit_profile_enables_training():
    assert normalize_augmentation(None) == default_augmentation(enabled=False)
    assert normalize_augmentation({}) == default_augmentation(enabled=True)
    assert normalize_augmentation({"enabled": False})["enabled"] is False
    assert augmentation_sha256(None) == augmentation_sha256(default_augmentation(enabled=False))
    assert augmentation_sha256(None) != augmentation_sha256({})


def test_profile_yaml_and_hydra_nested_config_match_canonical_defaults():
    OmegaConf = pytest.importorskip("omegaconf").OmegaConf
    profile = OmegaConf.load(REPO / "configs/wholebody/augmentation_v1.yaml")
    assert normalize_augmentation(profile) == default_augmentation()
    assert augmentation_sha256(profile) == augmentation_sha256(default_augmentation())
    assert normalize_visual_config(profile.visual) == default_augmentation()["visual"]
    assert normalize_proprio_config(profile.proprio) == default_augmentation()["proprio"]


def test_config_and_normalized_result_are_fully_independent_deep_copies():
    supplied = default_augmentation()
    saved = copy.deepcopy(supplied)
    normalized = normalize_augmentation(supplied)
    normalized["visual"]["exposure"]["gain"][0] = 0.11
    normalized["proprio"]["state_dropout"]["max_dropped_dimensions"] = 1
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
                    {"proprio": {"hand_noise_std": 0.01}}, {"enabled": False}):
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
    {1: True, "typo": False}, {"visual": {1: True, "typo": False}},
])
def test_unknown_and_non_string_fields_fail_with_clear_value_errors(invalid):
    with pytest.raises(ValueError, match="Unknown|keys must be strings"):
        normalize_augmentation(invalid)


@pytest.mark.parametrize("invalid", [
    {"visual": {"overexposure": {"probability": 1.01}}},
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
    {"proprio": {"body_noise_std": -0.01}},
    {"proprio": {"body_bias_std": 0.11}},
    {"proprio": {"hand_noise_std": 0.11}},
    {"proprio": {"gravity_rotation_std_deg": 6}},
    {"proprio": {"state_dropout": {"probability": 1.1}}},
    {"proprio": {"state_dropout": {"dimension_probability": -0.01}}},
    {"proprio": {"state_dropout": {"max_dropped_dimensions": 4}}},
    {"proprio": {"state_dropout": {"max_dropped_dimensions": 0}}},
    {"proprio": {"state_dropout": {"max_dropped_dimensions": 3.0}}},
])
def test_numeric_bounds_and_range_shape_are_validated(invalid):
    with pytest.raises(ValueError): normalize_augmentation(invalid)


def test_zero_augmentation_strengths_and_zero_dimension_probability_are_valid():
    config = normalize_augmentation({"visual": {"color": {"saturation": [0, 0], "hue": 0}},
        "proprio": {"body_noise_std": 0, "hand_noise_std": 0,
                    "state_dropout": {"dimension_probability": 0}}})
    assert config["visual"]["color"]["saturation"] == [0.0, 0.0]
    assert config["proprio"]["state_dropout"]["dimension_probability"] == 0.0


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
