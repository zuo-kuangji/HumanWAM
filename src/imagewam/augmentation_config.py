"""Training-only augmentation profiles; validation does not import Torch."""

from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
import math

SCHEMA = "wholebody_training_augmentation_v2"
LEGACY_SCHEMA = "wholebody_training_augmentation_v1"

_VISUAL_V1 = {
    "enabled": True,
    "camera_mode": "independent",
    "exposure": {"probability": 0.35, "gain": [0.85, 1.25], "gamma": [0.85, 1.15]},
    "overexposure": {"probability": 0.12, "gain": [1.25, 1.8]},
    "color": {
        "probability": 0.7, "contrast": [0.9, 1.1],
        "saturation": [0.85, 1.15], "hue": 0.01,
    },
    "jitter": {
        "probability": 0.4, "translation": 0.02,
        "rotation_degrees": 1.5, "scale": [0.98, 1.02],
    },
    "blur": {"probability": 0.12, "sigma": [0.2, 0.8], "kernel_size": 5},
}
_VISUAL = copy.deepcopy(_VISUAL_V1)
_VISUAL["probability"] = 0.5
_VISUAL["color"].update(contrast=[0.85, 1.15], saturation=[0.8, 1.2], hue=5.0 / 360.0)
_VISUAL["blur"].update(sigma=[0.4, 1.2], kernel_size=9)
_PROPRIO_V1 = {
    "enabled": True,
    "body_noise_std": 0.01,
    "body_bias_std": 0.0,
    "hand_noise_std": 0.005,
    "hand_bias_std": 0.0,
    "gravity_rotation_std_deg": 0.0,
    "state_dropout": {
        "probability": 0.1, "dimension_probability": 0.05,
        "max_dropped_dimensions": 3,
    },
}
_PROPRIO = {
    "enabled": True,
    "noise_probability": 0.25,
    "body_noise_std": 0.01,
    "body_bias_std": 0.0,
    "hand_noise_std": 0.005,
    "hand_bias_std": 0.0,
    "gravity_rotation_std_deg": 0.0,
    "state_dropout": {"probability": 0.1},
}


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain(item) for item in value]
    return value


def _merge(default, supplied, location):
    if not isinstance(supplied, Mapping):
        raise ValueError(f"{location} must be a configuration mapping")
    if any(not isinstance(key, str) for key in supplied):
        raise ValueError(f"{location} setting keys must be strings")
    unknown = set(supplied) - set(default)
    if unknown:
        raise ValueError(f"Unknown {location} settings: {sorted(unknown)}")
    result = copy.deepcopy(default)
    for key, value in supplied.items():
        result[key] = (_merge(default[key], value, location + "." + key)
                       if isinstance(default[key], dict) else value)
    return result


def _boolean(value, name):
    if type(value) is not bool:
        raise ValueError(f"{name} must be an explicit boolean")


def _number(value, name, *, minimum=0.0, maximum=None, strict_min=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            or (value <= minimum if strict_min else value < minimum)
            or (maximum is not None and value > maximum)):
        raise ValueError(f"Invalid finite value for {name}")
    return float(value)


def _range(value, name, *, positive=False):
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must contain two ordered bounds")
    lo = _number(value[0], name, strict_min=positive)
    hi = _number(value[1], name, strict_min=positive)
    if hi < lo:
        raise ValueError(f"{name} bounds must be ordered")
    return [lo, hi]


def _normalize_visual(config, *, legacy=False):
    result = _merge(_VISUAL_V1 if legacy else _VISUAL,
                    _plain({} if config is None else config), "augmentation.visual")
    _boolean(result["enabled"], "visual.enabled")
    if not legacy:
        result["probability"] = _number(result["probability"], "visual.probability", maximum=1)
    if result["camera_mode"] not in ("independent", "shared"):
        raise ValueError("visual.camera_mode must be independent or shared")
    for key in ("exposure", "overexposure", "color", "jitter", "blur"):
        result[key]["probability"] = _number(result[key]["probability"], key + ".probability", maximum=1)
    for key in ("exposure", "overexposure"):
        result[key]["gain"] = _range(result[key]["gain"], key + ".gain", positive=True)
    result["exposure"]["gamma"] = _range(result["exposure"]["gamma"], "exposure.gamma", positive=True)
    for key in ("contrast", "saturation"):
        result["color"][key] = _range(result["color"][key], "color." + key)
    result["color"]["hue"] = _number(result["color"]["hue"], "color.hue", maximum=0.5)
    result["jitter"]["translation"] = _number(result["jitter"]["translation"], "jitter.translation", maximum=0.25)
    result["jitter"]["rotation_degrees"] = _number(result["jitter"]["rotation_degrees"], "jitter.rotation_degrees", maximum=45)
    result["jitter"]["scale"] = _range(result["jitter"]["scale"], "jitter.scale", positive=True)
    result["blur"]["sigma"] = _range(result["blur"]["sigma"], "blur.sigma", positive=True)
    kernel = result["blur"]["kernel_size"]
    if type(kernel) is not int or kernel < 1 or kernel > 31 or kernel % 2 != 1:
        raise ValueError("blur.kernel_size must be an odd integer in1..31")
    return result


def normalize_visual_config(config=None):
    """Validate v2 visual settings, including the one-per-sample event gate."""
    return _normalize_visual(config)


def _normalize_proprio(config, *, legacy=False):
    result = _merge(_PROPRIO_V1 if legacy else _PROPRIO,
                    _plain({} if config is None else config), "augmentation.proprio")
    _boolean(result["enabled"], "proprio.enabled")
    for key in ("body_noise_std", "body_bias_std", "hand_noise_std", "hand_bias_std", "gravity_rotation_std_deg"):
        result[key] = _number(result[key], "proprio." + key,
                              maximum=5.0 if key == "gravity_rotation_std_deg" else 0.1)
    if not legacy:
        result["noise_probability"] = _number(result["noise_probability"], "proprio.noise_probability", maximum=1)
    drop = result["state_dropout"]
    for key in (("probability", "dimension_probability") if legacy else ("probability",)):
        drop[key] = _number(drop[key], "state_dropout." + key, maximum=1)
    if legacy:
        count = drop["max_dropped_dimensions"]
        if type(count) is not int or not 1 <= count <= 3:
            raise ValueError("Legacy state dropout allows at most3 body/hand dimensions")
    return result


def normalize_proprio_config(config=None):
    """Validate the v2 noise and whole-embedding dropout configuration."""
    return _normalize_proprio(config)


def default_augmentation(*, enabled=True):
    _boolean(enabled, "augmentation.enabled")
    return {
        "schema": SCHEMA, "enabled": enabled, "seed": 42,
        "visual": normalize_visual_config(), "proprio": normalize_proprio_config(),
    }


def normalize_augmentation(config=None):
    # Missing configuration preserves legacy training and every inference path.
    if config is None:
        return default_augmentation(enabled=False)
    supplied = _plain(config)
    legacy = isinstance(supplied, Mapping) and supplied.get("schema") == LEGACY_SCHEMA
    default = (dict(schema=LEGACY_SCHEMA, enabled=True, seed=42,
                    visual=_normalize_visual(None, legacy=True), proprio=_normalize_proprio(None, legacy=True))
               if legacy else default_augmentation())
    result = _merge(default, supplied, "augmentation")
    if result["schema"] not in (SCHEMA, LEGACY_SCHEMA):
        raise ValueError("Unknown training augmentation schema")
    _boolean(result["enabled"], "augmentation.enabled")
    if type(result["seed"]) is not int or not 0 <= result["seed"] < 2**63:
        raise ValueError("augmentation.seed must be a nonnegative signed64-bit integer")
    result["visual"] = _normalize_visual(result["visual"], legacy=legacy)
    result["proprio"] = _normalize_proprio(result["proprio"], legacy=legacy)
    return result


def augmentation_sha256(config):
    return hashlib.sha256(json.dumps(normalize_augmentation(config), sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()
