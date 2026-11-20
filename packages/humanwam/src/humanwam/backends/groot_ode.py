"""Lazy public ODE-POLICY G1 adapter with explicit last-sent-token requests.

Every prediction supplies normalized r0 explicitly. Previewed actions never
advance execution history, and a network reset is not a robot acknowledgement.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from .groot_ode_commands import gr00t_source_root, load_native_config, source_root

ODE_SOURCE_COMMIT = "fd8bec470faf7793702533358867db02819e8f27"
GR00T_SOURCE_COMMIT = "23ace64f17aa5015259b8609d371eb61a357c776"
CAMERAS = ("ego_view", "chest")
NATIVE_CAMERAS = tuple("observation.images." + key for key in CAMERAS)
LANGUAGE_KEY = "annotation.human.task_description"
STATE_FIELDS = (
    ("left_leg", 6), ("right_leg", 6), ("waist", 3),
    ("left_arm", 7), ("right_arm", 7),
    ("left_hand", 6), ("right_hand", 6), ("projected_gravity", 3),
)
ACTION_FIELDS = (("motion_token", 64), ("left_hand_joints", 6), ("right_hand_joints", 6))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_source(root, commit):
    """Read-only source guard, before importing any optional model packages."""
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode or result.stdout.strip() != commit:
        raise ValueError(f"Expected pinned source {commit} at {root}")
    changes = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True, text=True, check=False,
    )
    if changes.returncode or changes.stdout.strip():
        raise ValueError(f"Pinned runtime source has local changes: {root}")


def _binding_digest(binding):
    return hashlib.sha256(json.dumps(
        binding, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def _checkpoint_manifest(cfg):
    """CPU preflight; the public loader additionally validates tensors and base weights."""
    root = Path(cfg["inference"]["checkpoint_dir"] or cfg["paths"]["output"]).resolve()
    manifest = json.loads((root / "policy_manifest.json").read_text())
    if manifest.get("schema") != "ode-g1-policy-v1":
        raise ValueError("Use a public ODE G1 policy_manifest.json checkpoint")
    binding = manifest["binding"]
    if _binding_digest(binding) != manifest["binding_sha256"]:
        raise ValueError("ODE checkpoint binding changed")
    for file, key in (("ode_config.json", "config_sha256"), ("normalization.json", "normalizer_sha256")):
        if _sha256(root / file) != binding[key]:
            raise ValueError(f"ODE checkpoint binding mismatch: {file}")
    if (root / "initial_pose.json").is_file():
        if _sha256(root / "initial_pose.json") != binding.get("initial_pose_sha256"):
            raise ValueError("ODE initial pose binding mismatch")
    if binding["camera_keys"] != list(NATIVE_CAMERAS):
        raise ValueError("ODE checkpoint requires the ordered ego_view/chest G1 cameras")
    head_config = json.loads((root / "ode_config.json").read_text())
    if head_config != cfg["head"] or head_config["horizon"] != 16:
        raise ValueError("ODE checkpoint/head configuration must match the H16 G1 recipe")
    stats = json.loads((root / "normalization.json").read_text())
    task = {"name": stats["task"], "revision": stats["task_revision"]}
    if task != binding["task"] or task["name"] != cfg["dataset"]["name"]:
        raise ValueError("ODE checkpoint task binding differs from native configuration")
    if cfg["dataset"]["revision"] is not None and task["revision"] != cfg["dataset"]["revision"]:
        raise ValueError("ODE checkpoint task revision differs from native configuration")
    # The native TaskNormalizer is NumPy-only, but model package imports remain
    # lazy. Its complete normalization validation runs in public load_policy.
    if stats.get("schema") != "g1-task-ode-q01q99-fixed-hands-v1":
        raise ValueError("Incompatible ODE G1 normalization schema")
    head = manifest["head"]
    path = root / "final_head.safetensors"
    if head["path"] != path.name or path.stat().st_size != head["size_bytes"]:
        raise ValueError("ODE head file identity differs")
    if _sha256(path) != head["sha256"]:
        raise ValueError("ODE head full SHA differs")
    return copy.deepcopy(manifest)


def _native_imports(source, options=None):
    """Select the exact ODE package, leaving other GR00T environments untouched."""
    _git_source(source, ODE_SOURCE_COMMIT)
    configured_groot = gr00t_source_root(options or {}, source)
    spec = importlib.util.find_spec("gr00t")
    if spec is None or not spec.origin:
        raise ImportError("Install Isaac-GR00T-ODE at the pinned revision in the isolated ODE environment")
    package_file = Path(spec.origin).resolve()
    groot_root = package_file.parent.parent
    if groot_root != configured_groot:
        raise ValueError("Installed GR00T does not match gr00t_source_dir; use the isolated ODE environment")
    _git_source(groot_root, GR00T_SOURCE_COMMIT)
    if package_file != groot_root / "gr00t/__init__.py":
        raise ValueError("Unexpected GR00T package source location")
    # Refuse already imported modules from another source before model loading.
    for name, module in tuple(sys.modules.items()):
        if name == "gr00t" or name.startswith("gr00t."):
            file = getattr(module, "__file__", None)
            if file and not Path(file).resolve().is_relative_to(groot_root):
                raise ValueError("A different GR00T environment is already imported")
    package = source / "groot/src"
    existing = importlib.util.find_spec("ode_g1")
    if existing and (not existing.origin or Path(existing.origin).resolve() != package / "ode_g1/__init__.py"):
        raise ValueError("A different ode_g1 installation shadows the pinned source")
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))
    for name, module in tuple(sys.modules.items()):
        if name == "ode_g1" or name.startswith("ode_g1."):
            file = getattr(module, "__file__", None)
            if file and not Path(file).resolve().is_relative_to(package):
                raise ValueError("A different ODE runtime is already imported")
    loader = importlib.import_module("ode_g1.loader")
    runtime = importlib.import_module("ode_g1.runtime")
    for module in (loader, runtime):
        if not Path(module.__file__).resolve().is_relative_to(package):
            raise ValueError("Unexpected ODE runtime source")
    return loader, runtime


class _NativeRuntime:
    def __init__(self, options, cfg):
        loader, runtime = _native_imports(source_root(options["source_dir"]), options)
        base = cfg["base_model"]
        directory = cfg["inference"]["checkpoint_dir"] or cfg["paths"]["output"]
        model, head, normalizer = loader.load_policy(
            base["path"], directory, device=cfg["device"],
            revision=base["revision"], cosmos_revision=base["cosmos_revision"],
            cosmos_path=base["cosmos_path"], allow_download=False,
            channels_last_3d=base["channels_last_3d"],
            camera_keys=cfg["dataset"]["camera_keys"], verify_head_sha256=True,
        )
        if head.config.horizon != 16:
            raise ValueError("ODE runtime did not load an H16 head")
        self.normalizer = normalizer
        self.policy = runtime.ODEPolicy(head, normalizer, device=cfg["device"], rgb_encoder=runtime.RGBEncoder(
            model, camera_keys=cfg["dataset"]["camera_keys"], device=cfg["device"],
        ))

    def predict_rgb(self, *args, **kwargs):
        return self.policy.predict_rgb(*args, **kwargs)

    def predict_cached(self, *args, **kwargs):
        return self.policy.predict_cached(*args, **kwargs)


class GrootODEPolicy:
    """GR00T feedback76 modalities, with mandatory request-owned causal r0."""

    # The public ODE environment pins GR00T23ace, whose client predates mnp
    # ndarray envelopes and the newer ModalityConfig marker spelling.
    wire_format = "gr00t_npy"

    def __init__(self, config, *, runtime_factory=None):
        self.config = dict(config)
        self.source = source_root(config["source_dir"])
        _git_source(self.source, ODE_SOURCE_COMMIT)
        self.native_config = load_native_config(config, self.source)
        if self.native_config["head"]["horizon"] != 16:
            raise ValueError("HumanWAM SONIC alignment requires the H16 ODE checkpoint")
        if self.native_config["dataset"]["camera_keys"] != list(NATIVE_CAMERAS):
            raise ValueError("Use ordered observation.images.ego_view/chest cameras")
        self.manifest = _checkpoint_manifest(self.native_config)
        self._runtime_factory = runtime_factory or _NativeRuntime
        self._runtime = None

    def get_modality_config(self):
        return {
            "video": {"delta_indices": [0], "modality_keys": list(CAMERAS)},
            "state": {"delta_indices": [0], "modality_keys": [key for key, _ in STATE_FIELDS]},
            "action": {"delta_indices": list(range(16)), "modality_keys": [key for key, _ in ACTION_FIELDS],
                       "action_configs": [{"rep": "ABSOLUTE", "type": "NON_EEF", "format": "DEFAULT"} for _ in ACTION_FIELDS]},
            "language": {"delta_indices": [0], "modality_keys": [LANGUAGE_KEY]},
        }

    def _carry(self, options):
        import numpy as np

        if not isinstance(options, dict) or set(options) not in ({"at_episode_start"}, {"previous_body64"}):
            raise ValueError("Each request must declare at_episode_start=true OR previous_body64 from the last actually sent command")
        if "at_episode_start" in options:
            if options["at_episode_start"] is not True:
                raise ValueError("at_episode_start must be the explicit boolean true")
            return None
        previous = np.asarray(options["previous_body64"])
        numeric = np.issubdtype(previous.dtype, np.floating) or np.issubdtype(previous.dtype, np.integer)
        if previous.shape != (64,) or not numeric or not np.isfinite(previous).all():
            raise ValueError("previous_body64 must be finite physical [64]")
        previous = previous.astype(np.float32)
        if not np.isfinite(previous).all():
            raise ValueError("previous_body64 exceeds finite float32 coordinates")
        return previous.copy()

    def _state(self, state):
        import numpy as np

        if not isinstance(state, dict) or set(state) != {key for key, _ in STATE_FIELDS}:
            raise ValueError("Use the GR00T feedback76 named raw state44 fields")
        pieces = []
        for key, width in STATE_FIELDS:
            value = np.asarray(state[key])
            if value.dtype != np.float32 or value.shape != (1, 1, width) or not np.isfinite(value).all():
                raise ValueError(f"state.{key} must be finite raw float32 [1,1,{width}]")
            pieces.append(value[0, 0])
        result = np.concatenate(pieces)
        if ((result[29:41] < 0) | (result[29:41] > 1)).any():
            raise ValueError("Measured Inspire hands must use physical [0,1]")
        return result

    def _runtime_initial(self, previous):
        import numpy as np

        if self._runtime is None:
            self._runtime = self._runtime_factory(self.config, self.native_config)
        if previous is None:
            return np.zeros(64, np.float32)
        physical = np.concatenate((previous, np.zeros(12, np.float32)))
        normalized = np.asarray(self._runtime.normalizer.normalize_actions_unclipped(physical), dtype=np.float32)
        if normalized.shape != (76,) or not np.isfinite(normalized).all():
            raise ValueError("Native unclipped action normalization must return finite [76]")
        return normalized[:64].copy()

    def _result(self, output):
        import numpy as np

        values = np.asarray(output, dtype=np.float32)
        if values.shape != (16, 76) or not np.isfinite(values).all():
            raise ValueError("ODE prediction must be finite physical [16,76]")
        if ((values[:, 64:] < 0) | (values[:, 64:] > 1)).any():
            raise ValueError("ODE runtime returned invalid physical hand targets")
        offset, actions = 0, {}
        for key, width in ACTION_FIELDS:
            actions[key] = values[None, :, offset:offset + width].copy()
            offset += width
        return actions, {
            "method": "control_step_sonic_latent_ode", "horizon": 16,
            "action_space": "sonic_motion64_inspire12", "normalized": False,
            "action_frequency_hz": 30, "carry_source": "request_last_actually_sent_body64_or_explicit_BOS",
            "robot_execution_verified": False, "upstream_commit": ODE_SOURCE_COMMIT,
        }

    def get_action(self, observation, options=None):
        import numpy as np

        previous = self._carry(options)
        if not isinstance(observation, dict) or set(observation) != {"video", "state", "language"}:
            raise ValueError("Use native GR00T video/state/language observations")
        state = self._state(observation["state"])
        video = observation["video"]
        if not isinstance(video, dict) or set(video) != set(CAMERAS):
            raise ValueError("Provide exactly current ego_view/chest RGB")
        images = {}
        for key, native_key in zip(CAMERAS, NATIVE_CAMERAS):
            value = np.asarray(video[key])
            if value.dtype != np.uint8 or value.ndim != 5 or value.shape[:2] != (1, 1) or value.shape[-1] != 3 or min(value.shape[2:4]) < 1:
                raise ValueError(f"video.{key} must be current uint8 RGB [1,1,H,W,3]")
            images[native_key] = value[0, 0].copy()
        language = observation["language"]
        if not isinstance(language, dict) or set(language) != {LANGUAGE_KEY}:
            raise ValueError("Use the native annotation.human.task_description language key")
        text = language[LANGUAGE_KEY]
        if not isinstance(text, list) or len(text) != 1 or not isinstance(text[0], list) or len(text[0]) != 1 or not isinstance(text[0][0], str) or not text[0][0].strip():
            raise ValueError("Task instruction must be nonempty [[str]]")
        initial = self._runtime_initial(previous)
        return self._result(self._runtime.predict_rgb(images, text[0][0], state, initial_body=initial))

    def predict_cached(self, condition, state, *, options):
        """Explicit current feature API; cached tokens are not SONIC wire modalities."""
        import numpy as np

        previous = self._carry(options)
        if not isinstance(condition, dict) or not {"backbone_features", "backbone_attention_mask", "image_mask"} <= set(condition) or set(condition) - {"backbone_features", "backbone_attention_mask", "image_mask", "frame_ids"}:
            raise ValueError("Use only native current backbone features/masks/frame_ids")
        features = np.asarray(condition["backbone_features"])
        if features.dtype not in (np.float16, np.float32) or features.ndim != 3 or features.shape[0] != 1 or features.shape[1] < 1 or features.shape[2] != 2048 or not np.isfinite(features).all():
            raise ValueError("Current N1.7 backbone_features must be finite float16/32 [1,S,2048]")
        clean = {"backbone_features": features.copy()}
        for name in ("backbone_attention_mask", "image_mask"):
            value = np.asarray(condition[name])
            if value.shape != features.shape[:2] or not np.isin(value, [0, 1]).all():
                raise ValueError(f"{name} must be binary [1,S]")
            clean[name] = value.astype(bool)
        valid, image = clean["backbone_attention_mask"], clean["image_mask"]
        if not (valid & image).any() or not (valid & ~image).any():
            raise ValueError("Each current condition needs valid image and text tokens")
        frame_ids = np.asarray(condition.get("frame_ids", np.zeros(features.shape[:2], np.int64)))
        if frame_ids.shape != features.shape[:2] or not np.issubdtype(frame_ids.dtype, np.integer) or (frame_ids[valid] != 0).any():
            raise ValueError("frame_ids must declare a single current observation slot")
        clean["frame_ids"] = frame_ids.astype(np.int64)
        raw_state = np.asarray(state)
        if raw_state.dtype != np.float32 or raw_state.shape != (44,) or not np.isfinite(raw_state).all() or ((raw_state[29:41] < 0) | (raw_state[29:41] > 1)).any():
            raise ValueError("Cached prediction requires finite raw float32 state44 and physical hands")
        initial = self._runtime_initial(previous)
        return self._result(self._runtime.predict_cached(clean, raw_state.copy(), initial_body=initial))

    def predict(self, observation):
        """Offline RGB NPZ/metadata, or explicit {condition,current_state} cached input."""
        import numpy as np

        observation = dict(observation)
        options = {key: observation.pop(key) for key in ("at_episode_start", "previous_body64") if key in observation}
        if set(observation) == {"condition", "current_state"}:
            actions, info = self.predict_cached(observation["condition"], observation["current_state"], options=options)
        else:
            actions, info = self.get_action(observation, options)
        return {"actions": actions, "info": info,
                "action": np.concatenate([actions[key][0] for key, _ in ACTION_FIELDS], axis=-1)}

    def reset(self, options=None):
        if options is not None and not isinstance(options, dict):
            raise ValueError("reset options must be a mapping or null")
        # Deliberately no persistent carry and no execution acknowledgement.
        return {}


def build_policy(config):
    return GrootODEPolicy(config)
