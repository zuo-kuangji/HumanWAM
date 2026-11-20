"""Exercise HB adapter function bodies without model/VLM dependencies.

The upstream processor boundary is substituted. Actual shared image/state
augmentation, tensor layouts and raw-to-normalized ordering run on CPU.
"""

import ast
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from PIL import Image  # noqa: E402
from torchvision.transforms import v2  # noqa: E402
from torchvision.transforms.functional import pil_to_tensor, to_pil_image  # noqa: E402

from humanwam.augmentation import SampleKey, UnifiedAugmentation  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
BODY = [("left_leg", 6), ("right_leg", 6), ("waist", 3), ("left_arm", 7), ("right_arm", 7)]


def source_nodes(relative, names):
    tree = ast.parse((ROOT / relative).read_text())
    result = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert len(result) == len(names)
    return ast.Module(body=result, type_ignores=[])


def profile_startup(relative):
    """Run actual env/profile/constructor/log statements without VLM imports."""
    tree = ast.parse((ROOT / relative).read_text())
    selected = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(target, "id", None) in {"_PROFILE", "_UNIFIED"} for target in node.targets):
            selected.append(node)
        elif isinstance(node, ast.If) and any(isinstance(value, ast.Name) and value.id == "_PROFILE" for value in ast.walk(node.test)):
            selected.append(node)
        elif isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "_UNIFIED":
            body = [value for value in node.body if (
                isinstance(value, ast.Assign) and any(getattr(target, "id", None) == "_shared_aug" for target in value.targets)
            ) or (isinstance(value, ast.Expr) and isinstance(value.value, ast.Call) and getattr(value.value.func, "id", None) == "print")]
            selected.append(ast.If(test=node.test, body=body, orelse=[]))
    namespace = {"os": os, "UnifiedAugmentation": UnifiedAugmentation}
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])), str(relative), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("relative", ["hb_training/hb_groot/hb_launch.py", "hb_training/hb_psi0/hb_psi0_config.py"])
def test_actual_profile_startup_honors_seed_and_rejects_unknown_profile(relative, monkeypatch, capsys):
    monkeypatch.setenv("HB_AUG_PROFILE", "unified")
    monkeypatch.setenv("HB_AUG_SEED", "123")
    selected = profile_startup(relative)
    assert selected["_shared_aug"].seed == 123
    assert "seed=123" in capsys.readouterr().out
    raw = np.zeros((1, 29), np.float32)
    sample = SampleKey(3, 7)
    np.testing.assert_array_equal(selected["_shared_aug"].apply_state(raw, body_dims=29, sample=sample),
                                  UnifiedAugmentation(seed=123).apply_state(raw, body_dims=29, sample=sample))
    monkeypatch.setenv("HB_AUG_SEED", "-1")
    with pytest.raises(ValueError):
        profile_startup(relative)
    monkeypatch.setenv("HB_AUG_PROFILE", "typo")
    with pytest.raises(ValueError, match="Unknown HB_AUG_PROFILE"):
        profile_startup(relative)
    # Legacy selection never reads/validates the unified-only seed option.
    for legacy in (None, "hb_v2"):
        if legacy is None:
            monkeypatch.delenv("HB_AUG_PROFILE", raising=False)
        else:
            monkeypatch.setenv("HB_AUG_PROFILE", legacy)
        selected = profile_startup(relative)
        assert selected["_UNIFIED"] is False and "_shared_aug" not in selected


def groot_patch(unified=True):
    legacy = SimpleNamespace(
        augment=lambda x, rng: (x + 0.01, {"boost": "legacy"}),
        perturb_joints=lambda x, rng: np.asarray(x) + 1,
        VERSION="legacy-v2",
    )
    ns = dict(np=np, torch=torch, os=os, json=json, SampleKey=SampleKey,
              _shared_aug=UnifiedAugmentation(seed=42), _UNIFIED=unified,
              BODY_KEYS=[key for key, _ in BODY], LOG=None, A=legacy,
              _state={"count": 0}, _rng=lambda: np.random.default_rng(3))

    def original_get(self, keys, images, masks, transform, language):
        self.geometry = transform
        # Native ordering is time-major, then camera.
        stacked = torch.stack([torch.stack(images[key]) for key in keys], dim=1).flatten(0, 1)
        return self._apply_vlm_processing(stacked, language)

    def original_call(self, messages):
        content = messages[0]["content"]
        assert "hb_unified_sample" not in content.metadata  # never a model feature
        images = self._get_vlm_inputs(["ego_view", "chest"], content.images, None, "random", content.text)
        return {"images": images, "normalized_state": {key: value * 100 for key, value in content.states.items()},
                "actions": content.actions}

    ns.update(_orig_get=original_get, _orig_apply=lambda self, images, language: images,
              _orig_call=original_call, _orig_parquet=lambda self, episode: SimpleNamespace(attrs={}),
              _orig_extract=lambda frame, index, *args, **kwargs: SimpleNamespace(metadata={}))
    names = {"_load_parquet_with_identity", "_extract_with_identity", "_get_vlm_inputs", "_apply_vlm_processing", "_call"}
    exec(compile(source_nodes("hb_training/hb_groot/hb_launch.py", names), "hb_launch.py", "exec"), ns)
    proc = SimpleNamespace(training=True, eval_image_transform="eval", modality_configs={"g1": {
        "video": SimpleNamespace(delta_indices=[0])}})
    proc._get_vlm_inputs = lambda *args: ns["_get_vlm_inputs"](proc, *args)
    proc._apply_vlm_processing = lambda *args: ns["_apply_vlm_processing"](proc, *args)
    return ns, proc


def groot_content(sample):
    return SimpleNamespace(
        states={**{key: np.full((1, width), i / 10, np.float32) for i, (key, width) in enumerate(BODY)},
                "hands": np.full((1, 12), 0.5, np.float32)},
        actions={"motion_token": np.ones((16, 64), np.float32)},
        images={key: [torch.full((3, 12, 14), 80 + index * 10, dtype=torch.uint8)] for index, key in enumerate(("ego_view", "chest"))},
        metadata={} if sample is None else {"hb_unified_sample": sample},
        embodiment=SimpleNamespace(value="g1"), text="move",
    )


def test_groot_real_loader_identity_and_raw_group_order():
    ns, proc = groot_patch()
    frame = ns["_load_parquet_with_identity"](None, 9)
    sampled = ns["_extract_with_identity"](frame, 17)
    sample = sampled.metadata["hb_unified_sample"]
    assert sample == SampleKey(episode=9, frame=17, epoch=0)
    content = groot_content(sample)
    saved = copy.deepcopy(content)
    result = ns["_call"](proc, [{"content": content}])
    raw = np.concatenate([saved.states[key] for key, _ in BODY], axis=-1)
    expected = UnifiedAugmentation(seed=42).apply_state(raw, body_dims=29, sample=sample)
    actual = np.concatenate([result["normalized_state"][key] for key, _ in BODY], axis=-1)
    np.testing.assert_array_equal(actual, expected * 100)
    np.testing.assert_array_equal(result["normalized_state"]["hands"], saved.states["hands"] * 100)
    for key in content.states:
        np.testing.assert_array_equal(content.states[key], saved.states[key])
    np.testing.assert_array_equal(result["actions"]["motion_token"], saved.actions["motion_token"])
    assert proc.geometry == "eval"
    assert proc._hb_unified_sample is None
    repeated = ns["_call"](proc, [{"content": content}])
    torch.testing.assert_close(repeated["images"], result["images"], rtol=0, atol=0)
    for key in result["normalized_state"]:
        np.testing.assert_array_equal(repeated["normalized_state"][key], result["normalized_state"][key])


def test_groot_missing_identity_fails_but_validation_and_legacy_stay_available():
    ns, proc = groot_patch()
    with pytest.raises(ValueError, match="identity"):
        ns["_extract_with_identity"](SimpleNamespace(attrs={}), 0)
    with pytest.raises(ValueError, match="sample metadata"):
        ns["_call"](proc, [{"content": groot_content(None)}])
    proc.training = False
    content = groot_content(None)
    result = ns["_call"](proc, [{"content": content}])
    np.testing.assert_array_equal(result["normalized_state"]["left_leg"], content.states["left_leg"] * 100)
    ns, proc = groot_patch(unified=False)
    content = groot_content(None)
    before = content.states["left_leg"].copy()
    result = ns["_call"](proc, [{"content": content}])
    np.testing.assert_array_equal(result["normalized_state"]["left_leg"], (before + 1) * 100)


class RepackBase:
    state_keys = ["raw_state"]
    pad_state_dim = 44
    pad_action_dim = 76
    state_temporal_jitter = 0
    num_past_frames = 0

    def delta_timestamps(self, fps):
        return {"raw_state": [0]}

    def __call__(self, data, **kwargs):
        return {"states": data["raw_state"].copy(), "actions": data["raw_action"].copy(),
                "actions_mask": np.ones_like(data["raw_action"], bool), "observations": data["images"]}


class ModelBase:
    img_aug = False
    view_aug = False
    adaptive_resize = False
    resize = staticmethod(lambda: lambda image: image)
    center_crop = staticmethod(lambda: lambda image: image)

    def __call__(self, data, **kwargs):
        assert "_hb_unified_sample" not in data
        return data


def psi_patch(unified=True):
    ns = dict(np=np, torch=torch, os=os, json=json, math=__import__("math"), Any=Any,
              SampleKey=SampleKey, _UNIFIED=unified, _shared_aug=UnifiedAugmentation(seed=42),
              ps=SimpleNamespace(SonicRepackTransform=RepackBase, parse_modality_key=lambda key: (key, None)),
              pt=SimpleNamespace(Psi0ModelTransform=ModelBase), v2=v2, pil_to_tensor=pil_to_tensor,
              to_pil_image=to_pil_image, get_epoch=lambda: 4, _log_aug=lambda *args: None,
              NOISE_DEG=0.25, A=SimpleNamespace(augment=lambda x, rng: (x + 0.01, {"boost": "legacy"})),
              hb_rng=lambda: np.random.default_rng(1))
    exec(compile(source_nodes("hb_training/hb_psi0/hb_psi0_config.py", {"unified_sample", "HbSonicRepackTransform", "HbPsi0ModelTransform"}), "hb_psi0_config.py", "exec"), ns)
    return ns


def raw_psi():
    return {"raw_state": np.stack([np.full(44, value, np.float32) for value in (0.1, 0.2, 0.3)]),
            "raw_action": np.ones((16, 76), np.float32),
            "images": [Image.fromarray(np.full((12, 14, 3), value, np.uint8)) for value in (80, 90)],
            "episode_index": np.array(9, np.int64), "frame_index": np.array(17, np.int64)}


def test_psi_shared_identity_raw_state_then_field_and_model_no_metadata_leak():
    ns = psi_patch()
    repack, model = ns["HbSonicRepackTransform"](), ns["HbPsi0ModelTransform"]()
    raw = raw_psi()
    sample = SampleKey(episode=9, frame=17, epoch=4)
    first = repack(raw)
    expected = UnifiedAugmentation(seed=42).apply_state(raw["raw_state"][-1:], body_dims=29, sample=sample)
    np.testing.assert_array_equal(first["states"], expected)
    np.testing.assert_array_equal(first["states"][:, 29:], raw["raw_state"][-1:, 29:])
    np.testing.assert_array_equal(first["actions"], raw["raw_action"])
    assert first["_hb_unified_sample"] == sample
    # Simulate the native field normalizer between repack and model.
    first["states"] = first["states"] * 100
    output = model(first)
    assert "_hb_unified_sample" not in output
    np.testing.assert_array_equal(output["states"], expected * 100)
    other = model(repack(raw))
    for left, right in zip(output["observations"], other["observations"]):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
    np.testing.assert_array_equal(raw["raw_state"][-1], np.full(44, 0.3, np.float32))
    ns["get_epoch"] = lambda: 5
    changed = repack(raw)
    assert changed["_hb_unified_sample"].epoch == 5
    assert not np.array_equal(changed["states"], expected)


def test_psi_validation_needs_no_identity_and_unified_training_does():
    ns = psi_patch()
    repack, model = ns["HbSonicRepackTransform"](), ns["HbPsi0ModelTransform"]()
    raw = raw_psi()
    raw.pop("episode_index")
    raw.pop("frame_index")
    with pytest.raises(ValueError, match="episode_index"):
        repack(raw)
    clean = repack(raw, no_aug=True)
    np.testing.assert_array_equal(clean["states"], raw["raw_state"][-1:])
    assert "_hb_unified_sample" not in model(clean, no_aug=True)


def test_psi_h2_body31_preserves_every_other_state_coordinate():
    ns = psi_patch()
    repack = ns["HbSonicRepackTransform"]()
    repack.hb_body_dims, repack.pad_state_dim, repack.pad_action_dim = 31, 78, 75
    raw = raw_psi()
    raw["raw_state"] = np.zeros((3, 78), np.float32)
    raw["raw_state"][:, 31:] = np.linspace(-1, 1, 47)
    raw["raw_action"] = np.ones((16, 75), np.float32)
    result = repack(raw)
    expected = UnifiedAugmentation(seed=42).apply_state(raw["raw_state"][-1:], body_dims=31,
                                                      sample=SampleKey(9, 17, 4))
    np.testing.assert_array_equal(result["states"], expected)
    np.testing.assert_array_equal(result["states"][:, 31:], raw["raw_state"][-1:, 31:])
    np.testing.assert_array_equal(result["actions"], raw["raw_action"])


def test_act_shared_rgb_before_native_image_normalization():
    ns = psi_patch()
    ns["HbSonicRepackTransform"]  # Shared state path tested above.
    act_ns = dict(Any=Any, np=np, torch=torch, v2=v2, HP=SimpleNamespace(**ns),
                  pt=SimpleNamespace(ACTModelTransform=ModelBase))
    exec(compile(source_nodes("hb_training/hb_act/hb_act_config.py", {"HbACTModelTransform"}), "hb_act_config.py", "exec"), act_ns)
    model = act_ns["HbACTModelTransform"]()
    model.normalize = lambda: lambda images: images * 2 - 1
    data = ns["HbSonicRepackTransform"]()(raw_psi())
    output = model(data)
    clip = torch.stack([pil_to_tensor(image).float() / 255 for image in data["observations"]])[None]
    expected = UnifiedAugmentation(seed=42).apply_rgb(clip, sample=data["_hb_unified_sample"], observation_mask=[True])
    torch.testing.assert_close(output["observation.images"], expected[0] * 2 - 1, rtol=0, atol=0)
    assert set(output) == {"observation.images", "observation.state", "action", "action_is_pad"}
