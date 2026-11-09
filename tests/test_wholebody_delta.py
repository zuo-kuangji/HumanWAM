"""CPU PyTorch routing tests with actual native loss/IC/sampler method bodies.

Backbone kernels, VAE and scheduler are small CPU fixtures, not FLUX.2 model
weights. This verifies routing/autograd/serialization, NOT GPU numerical parity,
throughput, dataset correctness or successful robot behavior. No model download,
GPU operation or simulator is started. Run again with the real model preflight
before training. --native-source points to the untouched upstream source tree.
"""
from __future__ import annotations

import argparse
import ast
import copy
import importlib.util
import inspect
import logging
from pathlib import Path
import sys
import tempfile
from types import MethodType, ModuleType, SimpleNamespace
import unittest
from unittest import mock

import torch
from torch import nn
from torch.nn import functional as F


HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--native-source", type=Path)
options, remaining = parser.parse_known_args()
sys.argv = [sys.argv[0], *remaining]
if options.native_source:
    NATIVE = options.native_source / "src/imagewam/models/backbones"
else:
    NATIVE = HERE.parent / "src/imagewam/models/backbones"


class TinyScheduler:
    def __init__(self, num_train_timesteps=1000, **kwargs):
        self.num_train_timesteps = num_train_timesteps

    def sample_training_t(self, batch_size, device, dtype):
        return torch.full((batch_size,), 500., device=device, dtype=dtype)

    def add_noise(self, target, source, t):
        t = (t / self.num_train_timesteps).reshape(-1, *([1] * (target.ndim - 1)))
        return target * (1 - t) + source * t

    def training_target(self, target, source, t):
        return source - target

    def training_weight(self, t):
        return torch.ones_like(t)

    def build_inference_schedule(self, num_inference_steps, device, dtype, **kwargs):
        return (torch.linspace(1000, 0, num_inference_steps + 1, device=device, dtype=dtype)[:-1],
                torch.full((num_inference_steps,), -1 / num_inference_steps, device=device, dtype=dtype))

    def step(self, velocity, delta, current):
        return current + delta * velocity


def load_native_class(name, path, namespace, methods=None):
    tree = ast.parse(path.read_text())
    node = next(row for row in tree.body if isinstance(row, ast.ClassDef) and row.name == name)
    if methods is not None:
        node = copy.deepcopy(node)
        node.body = [row for row in node.body if isinstance(row, (ast.FunctionDef, ast.AsyncFunctionDef)) and row.name in methods]
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


# Load the actual native class/method AST without its heavyweight import graph.
for package in ("imagewam", "imagewam.models", "imagewam.models.backbones", "flux2"):
    module = ModuleType(package)
    module.__path__ = []
    sys.modules[package] = module
augmentation_spec = importlib.util.spec_from_file_location(
    "imagewam.augmentation_config", HERE.parent / "src/imagewam/augmentation_config.py")
augmentation_module = importlib.util.module_from_spec(augmentation_spec)
sys.modules[augmentation_spec.name] = augmentation_module
augmentation_spec.loader.exec_module(augmentation_module)
native_module = ModuleType("imagewam.models.backbones.imagewam")
native_module.__file__ = str(NATIVE / "imagewam.py")
namespace = native_module.__dict__
namespace.update(torch=torch, nn=nn, F=F, inspect=inspect, logger=logging.getLogger("fixture"),
                 WanContinuousFlowMatchScheduler=TinyScheduler)
ImageWAM = load_native_class("ImageWAM", NATIVE / "imagewam.py", namespace)
sys.modules[native_module.__name__] = native_module
action_module = ModuleType("imagewam.models.backbones.action_dit_flux2")
action_module.__file__ = str(NATIVE / "action_dit_flux2.py")
action_module.__dict__.update(torch=torch, nn=nn)
ActionDiTFlux2 = load_native_class("ActionDiTFlux2", NATIVE / "action_dit_flux2.py", action_module.__dict__,
                                {"pre_dit", "post_dit", "build_action_ids"})
sys.modules[action_module.__name__] = action_module
flux_module = ModuleType("flux2.model")
flux_module.timestep_embedding = lambda t, dim: t[:, None].expand(-1, dim)
sys.modules[flux_module.__name__] = flux_module
video_module = ModuleType("imagewam.models.backbones.flux2_video_expert")


class Flux2VideoExpert:
    @staticmethod
    def build_img_ids(batch_size, token_height, token_width, time_value, device, dtype):
        value = torch.zeros(batch_size, token_height * token_width, 4, device=device, dtype=dtype)
        value[..., 0] = time_value
        return value


video_module.Flux2VideoExpert = Flux2VideoExpert
sys.modules[video_module.__name__] = video_module
spec = importlib.util.spec_from_file_location("wholebody_adapter_test_target", HERE.parent / "src/imagewam/wholebody_delta.py")
adapter = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = adapter
spec.loader.exec_module(adapter)
joint_spec = importlib.util.spec_from_file_location(
    "wholebody_joint_reference_test_target", HERE.parent / "src/imagewam/wholebody_joint_reference.py")
joint_adapter = importlib.util.module_from_spec(joint_spec)
sys.modules[joint_spec.name] = joint_adapter
with mock.patch.dict(sys.modules, {"imagewam.wholebody_delta": adapter}):
    joint_spec.loader.exec_module(joint_adapter)


class PairModulation(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)

    def forward(self, value):
        return self.linear(value), None


class TinyHead(nn.Module):
    def __init__(self, action_dim=76):
        super().__init__()
        self.norm_final = nn.LayerNorm(8, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(8, 16, bias=False))
        self.linear = nn.Linear(8, action_dim, bias=False)

    def forward(self, tokens, vec):
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=-1)
        return self.linear((1 + scale[:, None]) * self.norm_final(tokens) + shift[:, None])


class TinyVideo(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(128, 128)
        self.to_action = nn.Linear(128, 8)
        self.records = []
        self.flux2_lora_enabled = False

    def pre_dit(self, x, timestep, context, context_mask, ref_image_hidden_states, **kwargs):
        self.records.append({"x": x.detach().clone(), "text": context.detach().clone(),
                             "ref": ref_image_hidden_states.detach().clone()})
        return {"tokens": {"img": torch.cat([ref_image_hidden_states, x], dim=1), "txt": context},
                "txt_len": context.shape[1], "target_len": x.shape[1], "cond_len": ref_image_hidden_states.shape[1],
                "text_mask": context_mask, "freqs": {}, "t_mod": {}}

    def post_dit(self, hidden, pre):
        return hidden["img"][:, pre["cond_len"]:pre["cond_len"] + pre["target_len"]]


class TinyMoT(nn.Module):
    block_protocol = "flux2"

    def __init__(self, video, action):
        super().__init__()
        self.mixtures = nn.ModuleDict({"video": video, "action": action})
        self.action_block = nn.Linear(8, 8)
        self.caches = []
        self.fail_action = False

    def prefill_flux2_video_cache(self, video_tokens, **kwargs):
        video = self.mixtures["video"]
        hidden = video.linear(video_tokens["img"] + video_tokens["txt"].mean(1)[:, None])
        key = video.to_action(hidden.mean(1))
        self.caches.append(key.detach().clone())
        return {"double": [{"k": key, "v": key}], "single": [], "final_video": {"img": hidden, "txt": video_tokens["txt"]}}

    def forward_action_with_video_cache(self, action_tokens, video_kv_cache, **kwargs):
        if self.fail_action:
            raise RuntimeError("fixture action failure")
        return self.action_block(action_tokens + video_kv_cache["double"][0]["k"][:, None])


def make_base(action_dim=76):
    action = ActionDiTFlux2.__new__(ActionDiTFlux2)
    nn.Module.__init__(action)
    action.action_dim, action.hidden_dim, action.max_action_horizon = action_dim, 8, 64
    action.action_encoder = nn.Linear(action_dim, 8)
    action.time_in = nn.Linear(256, 8)
    action.double_stream_modulation_img = nn.Linear(8, 8)
    action.single_stream_modulation = PairModulation()
    action.head = TinyHead(action_dim)
    video = TinyVideo()
    mot = TinyMoT(video, action)
    model = ImageWAM(video, action, mot, nn.Linear(2, 2), text_dim=128, proprio_dim=None,
        device="cpu", torch_dtype=torch.float32, stack="flux2", pack_proprio_after_text=False,
        loss_lambda_video=.5, loss_lambda_action=1.,
        action_a2a={"enabled": True, "fm_weight": 1., "ic_weight": .5, "ic_steps": 1, "ic_loss": "l1",
                    "source_noise_std": .5, "source_noise_dim_mask": [True] * action_dim},
        image_i2i={"enabled": True, "source_mode": "gaussian", "current_residual_scale": adapter.IMAGE_RESIDUAL_SCALE,
                   "variance_normalize": True, "ic_weight": 1., "ic_steps": 1, "ic_loss": "l1"},
        flux2_action_image_context={"mode": "noise_idm", "include_future_tokens": True,
                                   "knowledge_insulation": True, "reuse_image_fm_workspace": False})
    model.model_paths = {}
    # Fixture kernels only; native losses, source constructors, IC and action
    # inference loops remain the actual pinned upstream implementations.
    model._build_mot_attention_mask_flux2 = MethodType(lambda self, **kwargs: {}, model)
    model._forward_flux2_video_only = MethodType(lambda self, pre, mask:
        self.mot.prefill_flux2_video_cache(video_tokens=pre["tokens"])["final_video"], model)
    model._encode_flux2_image_tokens = MethodType(lambda self, image, time_value:
        (F.avg_pool2d(image.mean(1, keepdim=True), 16).flatten(2).transpose(1, 2).repeat(1, 1, 128),
         Flux2VideoExpert.build_img_ids(image.shape[0], 14, 28, time_value, image.device, image.dtype)), model)
    return model


def make_model(horizon=40):
    # Historical fixtures deliberately exercise the explicitly selected H40 protocol.
    return adapter.adopt_state_only_delta(make_base(), expected_action_horizon=horizon)


def sample(horizon=40, action_dim=76):
    result = {"target_latent": torch.randn(1, 392, 128), "target_img_ids": torch.zeros(1, 392, 4),
            "ref_image_latents": torch.randn(1, 392, 128), "ref_img_ids": torch.zeros(1, 392, 4),
            "text_hidden_states": torch.randn(1, 4, 128), "text_attention_mask": torch.ones(1, 4, dtype=torch.bool),
            "action": torch.randn(1, horizon, action_dim), "past_action": torch.randn(1, horizon, action_dim),
            "action_is_pad": torch.zeros(1, horizon, dtype=torch.bool),
            "action_dim_is_pad": torch.zeros(1, action_dim, dtype=torch.bool),
            "proprio": torch.randn(1, horizon, 44), "proprio_dim_is_pad": torch.zeros(1, 44, dtype=torch.bool),
            "proprio_is_pad": torch.zeros(1, horizon, dtype=torch.bool)}
    if action_dim == 47:
        # Synthetic pose observations in this fixture are all explicitly valid.
        # Recorder fallback rows need their own per-time-step orientation masks.
        result["joint_orientation_is_pad"] = torch.zeros(1, horizon, dtype=torch.bool)
        result["past_joint_orientation_is_pad"] = torch.zeros(1, horizon, dtype=torch.bool)
    return result


def has_grad(module):
    return any(param.grad is not None and bool(param.grad.abs().sum() > 0) for param in module.parameters())


class WholeBodyAdapterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)
        self.model, self.sample = make_model(), sample()

    def test_registered_projection_and_no_visual_state_or_pose_head(self):
        model = self.model
        self.assertIs(model.action_expert, model.mot.mixtures["action"])
        self.assertIsNone(model.proprio_encoder)
        self.assertFalse(hasattr(model, "pose_head"))
        state_ids = {id(param) for param in model.action_expert.state_encoder.parameters()}
        params = list(model.dit.parameters())
        self.assertTrue(state_ids <= {id(param) for param in params})
        self.assertEqual(len(params), len({id(param) for param in params}))
        self.assertTrue(any(key.endswith("state_encoder.weight") for key in model.mot.state_dict()))
        trainer_path = NATIVE / "trainer.py" if (NATIVE / "trainer.py").exists() else NATIVE.parent.parent / "trainer.py"
        tree = ast.parse(trainer_path.read_text())
        trainer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Wan22Trainer")
        policy = copy.deepcopy(next(node for node in trainer.body if isinstance(node, ast.FunctionDef) and node.name == "_apply_dit_only_train_mode"))
        policy.decorator_list = []
        namespace = {}
        exec(compile(ast.Module(body=[policy], type_ignores=[]), str(trainer_path), "exec"), namespace)
        namespace["_apply_dit_only_train_mode"](model)
        params = [param for param in model.dit.parameters() if param.requires_grad]
        self.assertEqual({id(param) for param in params}, {id(param) for param in model.parameters() if param.requires_grad})
        self.assertTrue(all(param.requires_grad for param in model.action_expert.state_encoder.parameters()))
        self.assertFalse(any(param.requires_grad for param in model.vae.parameters()))

    def test_real_native_noise_formulas(self):
        history = self.sample["past_action"]
        torch.manual_seed(91); expected = history + .5 * torch.randn_like(history)
        torch.manual_seed(91); actual = self.model._apply_action_a2a_source_noise(history)
        self.assertTrue(torch.equal(actual, expected))
        ref = self.sample["ref_image_latents"]
        scale = adapter.IMAGE_RESIDUAL_SCALE
        torch.manual_seed(92); expected = (torch.randn_like(ref) + scale * ref) / (1 + scale**2)**.5
        torch.manual_seed(92); actual = self.model._apply_image_i2i_source_noise(ref)
        self.assertTrue(torch.equal(actual, expected))

    def test_visual_inputs_are_exactly_invariant_to_current_and_future_state(self):
        changed = {**self.sample, "proprio": torch.full_like(self.sample["proprio"], 123.)}
        first, second = self.model.build_inputs_flux2(self.sample), self.model.build_inputs_flux2(changed)
        for key, value in first.items():
            if torch.is_tensor(value):
                self.assertTrue(torch.equal(value, second[key]), key)
        self.assertFalse(adapter.STATE_KEYS.intersection(first))
        with self.assertRaisesRegex(ValueError, "reached"):
            self.model._append_proprio_to_context(first["text_hidden_states"], first["text_attention_mask"], torch.zeros(1, 44))

    def test_native_three_losses_backward_independently_and_scope_is_not_live_at_yield(self):
        iterator = self.model.iter_training_losses(self.sample)
        rows = []
        for index in range(3):
            self.model.zero_grad(set_to_none=True)
            loss, metrics = next(iterator)
            self.assertIsNone(adapter._CURRENT_ACTION_STATE.get())
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            rows.append(metrics)
            self.assertEqual(has_grad(self.model.action_expert.state_encoder), index in (1, 2))
            self.assertEqual(has_grad(self.model.video_expert), index in (0, 1))
        with self.assertRaises(StopIteration):
            next(iterator)
        self.assertIn("loss_image_ic", rows[0]); self.assertIn("loss_action_fm", rows[1]); self.assertIn("action_ic_loss", rows[2])

    def test_action_fm_and_ic_gradients_do_not_enter_visual_branch(self):
        original = self.model._compute_action_loss_per_sample
        def capture(model, *args, **kwargs):
            value = original(*args, **kwargs)
            model.fixture_action_loss = value.mean()
            return value
        self.model._compute_action_loss_per_sample = MethodType(capture, self.model)
        self.model._training_loss_flux2(self.sample, include_action_ic=False)
        self.model.fixture_action_loss.backward()
        self.assertTrue(has_grad(self.model.action_expert.state_encoder))
        self.assertFalse(has_grad(self.model.video_expert))
        self.model.zero_grad(set_to_none=True)
        loss, _ = self.model._training_loss_flux2_action_ic(self.sample)
        loss.backward()
        self.assertTrue(has_grad(self.model.action_expert.state_encoder))
        self.assertFalse(has_grad(self.model.video_expert))

    def test_iterator_close_and_two_models_do_not_leak_or_mix_scope(self):
        other = make_model()
        first, second = self.model.iter_training_losses(self.sample), other.iter_training_losses(self.sample)
        next(first); self.assertIsNone(adapter._CURRENT_ACTION_STATE.get())
        next(second); self.assertIsNone(adapter._CURRENT_ACTION_STATE.get())
        first.close(); second.close(); self.assertIsNone(adapter._CURRENT_ACTION_STATE.get())
        with self.assertRaisesRegex(RuntimeError, "scope"):
            self.model.action_expert.pre_dit(torch.zeros(1, 40, 76), torch.zeros(1))
        with self.assertRaisesRegex(ValueError, "state-free model"):
            self.model._training_loss_flux2_action_ic(self.sample, action_video_workspace={"video_kv_cache": {}})

    def test_state_schema_padding_and_missing_measurement_fail_closed(self):
        for value in (None, torch.zeros(1, 43), torch.zeros(1, 2, 3, 44), torch.full((1, 44), float("nan"))):
            bad = {**self.sample, "proprio": value}
            with self.assertRaises((ValueError, RuntimeError)):
                next(self.model.iter_training_losses(bad))
            self.assertIsNone(adapter._CURRENT_ACTION_STATE.get())
        bad = {**self.sample, "proprio_dim_is_pad": torch.ones(1, 44, dtype=torch.bool)}
        with self.assertRaisesRegex(ValueError, "padding"):
            next(self.model.iter_training_losses(bad))
        measured = self.sample["proprio"].clone(); measured[:, 1:] = float("nan")
        current = self.model._current_measured_state(measured)
        self.assertTrue(torch.equal(current, measured[:, 0]))

    def test_native_inference_state_changes_only_action_and_clears_scope_on_failure(self):
        kwargs = {"prompt": None, "input_image": torch.rand(1, 3, 224, 448), "action_horizon": 40,
                  "context": self.sample["text_hidden_states"], "context_mask": self.sample["text_attention_mask"],
                  "action_init": self.sample["past_action"], "seed": 73, "rand_device": "cpu"}
        first = self.model.infer_action(**kwargs, proprio=torch.zeros(1, 44))
        cache_first = self.model.mot.caches[-1].clone()
        text_first = self.model.video_expert.records[-1]["text"].clone()
        second = self.model.infer_action(**kwargs, proprio=torch.ones(1, 44))
        self.assertFalse(torch.allclose(first["action"], second["action"]))
        self.assertTrue(torch.equal(cache_first, self.model.mot.caches[-1]))
        self.assertTrue(torch.equal(text_first, self.model.video_expert.records[-1]["text"]))
        self.assertEqual(second["state_only_action_condition_calls"], 1)
        self.assertIsNone(adapter._CURRENT_ACTION_STATE.get())
        direct = self.model.infer_action_flux2(**kwargs, proprio=torch.ones(1, 44))
        self.assertTrue(torch.equal(second["action"], direct["action"]))
        self.model.mot.fail_action = True
        with self.assertRaisesRegex(RuntimeError, "fixture action failure"):
            self.model.infer_action(**kwargs, proprio=torch.zeros(1, 44))
        self.assertIsNone(adapter._CURRENT_ACTION_STATE.get())
        for override in ({"action_init": None}, {"action_init_noise_strength": 0.}, {"action_init_noise_type": "blend"}):
            with self.assertRaises(ValueError):
                self.model.infer_action(**{**kwargs, **override}, proprio=torch.zeros(1, 44))

    def test_strict_checkpoint_roundtrip_and_rejection_before_parameter_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            self.model.save_checkpoint(path, step=7)
            other = make_model()
            payload = other.load_checkpoint(path)
            self.assertEqual(payload["step"], 7)
            for key, value in self.model.mot.state_dict().items():
                self.assertTrue(torch.equal(value, other.mot.state_dict()[key]), key)
            for kind in ("missing", "shape", "protocol", "old", "nonfinite"):
                bad = copy.deepcopy(payload)
                if kind == "missing": bad["mot"].pop("mixtures.action.state_encoder.weight")
                if kind == "shape": bad["mot"]["mixtures.action.state_encoder.weight"] = torch.zeros(8, 43)
                if kind == "protocol": bad["state_only_delta"]["visual_state_route"] = "text_token"
                if kind == "old": bad.pop("checkpoint_format")
                if kind == "nonfinite": bad["mot"]["mixtures.action.state_encoder.weight"].fill_(float("nan"))
                before = {key: value.clone() for key, value in other.mot.state_dict().items()}
                torch.save(bad, path)
                with self.assertRaises(ValueError): other.load_checkpoint(path)
                self.assertTrue(all(torch.equal(value, other.mot.state_dict()[key]) for key, value in before.items()))

    def test_new_protocol_records_measured_hands_and_exact_prior(self):
        protocol = self.model.state_only_protocol()
        self.assertEqual(protocol['action_dim'], 76)
        self.assertEqual(protocol['hand_target_time_offset'], 0)
        self.assertFalse(protocol['hand_target_is_issued_command'])
        self.assertEqual(protocol['image_prior_squared_coefficient'], .25)
        self.assertEqual(protocol['action_history_dropout_probability'], 0.)
        self.assertFalse({'dataset_id', 'dataset_episodes', 'dataset_frames',
                          'requested_training_epochs', 'training_split_mode'}.intersection(protocol))
        scale = self.model.image_i2i_current_residual_scale
        self.assertAlmostEqual(scale / (1 + scale**2)**.5, .5)
        self.assertAlmostEqual(1 / (1 + scale**2)**.5, .75**.5)

    def test_state_dropout_gates_the_whole_encoded_condition_including_bias(self):
        expert = self.model.action_expert
        with torch.no_grad():
            expert.state_encoder.bias.fill_(0.7)
        actions = torch.randn(2, 40, 76)
        time = torch.ones(2)
        current = torch.stack((torch.ones(44), torch.full((44,), 2.)))
        keep = torch.tensor([True, False])
        baseline = ActionDiTFlux2.pre_dit(expert, actions, time)["tokens"]
        with self.model.action_state_scope(current, keep):
            actual = expert.pre_dit(actions, time)["tokens"]
        feature = expert.state_encoder(current)
        self.assertTrue(torch.allclose(actual[0], baseline[0] + feature[0]))
        self.assertTrue(torch.equal(actual[1], baseline[1]))
        self.assertTrue(bool(feature[1].abs().sum() > 0))
        actual.sum().backward()
        expected_bias_gradient = torch.full_like(expert.state_encoder.bias, 40.)
        self.assertTrue(torch.equal(expert.state_encoder.bias.grad, expected_bias_gradient))

    def test_one_dataset_gate_is_shared_by_native_fm_and_endpoint_losses(self):
        self.sample["state_condition_keep"] = torch.tensor([False])
        before = self.sample["proprio"].clone()

        def run(state):
            losses = []
            self.model.zero_grad(set_to_none=True)
            torch.manual_seed(123)
            for loss, metrics in self.model.iter_training_losses(dict(self.sample, proprio=state)):
                loss.backward()
                losses.append(float(loss.detach()))
                self.assertFalse(has_grad(self.model.action_expert.state_encoder))
                self.assertIsNone(adapter._CURRENT_ACTION_STATE.get())
            return losses

        first = run(before)
        second = run(before + 9.)
        self.assertEqual(len(first), 3)
        self.assertEqual(first, second)
        self.assertTrue(torch.equal(self.sample["proprio"], before))
        self.assertNotIn("state_condition_keep", self.model._without_state(self.sample))

    def test_declared_condition_dropout_requires_a_valid_sample_gate(self):
        self.model.training_augmentation_config = {
            "enabled": True, "proprio": {"enabled": True, "state_dropout": {"probability": .2}}}
        with self.assertRaisesRegex(ValueError, "dropout gate"):
            list(self.model.iter_training_losses(self.sample))
        for keep in (torch.tensor([1.]), torch.tensor([[True]]), torch.tensor([True, False])):
            with self.assertRaises(ValueError):
                list(self.model.iter_training_losses(dict(self.sample, state_condition_keep=keep)))

    def test_inference_keeps_true_state_even_when_training_dropout_probability_is_one(self):
        self.model.training_augmentation_config = {
            "enabled": True, "proprio": {"enabled": True, "state_dropout": {"probability": 1.}}}
        kwargs = {"prompt": None, "input_image": torch.rand(1, 3, 224, 448), "action_horizon": 40,
                  "context": self.sample["text_hidden_states"], "context_mask": self.sample["text_attention_mask"],
                  "action_init": self.sample["past_action"], "seed": 73, "rand_device": "cpu"}
        first = self.model.infer_action(**kwargs, proprio=torch.zeros(1, 44))
        second = self.model.infer_action(**kwargs, proprio=torch.ones(1, 44))
        self.assertFalse(torch.allclose(first["action"], second["action"]))

    def test_training_augmentation_provenance_roundtrip_and_tamper_rejection(self):
        self.model.training_augmentation_config = augmentation_module.default_augmentation()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "augmented.pt"
            self.model.save_checkpoint(path, step=8)
            other = make_model()
            payload = other.load_checkpoint(path)
            metadata = payload["training_augmentation"]
            self.assertEqual(metadata["sha256"], augmentation_module.augmentation_sha256(metadata["config"]))
            self.assertEqual(other.loaded_training_augmentation, metadata)
            self.assertEqual(other.state_only_protocol(), self.model.state_only_protocol())
            # The record is provenance, not an inference transform/controller.
            self.assertFalse(hasattr(other, "augmentation"))
            self.assertFalse(hasattr(other, "training_augmentation_config"))
            resaved = Path(directory) / "resaved.pt"
            other.save_checkpoint(resaved)
            self.assertEqual(torch.load(resaved, weights_only=False)["training_augmentation"], metadata)
            other.loaded_training_augmentation["config"]["seed"] += 1
            with self.assertRaisesRegex(ValueError, "digest"):
                other.save_checkpoint(Path(directory) / "invalid-resave.pt")
            other.loaded_training_augmentation = copy.deepcopy(metadata)
            for kind in ("digest", "config", "schema", "partial"):
                bad = copy.deepcopy(payload)
                if kind == "digest": bad["training_augmentation"]["sha256"] = "0" * 64
                if kind == "config": bad["training_augmentation"]["config"]["seed"] += 1
                if kind == "schema": bad["training_augmentation"]["config"]["schema"] = "unknown"
                if kind == "partial": bad["training_augmentation"]["config"].pop("visual")
                torch.save(bad, path)
                before = {key: value.clone() for key, value in other.mot.state_dict().items()}
                with self.assertRaises(ValueError): other.load_checkpoint(path)
                self.assertTrue(all(torch.equal(value, other.mot.state_dict()[key]) for key, value in before.items()))
            legacy = Path(directory) / "legacy.pt"
            make_model().save_checkpoint(legacy)
            other.load_checkpoint(legacy)
            other.save_checkpoint(resaved)
            self.assertNotIn("training_augmentation", torch.load(resaved, weights_only=False))

    def test_legacy_checkpoint_without_augmentation_remains_loadable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            self.model.save_checkpoint(path)
            other = make_model()
            payload = other.load_checkpoint(path)
            self.assertNotIn("training_augmentation", payload)
            self.assertIsNone(other.loaded_training_augmentation)
            self.assertFalse(hasattr(other, "training_augmentation_config"))

    def test_v1_augmentation_checkpoint_provenance_retains_its_original_digest(self):
        legacy = augmentation_module.normalize_augmentation({"schema": augmentation_module.LEGACY_SCHEMA})
        self.model.training_augmentation_config = legacy
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v1-provenance.pt"
            self.model.save_checkpoint(path)
            other = make_model()
            payload = other.load_checkpoint(path)
            self.assertEqual(payload["training_augmentation"]["config"], legacy)
            self.assertEqual(payload["training_augmentation"]["sha256"], augmentation_module.augmentation_sha256(legacy))
            self.assertFalse(hasattr(other, "training_augmentation_config"))

    def test_legacy_prior_and_action_dim_fail_closed(self):
        self.model.image_i2i_current_residual_scale = 1.
        with self.assertRaisesRegex(ValueError, 'current-image'):
            self.model.validate_state_only_route()
        self.model.image_i2i_current_residual_scale = adapter.IMAGE_RESIDUAL_SCALE
        self.model.action_expert.action_dim = 78
        with self.assertRaisesRegex(ValueError, 'dimensions|action76'):
            self.model.validate_state_only_route()



class TinyPartialAction(nn.Module):
    """120-key fixture: exactly114 shared base tensors plus six new projections."""
    def __init__(self, action_dim=76):
        super().__init__()
        self.action_dim, self.action_state_dim = action_dim, 44
        self.action_encoder = nn.Linear(action_dim, 2)
        self.state_encoder = nn.Linear(44, 2)
        self.head = nn.Module()
        self.head.linear = nn.Linear(2, action_dim, bias=False)
        self.head.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(2, 4, bias=False))
        self.time_in = nn.ParameterDict({
            f"p{index:03d}": nn.Parameter(torch.randn(2, 2)) for index in range(114)
        })


def initializer_fixture(action_dim=76):
    expert = TinyPartialAction(action_dim)
    model = SimpleNamespace(action_expert=expert, expected_action_dim=action_dim, state_only_init_seed=42)
    state = {key: torch.full_like(value, index / 113)
             for index, (key, value) in enumerate(expert.state_dict().items())
             if key not in adapter.PARTIAL_NEW_ACTION_KEYS}
    metadata = {**adapter.PARTIAL_INITIALIZER_META,
                "source": "/fixture/flux-2-klein-base-4b.safetensors"}
    return model, {"state_dict": state, "meta": metadata}


class BaseInitializerTests(unittest.TestCase):
    def test_actual_float32_to_bfloat16_path_is_explicit_and_verified(self):
        model, payload = initializer_fixture()
        model.action_expert.to(dtype=torch.bfloat16)
        before = {key: value.clone() for key, value in model.action_expert.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "action_base_init.pt"
            torch.save(payload, path)
            expected = adapter.file_sha256(path)
            audit = adapter.load_compatible_action_initializer(model, path, expected_sha256=expected)
        actual = model.action_expert.state_dict()
        for key, value in payload["state_dict"].items():
            self.assertTrue(adapter._byte_equal(actual[key], value.to(torch.bfloat16)), key)
        for key in adapter.PARTIAL_NEW_ACTION_KEYS:
            self.assertTrue(adapter._byte_equal(actual[key], before[key]), key)
        self.assertEqual(len(audit["dtype_conversions"]), 114)
        self.assertTrue(audit["shared_tensors_copied_bitwise_after_dtype_conversion"])
        self.assertEqual(audit['source_sha256'], expected)
        self.assertEqual(audit['expected_sha256'], expected)

    def test_partial114_exact_copy_and_new76_state44_projections_are_preserved(self):
        model, payload = initializer_fixture()
        before = {key: value.clone() for key, value in model.action_expert.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "action_base_init.pt"
            torch.save(payload, path)
            expected = adapter.file_sha256(path)
            audit = adapter.load_compatible_action_initializer(model, path, expected_sha256=expected)
        actual = model.action_expert.state_dict()
        for key, value in payload["state_dict"].items():
            self.assertTrue(adapter._byte_equal(actual[key], value), key)
        for key in adapter.PARTIAL_NEW_ACTION_KEYS:
            self.assertTrue(adapter._byte_equal(actual[key], before[key]), key)
        self.assertEqual(audit["target_action_dim"], 76)
        self.assertEqual(audit["retained_tensor_count"], 114)
        self.assertEqual(audit["newly_initialized"], sorted(adapter.PARTIAL_NEW_ACTION_KEYS))
        self.assertEqual(audit["source_format"], "flux2_shared_partial_initializer_v1")
        self.assertFalse(audit["source_is_task_trained"])
        self.assertFalse(audit["parent_task_checkpoint_loaded"])
        self.assertFalse(audit["optimizer_loaded"])
        self.assertTrue(audit["new_projections_unchanged"])

    def test_wrong_unpinned_digest_is_rejected_before_loading(self):
        model, payload = initializer_fixture()
        before = {key: value.clone() for key, value in model.action_expert.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.pt"
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "SHA256"):
                adapter.load_compatible_action_initializer(model, path, expected_sha256='0' * 64)
        self.assertTrue(all(torch.equal(value, model.action_expert.state_dict()[key])
                            for key, value in before.items()))

    def test_expected_digest_is_required_and_validated_before_loading(self):
        model, _ = initializer_fixture()
        with self.assertRaises(TypeError):
            adapter.load_compatible_action_initializer(model, 'not-present.pt')
        for value in (None, '', 'a' * 63, 'a' * 65, 'g' * 64, 123):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'SHA256'):
                adapter.load_compatible_action_initializer(model, 'not-present.pt', expected_sha256=value)
        self.assertEqual(adapter.validate_expected_sha256('AB' * 32), 'ab' * 32)

    def test_corrupt_or_trained_initializer_fails_without_partial_parameter_copy(self):
        for case in ("missing", "unexpected", "shape", "dtype", "nan", "metadata", "trained", "new_projection"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                model, payload = initializer_fixture()
                before = {key: value.clone() for key, value in model.action_expert.state_dict().items()}
                key = next(iter(payload["state_dict"]))
                if case == "missing":
                    payload["state_dict"].pop(key)
                elif case == "unexpected":
                    payload["state_dict"]["time_in.unexpected"] = torch.zeros(2, 2)
                elif case == "shape":
                    payload["state_dict"][key] = torch.zeros(2, 3)
                elif case == "dtype":
                    payload["state_dict"][key] = payload["state_dict"][key].double()
                elif case == "nan":
                    payload["state_dict"][key].fill_(float("nan"))
                elif case == "metadata":
                    payload["meta"]["source"] = "/fixture/task_final.pt"
                elif case == "trained":
                    payload = {"mot": payload["state_dict"], "optimizer": {}}
                else:
                    payload["state_dict"].pop(key)
                    payload["state_dict"]["action_encoder.weight"] = before["action_encoder.weight"].clone()
                path = Path(directory) / "candidate.pt"
                torch.save(payload, path)
                with self.assertRaises(ValueError):
                    adapter.load_compatible_action_initializer(model, path, expected_sha256=adapter.file_sha256(path))
                self.assertTrue(all(torch.equal(value, model.action_expert.state_dict()[name])
                                    for name, value in before.items()), case)


class H16AdapterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)
        self.model, self.sample = make_model(16), sample(16)

    test_real_native_noise_formulas = WholeBodyAdapterTests.test_real_native_noise_formulas
    test_visual_inputs_are_exactly_invariant_to_current_and_future_state = WholeBodyAdapterTests.test_visual_inputs_are_exactly_invariant_to_current_and_future_state
    test_native_three_losses_backward_independently_and_scope_is_not_live_at_yield = WholeBodyAdapterTests.test_native_three_losses_backward_independently_and_scope_is_not_live_at_yield
    test_action_fm_and_ic_gradients_do_not_enter_visual_branch = WholeBodyAdapterTests.test_action_fm_and_ic_gradients_do_not_enter_visual_branch

    def test_native_h16_inference_defaults_match_and_legacy_shape_is_rejected(self):
        kwargs = dict(prompt=None, input_image=torch.rand(1, 3, 224, 448),
            context=self.sample["text_hidden_states"], context_mask=self.sample["text_attention_mask"],
            action_init=self.sample["past_action"], seed=73, rand_device="cpu", proprio=torch.ones(1,44))
        result = self.model.infer_action(**kwargs)
        self.assertEqual(tuple(result["action"].shape), (16,76))
        self.assertEqual(self.model.state_only_protocol()["hand_target_offsets"], [0,15])
        for changes in [dict(action_horizon=40),dict(action_init=torch.zeros(1,40,76))]:
            with self.assertRaises(ValueError): self.model.infer_action(**dict(kwargs,**changes))

    def test_checkpoint_protocol_rejects_cross_horizon_before_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"checkpoint.pt"
            self.model.save_checkpoint(path,step=1)
            target=make_model(40)
            before={k:v.clone() for k,v in target.mot.state_dict().items()}
            with self.assertRaisesRegex(ValueError,"protocol"):
                target.load_checkpoint(path)
            self.assertTrue(all(torch.equal(v,target.mot.state_dict()[k]) for k,v in before.items()))
            make_model(16).load_checkpoint(path)
            target.save_checkpoint(path,step=2)
            with self.assertRaisesRegex(ValueError,"protocol"): self.model.load_checkpoint(path)

    def test_training_window_mismatch_fails_before_visual_loss(self):
        with self.assertRaisesRegex(ValueError,"configured"):
            self.model.build_inputs_flux2(sample(40))


def make_joint_model(action_dim=41, body_target="wbc", *,
                     statistics_sha256="a"*64, instruction_sha256="b"*64):
    return adapter._adopt_state_only_delta(
        make_base(action_dim), expected_action_dim=action_dim, expected_action_horizon=16,
        state_only_schema_id=joint_adapter.schema_id(action_dim, body_target),
        route_metadata={"statistics_sha256":statistics_sha256,"instruction_sha256":instruction_sha256},
        route_class=joint_adapter.JointReferenceDeltaImageWAM)


class Joint41NativeAdapterTests(unittest.TestCase):
    """Actual native losses and sampler on CPU fixtures, not pretrained/GPU proof."""
    action_dim = 41

    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)
        self.model = make_joint_model(self.action_dim)
        self.sample = sample(16, action_dim=self.action_dim)

    test_real_native_noise_formulas = WholeBodyAdapterTests.test_real_native_noise_formulas
    test_visual_inputs_are_exactly_invariant_to_current_and_future_state = WholeBodyAdapterTests.test_visual_inputs_are_exactly_invariant_to_current_and_future_state
    test_native_three_losses_backward_independently_and_scope_is_not_live_at_yield = WholeBodyAdapterTests.test_native_three_losses_backward_independently_and_scope_is_not_live_at_yield
    test_action_fm_and_ic_gradients_do_not_enter_visual_branch = WholeBodyAdapterTests.test_action_fm_and_ic_gradients_do_not_enter_visual_branch
    test_state_schema_padding_and_missing_measurement_fail_closed = WholeBodyAdapterTests.test_state_schema_padding_and_missing_measurement_fail_closed

    def inference_inputs(self):
        return dict(prompt=None, input_image=torch.rand(1, 3, 224, 448),
            context=self.sample["text_hidden_states"], context_mask=self.sample["text_attention_mask"],
            action_init=self.sample["past_action"], seed=73, rand_device="cpu")

    def test_native_h16_sampler_uses_current_state_only_and_exact_action_dim(self):
        inputs = self.inference_inputs()
        first = self.model.infer_action(**inputs, proprio=torch.zeros(1, 44))
        visual = self.model.mot.caches[-1].clone()
        text = self.model.video_expert.records[-1]["text"].clone()
        second = self.model.infer_action(**inputs, proprio=torch.ones(1, 44))
        self.assertEqual(tuple(first["action"].shape), (16, self.action_dim))
        self.assertTrue(torch.isfinite(first["action"]).all())
        self.assertFalse(torch.allclose(first["action"], second["action"]))
        self.assertTrue(torch.equal(visual, self.model.mot.caches[-1]))
        self.assertTrue(torch.equal(text, self.model.video_expert.records[-1]["text"]))
        self.assertEqual(second["state_only_action_dim"], self.action_dim)
        self.assertGreaterEqual(second["state_only_action_condition_calls"], 1)
        self.assertIsNone(adapter._CURRENT_ACTION_STATE.get())
        for changes in [dict(action_horizon=40), dict(action_init=torch.zeros(1,16,76)),
                        dict(action_init=torch.zeros(1,40,self.action_dim))]:
            with self.subTest(changes=list(changes)), self.assertRaises(ValueError):
                self.model.infer_action(**dict(inputs, **changes), proprio=torch.ones(1,44))

    def test_future_measured_state_is_ignored_and_current_measurement_is_detached(self):
        changed = copy.deepcopy(self.sample)
        changed["proprio"][:, 1:] = float("nan")
        changed["proprio"].requires_grad_()
        iterator = self.model.iter_training_losses(changed)
        for _ in range(3):
            loss, _ = next(iterator)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertIsNone(changed["proprio"].grad)
        iterator.close()
        self.assertTrue(has_grad(self.model.action_expert.state_encoder))
        self.assertIsNone(adapter._CURRENT_ACTION_STATE.get())

    def test_protocol_identifies_joint_labels_orientation_and_timebase(self):
        protocol = self.model.state_only_protocol()
        self.assertEqual(protocol["action_dim"], self.action_dim)
        self.assertEqual(protocol["schema_id"], joint_adapter.schema_id(self.action_dim, "wbc"))
        self.assertEqual(protocol["statistics_sha256"], "a"*64)
        self.assertEqual(protocol["instruction_sha256"], "b"*64)
        self.assertEqual(protocol["format"], f"wholebody_joint_reference{self.action_dim}_p25_flux2_v1")
        self.assertEqual(protocol["hand_target_offsets"], [0, 30])
        self.assertEqual(protocol["future_action_frame_offsets"], list(range(0,32,2)))
        self.assertEqual(protocol["past_action_frame_offsets"], list(range(-32,0,2)))
        self.assertEqual(protocol["future_image_offset"], 32)
        self.assertEqual(protocol["sonic_input_protocol"], "v1_joint_reference_not_v4_motion_token")
        self.assertEqual(protocol["visual_state_route"], "none")
        self.assertEqual(protocol["text_state_route"], "none")
        self.assertFalse(protocol["action_to_visual_gradients"])
        self.assertEqual(protocol["root_orientation_source"],
            "external_required" if self.action_dim == 41 else "predicted_teleop_body_quat_w")

    def test_shared_initializer_preserves_new_joint_projections_and_records_dimensions(self):
        model, payload = initializer_fixture(self.action_dim)
        before = {key: value.clone() for key,value in model.action_expert.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "action_base_init.pt"
            torch.save(payload,path)
            audit = adapter.load_compatible_action_initializer(
                model,path,expected_sha256=adapter.file_sha256(path))
        for key,value in payload["state_dict"].items():
            self.assertTrue(adapter._byte_equal(model.action_expert.state_dict()[key],value),key)
        for key in adapter.PARTIAL_NEW_ACTION_KEYS:
            self.assertTrue(adapter._byte_equal(model.action_expert.state_dict()[key],before[key]),key)
        self.assertEqual(audit["target_action_dim"],self.action_dim)
        self.assertEqual(audit["retained_tensor_count"],114)
        self.assertEqual(audit["new_parameter_shapes"]["action_encoder.weight"],[2,self.action_dim])
        self.assertEqual(audit["new_parameter_shapes"]["head.linear.weight"],[self.action_dim,2])
        self.assertFalse(audit["source_is_task_trained"])
        self.assertFalse(audit["optimizer_loaded"])

    def test_task_statistics_and_instruction_hashes_are_mandatory(self):
        for changes in ({"statistics_sha256":None}, {"instruction_sha256":None},
                        {"statistics_sha256":"a"*63}, {"instruction_sha256":"g"*64}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError,"SHA256|statistics|instruction|metadata"):
                make_joint_model(self.action_dim, **changes)
        with self.assertRaisesRegex(ValueError,"SHA256|statistics|instruction|metadata"):
            adapter._adopt_state_only_delta(make_base(self.action_dim), expected_action_dim=self.action_dim,
                expected_action_horizon=16, state_only_schema_id=joint_adapter.schema_id(self.action_dim,"wbc"),
                route_class=joint_adapter.JointReferenceDeltaImageWAM)

    def test_joint_factory_validates_task_hashes_before_constructing_backbone(self):
        with mock.patch.object(joint_adapter,"_create_state_only_flux2") as create:
            for changes in ({"statistics_sha256":None}, {"instruction_sha256":"wrong"}):
                kwargs = {"statistics_sha256":"a"*64,"instruction_sha256":"b"*64,**changes}
                with self.subTest(changes=changes), self.assertRaisesRegex(ValueError,"SHA256"):
                    joint_adapter.create_imagewam_flux2_wholebody_joint_reference(
                        action_initializer_sha256="c"*64, expected_action_dim=self.action_dim,
                        orientation_target="external" if self.action_dim==41 else "predicted_rotation6d",
                        **kwargs)
            create.assert_not_called()

    def test_checkpoint_roundtrip_and_cross_schema_rejection_precedes_parameter_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "joint.pt"
            self.model.save_checkpoint(path, step=9)
            same = make_joint_model(self.action_dim)
            payload = same.load_checkpoint(path)
            self.assertEqual(payload["step"], 9)
            for key, value in self.model.mot.state_dict().items():
                self.assertTrue(torch.equal(value, same.mot.state_dict()[key]), key)
            others = [make_joint_model(self.action_dim, "measured"),
                      make_joint_model(47 if self.action_dim == 41 else 41), make_model(16),
                      make_joint_model(self.action_dim, statistics_sha256="c"*64),
                      make_joint_model(self.action_dim, instruction_sha256="d"*64)]
            for target in others:
                before = {key:value.clone() for key,value in target.mot.state_dict().items()}
                with self.subTest(schema=target.state_only_schema_id), self.assertRaisesRegex(ValueError, "schema|protocol"):
                    target.load_checkpoint(path)
                self.assertTrue(all(torch.equal(value,target.mot.state_dict()[key]) for key,value in before.items()))
            legacy = make_model(16)
            legacy.save_checkpoint(path, step=10)
            with self.assertRaisesRegex(ValueError, "schema|protocol"):
                self.model.load_checkpoint(path)

    def test_legacy_public_adoption_and_factory_cannot_masquerade_as_joint(self):
        base = make_base(self.action_dim)
        with self.assertRaisesRegex(ValueError, "action76"):
            adapter.adopt_state_only_delta(base, expected_action_dim=self.action_dim,
                state_only_schema_id=joint_adapter.schema_id(self.action_dim,"wbc"))
        self.assertIs(type(base), ImageWAM)
        with self.assertRaisesRegex(ValueError, "action76"):
            adapter.create_imagewam_flux2_wholebody_delta(action_initializer_sha256="a"*64,
                expected_action_dim=self.action_dim,
                state_only_schema_id=joint_adapter.schema_id(self.action_dim,"wbc"))
        self.model.mot.fail_action = True
        with self.assertRaisesRegex(RuntimeError, "fixture action failure"):
            self.model.infer_action(**self.inference_inputs(), proprio=torch.zeros(1,44))
        self.assertIsNone(adapter._CURRENT_ACTION_STATE.get())


class Joint47NativeAdapterTests(Joint41NativeAdapterTests):
    action_dim = 47

    def invalid_pose_sample(self):
        result = copy.deepcopy(self.sample)
        result["joint_orientation_is_pad"][:, [2,7]] = True
        result["past_joint_orientation_is_pad"][:, [1,9]] = True
        result["action"][:, [2,7], 41:] = 0
        result["past_action"][:, [1,9], 41:] = 0
        return result

    def test_orientation_validity_is_required_and_invalid_targets_history_are_zero(self):
        for field in ("joint_orientation_is_pad","past_joint_orientation_is_pad"):
            bad = copy.deepcopy(self.sample)
            bad.pop(field)
            with self.subTest(missing=field), self.assertRaisesRegex(ValueError,"mask|orientation|valid"):
                self.model.build_inputs_flux2(bad)
        for field,mask in (("action","joint_orientation_is_pad"),
                           ("past_action","past_joint_orientation_is_pad")):
            bad = self.invalid_pose_sample()
            index = int(torch.where(bad[mask][0])[0][0])
            bad[field][0,index,41] = 1
            with self.subTest(nonzero=field), self.assertRaisesRegex(ValueError,"zero|invalid"):
                self.model.build_inputs_flux2(bad)
        bad = self.invalid_pose_sample()
        inputs = self.model.build_inputs_flux2(bad)
        self.assertEqual(tuple(inputs["action_dim_is_pad"].shape),(1,16,47))
        self.assertFalse(inputs["action_dim_is_pad"][:,:,:41].any())
        self.assertTrue(inputs["action_dim_is_pad"][0,2,41:].all())
        self.assertFalse(inputs["action_dim_is_pad"][0,3,41:].any())
        self.assertEqual(tuple(bad["action_dim_is_pad"].shape),(1,47))

    def test_real_native_fm_and_ic_ignore_only_invalid_orientation_outputs(self):
        bad = self.invalid_pose_sample()
        for name in ("fm","ic"):
            self.model.zero_grad(set_to_none=True)
            method = "_compute_action_loss_per_sample" if name=="fm" else "_compute_action_ic_loss_per_sample"
            original = getattr(self.model,method)
            captured = {}
            def capture(model,*args,_original=original,_captured=captured,**kwargs):
                pred = kwargs["pred_action"]
                pred.retain_grad()
                loss = _original(*args,**kwargs)
                _captured.update(pred=pred,loss=loss.mean(),mask=kwargs["action_dim_is_pad"])
                return loss
            setattr(self.model,method,MethodType(capture,self.model))
            try:
                if name=="fm":
                    self.model._training_loss_flux2(bad,include_action_ic=False)
                else:
                    self.model._training_loss_flux2_action_ic(bad)
                self.assertTrue(torch.isfinite(captured["loss"]))
                captured["loss"].backward()
                gradient = captured["pred"].grad
                self.assertTrue(torch.equal(gradient[0,[2,7],41:],torch.zeros(2,6)))
                self.assertGreater(float(gradient[0,[2,7],:41].abs().sum()),0.)
                self.assertGreater(float(gradient[0,3,41:].abs().sum()),0.)
                self.assertTrue(has_grad(self.model.action_expert.state_encoder))
                self.assertFalse(has_grad(self.model.video_expert))
                self.assertIsNone(adapter._CURRENT_ACTION_STATE.get())
            finally:
                setattr(self.model,method,original)

    def test_mask_reduction_averages_valid_dimensions_then_tokens(self):
        # A global sum/count reduction would differ because token0 has41 valid
        # dimensions and token1 has47. Padded token2 must contribute nothing.
        pred = torch.ones(1,3,47,requires_grad=True)
        with torch.no_grad():
            pred[0,0,:41] = 2
            pred[0,0,41:] = 1000
            pred[0,2] = 3000
        target = torch.zeros_like(pred)
        dim_pad = torch.zeros(1,3,47,dtype=torch.bool)
        dim_pad[0,0,41:] = True
        time_pad = torch.tensor([[False,False,True]])
        fm = self.model._compute_action_loss_per_sample(pred,target,time_pad,dim_pad)
        ic = self.model._compute_action_ic_loss_per_sample(pred,target,time_pad,dim_pad)
        self.assertEqual(tuple(fm.shape),(1,))
        self.assertAlmostEqual(float(fm.detach()),2.5)
        self.assertAlmostEqual(float(ic.detach()),1.5)
        (fm+ic).sum().backward()
        self.assertTrue(torch.equal(pred.grad[0,0,41:],torch.zeros(6)))
        self.assertTrue(torch.equal(pred.grad[0,2],torch.zeros(47)))


def make_joint_model_v2(*, source_contract=None):
    import json
    # This file deliberately stubs the native imagewam package. Load the pure
    # contract by its actual path instead of relying on that stub's __path__.
    name="imagewam.joint_reference_contract"
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,HERE.parent/"src/imagewam/joint_reference_contract.py")
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    make_joint_recipe=sys.modules[name].make_joint_recipe
    if source_contract is None:
        source_contract=json.loads((HERE.parent/"configs/wholebody/source_contracts/g1_existing_measured_v1.json").read_text())
    return adapter._adopt_state_only_delta(make_base(47),expected_action_dim=47,expected_action_horizon=40,
        state_only_schema_id=joint_adapter.schema_id(47,"measured","v2"),
        route_metadata={"statistics_sha256":"a"*64,"instruction_sha256":"b"*64,
                        "joint_recipe":make_joint_recipe(source_contract)},
        route_class=joint_adapter.JointReferenceDeltaImageWAM)


class Joint47LookaheadV2Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42);torch.set_num_threads(1)
        self.model=make_joint_model_v2()
        self.sample=sample(40,47)
        self.sample["past_action"]=self.sample["past_action"][:,:16].clone()
        self.sample["past_joint_orientation_is_pad"]=torch.zeros(1,16,dtype=torch.bool)
        self.sample["past_action_is_pad"]=torch.zeros(1,16,dtype=torch.bool)

    invalid_pose_sample=Joint47NativeAdapterTests.invalid_pose_sample
    test_three_native_losses_with_state_only_and_stop_gradient=WholeBodyAdapterTests.test_native_three_losses_backward_independently_and_scope_is_not_live_at_yield
    test_action_FM_and_IC_do_not_update_visual=WholeBodyAdapterTests.test_action_fm_and_ic_gradients_do_not_enter_visual_branch
    test_invalid_orientation_only_has_zero_gradients=Joint47NativeAdapterTests.test_real_native_fm_and_ic_ignore_only_invalid_orientation_outputs
    test_visual_inputs_invariant_to_numeric_current_and_future_state=WholeBodyAdapterTests.test_visual_inputs_are_exactly_invariant_to_current_and_future_state

    def test_real_history_is_not_a_future_trajectory_and_only_expands_as_source(self):
        source=self.model._expanded_history(self.sample["past_action"])
        self.assertEqual(tuple(source.shape),(1,40,47))
        self.assertTrue(torch.equal(source[:,:16],self.sample["past_action"]))
        self.assertTrue(torch.equal(source[:,16:],self.sample["past_action"][:,-1:].expand(1,24,47)))
        before=self.sample["past_action"].clone()
        self.model.build_inputs_flux2(self.sample)
        self.assertTrue(torch.equal(before,self.sample["past_action"]))
        for length in (32,40):
            with self.assertRaisesRegex(ValueError,"16 real"):
                self.model._expanded_history(torch.zeros(1,length,47))

    def test_native_sampler_outputs40_but_declares_execution16(self):
        output=self.model.infer_action(prompt=None,input_image=torch.rand(1,3,224,448),
            context=self.sample["text_hidden_states"],context_mask=self.sample["text_attention_mask"],
            action_init=self.sample["past_action"],proprio=torch.zeros(1,44),seed=73,rand_device="cpu")
        self.assertEqual(tuple(output["action"].shape),(40,47))
        self.assertEqual(output["execution_horizon"],16)
        self.assertEqual(output["lookahead_tail_horizon"],24)
        self.assertEqual(output["orientation_prediction_frame"],"current_heading_relative")

    def test_protocol_pins_relative_heading_and_exact_source_provenance(self):
        protocol=self.model.state_only_protocol()
        self.assertEqual(protocol["prediction_horizon"],40)
        self.assertEqual(protocol["execution_horizon"],16)
        self.assertEqual(protocol["future_image_offset"],32)
        self.assertEqual(protocol["future_action_frame_offsets"],list(range(0,80,2)))
        self.assertEqual(protocol["past_action_frame_offsets"],list(range(-32,0,2)))
        self.assertEqual(protocol["orientation_prediction_frame"],"current_heading_relative")
        self.assertEqual(protocol["reference_source_contract"]["label_semantics"],"reference_from_demonstration")
        self.assertFalse(protocol["source_provenance_deployment_ready"])
        self.model.joint_recipe["execution_horizon"]=40
        with self.assertRaisesRegex(ValueError,"immutable|recipe"):
            self.model.validate_state_only_route()

    def test_checkpoint_rejects_legacy_joint_and_motion_and_changed_provenance(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"joint_v2.pt";self.model.save_checkpoint(path,step=5)
            make_joint_model_v2().load_checkpoint(path)
            contract=json.loads((HERE.parent/"configs/wholebody/source_contracts/g1_existing_measured_v1.json").read_text())
            contract["provenance"]["revision"]+=" alternative"
            for target in [make_joint_model(47),make_model(40),make_joint_model_v2(source_contract=contract)]:
                before={k:v.clone() for k,v in target.mot.state_dict().items()}
                with self.assertRaises(ValueError):target.load_checkpoint(path)
                for key,value in before.items():self.assertTrue(torch.equal(value,target.mot.state_dict()[key]))

    @unittest.skipUnless(importlib.util.find_spec("hydra"),"Hydra required for native factory regression")
    def test_real_hydra_factory_receives_nested_contract_and_recipe_without_gpu(self):
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        helper=sys.modules["imagewam.joint_reference_contract"]
        contract=self.model.joint_recipe["reference_source_contract"]
        cfg=OmegaConf.create({"_target_":joint_adapter.__name__+".create_imagewam_flux2_wholebody_joint_reference",
            "action_initializer_sha256":"c"*64,"statistics_sha256":"a"*64,"instruction_sha256":"b"*64,
            "body_target":"measured","orientation_target":"predicted_rotation6d","expected_action_dim":47,
            "expected_action_horizon":40,"recipe_version":"v2","source_contract":contract,
            "source_contract_sha256":helper.source_contract_sha256(contract),
            "joint_recipe":helper.make_joint_recipe(contract)})
        with mock.patch.object(joint_adapter,"_create_state_only_flux2",side_effect=lambda **kw:kw) as create:
            result=instantiate(cfg)
        self.assertEqual(result["route_metadata"]["joint_recipe"],helper.make_joint_recipe(contract))
        self.assertEqual(result["expected_action_horizon"],40)
        create.assert_called_once()


class FreshFactoryTests(unittest.TestCase):
    def test_parent_or_continuation_arguments_are_always_rejected(self):
        runtime = ModuleType("imagewam.runtime")
        runtime.create_imagewam_flux2_klein = mock.Mock(side_effect=AssertionError("must not build a model"))
        with mock.patch.dict(sys.modules, {"imagewam.runtime": runtime}):
            for name in ("parent_checkpoint", "additional_training_epochs", "resume", "checkpoint", "pretrained_checkpoint"):
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "rejects"):
                    adapter.create_imagewam_flux2_wholebody_delta(action_initializer_sha256='a' * 64, **{name: None})
        runtime.create_imagewam_flux2_klein.assert_not_called()

    def test_factory_expected_digest_is_required_and_invalid_values_fail_before_build(self):
        runtime = ModuleType("imagewam.runtime")
        runtime.create_imagewam_flux2_klein = mock.Mock(side_effect=AssertionError("must not build a model"))
        with mock.patch.dict(sys.modules, {"imagewam.runtime": runtime}):
            with self.assertRaises(TypeError):
                adapter.create_imagewam_flux2_wholebody_delta()
            for value in (None, '', 'a' * 63, 'g' * 64):
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'SHA256'):
                    adapter.create_imagewam_flux2_wholebody_delta(action_initializer_sha256=value)
        runtime.create_imagewam_flux2_klein.assert_not_called()

    def test_base_factory_receives_no_numeric_visual_state_or_task_action_loader(self):
        runtime = ModuleType("imagewam.runtime")
        runtime.create_imagewam_flux2_klein = mock.Mock(side_effect=lambda **kwargs: make_base())
        def pinned_hash(path):
            path = Path(path)
            if path.name in adapter.UPSTREAM_SHA256:
                return adapter.UPSTREAM_SHA256[path.name]
            return 'a' * 64
        with tempfile.TemporaryDirectory() as directory:
            initializer = Path(directory) / "action_base_init.pt"
            initializer.write_bytes(b"synthetic fixture; no pretrained tensor content")
            with mock.patch.dict(sys.modules, {"imagewam.runtime": runtime}), \
                    mock.patch.object(adapter, "file_sha256", side_effect=pinned_hash), \
                    mock.patch.object(adapter, "load_compatible_action_initializer", return_value={}) as load:
                model = adapter.create_imagewam_flux2_wholebody_delta(
                    action_initializer_sha256='a' * 64,
                    variant="klein-base-4b", flux2_model_path="/fixture/flux-2-klein-base-4b.safetensors",
                    action_dit_config={"action_dim": 76}, proprio_dim=44, pack_proprio_after_text=False,
                    action_dit_pretrained_path=str(initializer))
            passed = runtime.create_imagewam_flux2_klein.call_args.kwargs
            self.assertIsNone(passed["proprio_dim"])
            self.assertFalse(passed["pack_proprio_after_text"])
            self.assertIsNone(passed["action_dit_pretrained_path"])
            load.assert_called_once_with(model, str(initializer), expected_sha256='a' * 64)
            self.assertIsNone(model.model_paths["parent_task"])
            protocol = model.state_only_protocol()
            self.assertFalse({'dataset_id', 'dataset_episodes', 'dataset_frames',
                              'requested_training_epochs'}.intersection(protocol))
            self.assertEqual(protocol["normalization_strategy"], adapter.NORMALIZATION_STRATEGY)
            self.assertFalse(protocol["parent_task_checkpoint_loaded"])
            self.assertFalse(protocol["base_action_initializer_is_task_trained"])
            self.assertEqual(protocol["format"], "wholebody_inspire76_p25_flux2_v1")
            self.assertEqual(protocol["schema_id"], adapter.SCHEMA_ID)


if __name__ == "__main__":
    unittest.main(verbosity=2)
