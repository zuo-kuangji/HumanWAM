"""CPU tests for portable config/schedule/contracts; no model or CUDA allocation."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

REPO = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location("wholebody_" + name, REPO / "scripts/wholebody" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load_script("build_config")
trainer = load_script("train")


def valid_stats():
    return {"schema": builder.STATS_SCHEMA, "normalization_strategy": builder.NORMALIZATION,
            "action_dim": 76, "proprio_dim": 44, "train_split_mode": "all_episodes",
            "heldout_episode_ids": [], "train_episode_ids": [0, 1, 2], "train_frames": 93064,
            "hand_target_time_offset": 0}


def valid_joint_stats(dim=41, body_target="wbc"):
    orientation = "external" if dim == 41 else "predicted_rotation6d"
    label = "wbc29" if body_target == "wbc" else "measured29"
    suffix = "external-orientation" if dim == 41 else "quatwxyz-rotation6d-columns"
    result = valid_stats()
    result.update(schema=f"wholebody-state44-joint{dim}-{label}-measured-inspire12-{suffix}-v1",
        normalization_strategy=builder.JOINT_NORMALIZATION, action_dim=dim,
        instruction="fixture task",
        body_target=body_target, orientation_target=orientation, fps=30, action_stride=2,
        action_horizon=16, past_action_size=16, future_image_offset=32,
        body_target_time_offset=0, orientation_condition_in_model_input=False)
    if dim == 47:
        result.update(orientation_validity_rule="smpl_frame_index>0_and_stream_mode_in_1_4",
            orientation_validity_source_keys=["teleop.smpl_frame_index", "teleop.stream_mode"],
            orientation_invalid_policy="mask_only_last6_future_and_zero_last6_normalized_future_history",
            orientation_statistics_use_valid_rows_only=True,
            orientation_statistics_valid_frames=90000, orientation_statistics_invalid_frames=3064,
            orientation_validity_rejects_missing_flags=True, orientation_age_threshold_applied=False,
            rotation6d_order=["R00", "R10", "R20", "R01", "R11", "R21"],
            orientation_input_order="wxyz", orientation_source_key="teleop.body_quat_w")
    return result


def fixture_config(root):
    stats = root / "stats.json"
    stats.write_text(json.dumps(valid_stats()))
    initializer = root / "initializer.pt"
    initializer.write_bytes(b"fixture-only-not-a-real-initializer")
    return {
        "resume": None, "learning_rate": 1e-4, "weight_decay": .01, "mixed_precision": "bf16", "seed": 42,
        "batch_size": 8, "gradient_accumulation_steps": 2, "num_epochs": 60, "max_steps": 43680,
        "lr_scheduler_total_steps": 43680, "warmup_steps": 2184, "save_every": 728,
        "save_steps": [], "save_at_end": True, "keep_latest_weights": 4, "keep_latest_state_only": True,
        "eval_every": 0, "output_dir": str(root / "production"),
        "data": {"val": None, "train": {"_target_": builder.DATASET_TARGET, "stats_path": str(stats),
                 "train_split_mode": "all_episodes", "action_horizon": 40, "past_action_size": 40,
                 "qwen_context_len": 128, "qwen_text_dim": 7680}},
        "model": {"_target_": builder.MODEL_TARGET, "flux2_lora_config": {"enabled": False},
            "load_text_encoder": False, "pack_proprio_after_text": False, "action_state_dim": 44,
            "proprio_dim": 44, "expected_action_dim": 76, "expected_action_horizon": 40,
            "action_dit_config": {"action_dim": 76}, "action_dit_pretrained_path": str(initializer),
            "action_initializer_sha256": builder.sha256(initializer),
            "flux2_action_image_context": {"mode": "noise_idm", "include_future_tokens": True,
                "knowledge_insulation": True, "reuse_image_fm_workspace": False},
            "loss": {"lambda_video": .5, "lambda_action": 1.,
                "action_a2a": {"enabled": True, "fm_weight": 1., "ic_weight": .5, "ic_steps": 1,
                    "ic_loss": "l1", "source_noise_std": .5, "source_noise_dim_mask": [True] * 76,
                    "aura": {"enabled": False}},
                "image_i2i": {"enabled": True, "source_mode": "gaussian", "variance_normalize": True,
                    "current_residual_scale": 1 / math.sqrt(3), "ic_weight": 1., "ic_steps": 1, "ic_loss": "l1"}}},
        "wholebody_contract": {"fresh_base": True, "zero_stage": 1, "world_size": 8,
            "statistics_sha256": builder.sha256(stats), "updates_per_epoch": 728,
            "smoke_output_dir": str(root / "smoke"), "production_output_dir": str(root / "production"),
            "measured_loader": {"world_size": 8, "microbatch": 8, "gradient_accumulation": 2,
                "per_rank_microbatches": 1455, "dataset_frames": 93064}},
    }


def fixture_joint_config(root, dim=41, body_target="wbc"):
    cfg = fixture_config(root)
    orientation = "external" if dim == 41 else "predicted_rotation6d"
    stats_path = Path(cfg["data"]["train"]["stats_path"])
    stats_path.write_text(json.dumps(valid_joint_stats(dim, body_target)))
    cfg["model"].update(_target_=builder.JOINT_MODEL_TARGET, expected_action_dim=dim,
        expected_action_horizon=16, body_target=body_target, orientation_target=orientation,
        action_stride=2, reference_fps=30,
        state_only_schema_id=f"wholebody.joint_reference{dim}.{body_target}29.state44.p25.v1")
    cfg["model"].update(statistics_sha256=builder.sha256(stats_path),
        instruction_sha256=hashlib.sha256("fixture task".encode()).hexdigest())
    cfg["model"]["action_dit_config"]["action_dim"] = dim
    cfg["model"]["loss"]["action_a2a"]["source_noise_dim_mask"] = [True] * dim
    cfg["data"]["train"].update(_target_=builder.JOINT_DATASET_TARGET, action_horizon=16,
        past_action_size=16, action_stride=2, future_image_offset=32,
        body_target=body_target, orientation_target=orientation)
    cfg["wholebody_contract"].update(representation=f"joint{dim}", action_dim=dim,
        action_stride=2, action_horizon=16, past_action_horizon=16,
        statistics_sha256=builder.sha256(stats_path))
    cfg["EVALUATION"] = {"action_horizon": 16, "num_inference_steps": 1}
    return cfg


class ScheduleTests(unittest.TestCase):
    def test_reference_measurement_is_derived_not_hardcoded(self):
        self.assertEqual(builder.derive_schedule({"per_rank_microbatches": 1455}, 60, 2),
                         {"updates_per_epoch": 728, "total_updates": 43680, "warmup_updates": 2184})

    def test_different_dataset_world_batch_and_epochs_have_dynamic_schedule(self):
        self.assertEqual(builder.derive_schedule({"per_rank_microbatches": 23}, 7, 3),
                         {"updates_per_epoch": 8, "total_updates": 56, "warmup_updates": 2})

    def test_short_schedule_still_has_native_warmup(self):
        self.assertEqual(builder.derive_schedule({"per_rank_microbatches": 1}, 5, 1)["warmup_updates"], 1)
        with self.assertRaises(ValueError):
            builder.derive_schedule({"per_rank_microbatches": 1}, 4, 1)

    def test_smoke_microbatches_include_partial_epoch_accumulation(self):
        for loader, accumulation, expected in ((1455, 2, 10), (1, 2, 5), (3, 2, 8),
                                                (4, 3, 11), (9, 2, 9), (10, 2, 10)):
            with self.subTest(loader=loader, accumulation=accumulation):
                self.assertEqual(trainer.expected_microbatches(loader, accumulation, 5), expected)
        with self.assertRaises(RuntimeError):
            trainer.expected_microbatches(0, 2, 5)

    def test_no_fixed_episode_count(self):
        self.assertEqual(builder.validate_statistics(valid_stats()), 93064)
        stats = valid_stats()
        stats.update(train_frames=567, train_episode_ids=[0, 1])
        self.assertEqual(builder.validate_statistics(stats), 567)

    def test_legacy_statistics_or_future_hand_targets_rejected(self):
        for key, value in (("schema", "old-cloth"), ("normalization_strategy", "inherit_parent"),
                           ("hand_target_time_offset", -1), ("heldout_episode_ids", [3]),
                           ("action_dim", 78), ("train_frames", 0)):
            stats = valid_stats()
            stats[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                builder.validate_statistics(stats)

    def test_cli_requires_portable_asset_arguments(self):
        args = builder.argument_parser().parse_args([
            "--dataset-dir", "/data", "--stats-path", "/stats.json", "--text-cache-dir", "/cache",
            "--output-dir", "/newrun", "--flux2-src", "/flux", "--flux2-model", "/model.safetensors",
            "--ae-model", "/ae.safetensors", "--qwen-model", "/qwen", "--action-initializer", "/init.pt"])
        self.assertEqual((args.epochs, args.batch_size, args.accumulation, args.num_processes), (60, 8, 2, 8))


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cfg = fixture_config(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_production_validates_without_importing_torch(self):
        result = trainer.validate_config(self.cfg, smoke=False)
        self.assertEqual(result["total_updates"], 43680)

    def test_both_horizons_validate_and_window_mismatches_are_rejected(self):
        for horizon in (16,40):
            c=copy.deepcopy(self.cfg)
            c['model']['expected_action_horizon']=horizon
            c['data']['train'].update(action_horizon=horizon,past_action_size=horizon)
            c['wholebody_contract'].update(action_horizon=horizon,past_action_horizon=horizon)
            c['EVALUATION']={'action_horizon':horizon}
            trainer.validate_config(c,smoke=False)
            for part,key in [('data','past_action_size'),('EVALUATION','action_horizon')]:
                wrong=copy.deepcopy(c)
                (wrong['data']['train'] if part=='data' else wrong[part])[key]=40 if horizon==16 else 16
                with self.assertRaises(RuntimeError):trainer.validate_config(wrong,smoke=False)

    def test_smoke_keeps_production_schedule_but_saves_nothing(self):
        self.cfg.update(output_dir=str(self.root / "smoke"), max_steps=5, save_every=0, save_at_end=False)
        self.assertEqual(trainer.validate_config(self.cfg, smoke=True)["total_updates"], 43680)
        self.cfg["save_at_end"] = True
        with self.assertRaises(RuntimeError):
            trainer.validate_config(self.cfg, smoke=True)

    def test_changed_world_or_resume_rejected(self):
        for change in (lambda c: c.update(resume="old-task.pt"),
                       lambda c: c["wholebody_contract"].update(world_size=4),
                       lambda c: c["model"].update(parent_checkpoint="cloth.pt"),
                       lambda c: c["data"].update(val={}),
                       lambda c: c["model"].update(pack_proprio_after_text=True),
                       lambda c: c["model"]["loss"]["action_a2a"]["source_noise_dim_mask"].__setitem__(75, False),
                       lambda c: c.update(warmup_steps=0)):
            bad = copy.deepcopy(self.cfg)
            change(bad)
            with self.assertRaises(RuntimeError):
                trainer.validate_config(bad, smoke=False)

    def test_statistics_and_initializer_are_content_pinned(self):
        Path(self.cfg["data"]["train"]["stats_path"]).write_text("{}")
        with self.assertRaisesRegex(RuntimeError, "Statistics changed"):
            trainer.validate_config(self.cfg, smoke=False)
        self.cfg = fixture_config(self.root)
        Path(self.cfg["model"]["action_dit_pretrained_path"]).write_bytes(b"changed")
        with self.assertRaisesRegex(RuntimeError, "initializer changed"):
            trainer.validate_config(self.cfg, smoke=False)

    def test_production_gate_requires_every_matching_smoke_rank(self):
        smoke = self.root / "smoke"
        smoke.mkdir()
        for rank in range(8):
            report = {"status": "FIVE_UPDATES_VERIFIED", "actual_optimizer_updates": 5, "skipped_updates": 0,
                "world_size": 8, "microbatch": 8, "accumulation": 2,
                "statistics_sha256": self.cfg["wholebody_contract"]["statistics_sha256"],
                "action_initializer_sha256": self.cfg["model"]["action_initializer_sha256"],
                "schedule_total_updates": 43680}
            (smoke / f"smoke_rank_{rank}.json").write_text(json.dumps(report))
        trainer.require_completed_smoke(self.cfg)
        wrong_horizon=copy.deepcopy(self.cfg)
        wrong_horizon['model']['expected_action_horizon']=16
        with self.assertRaisesRegex(RuntimeError,"horizon"):
            trainer.require_completed_smoke(wrong_horizon)
        report["actual_optimizer_updates"] = 4
        (smoke / "smoke_rank_7.json").write_text(json.dumps(report))
        with self.assertRaises(RuntimeError):
            trainer.require_completed_smoke(self.cfg)

    def test_output_claim_refuses_existing_run(self):
        config = self.root / "config.yaml"
        config.write_text("fixture")
        run = self.root / "output"
        with mock.patch.dict(os.environ, {"RANK": "0"}):
            trainer.claim_output(run, config)
            with self.assertRaises(FileExistsError):
                trainer.claim_output(run, config)

    def test_scripts_and_template_have_no_host_experiment_path(self):
        for path in (REPO / "scripts/wholebody/build_config.py", REPO / "scripts/wholebody/train.py",
                     REPO / "configs/wholebody/deltaflow76_p25.yaml"):
            text = path.read_text()
            self.assertNotIn("/mnt/data", text)
            self.assertNotIn("/Users/", text)
            self.assertNotIn("93064", text)


class JointContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_joint41_joint47_and_explicit_measured_ablation_validate(self):
        for dim in (41, 47):
            for body in ("wbc", "measured"):
                with self.subTest(dim=dim, body=body):
                    cfg = fixture_joint_config(self.root, dim, body)
                    self.assertEqual(trainer.validate_config(cfg, smoke=False)["total_updates"], 43680)
                    stats = valid_joint_stats(dim, body)
                    spec = builder.recipe_spec(stats, f"joint{dim}")
                    self.assertEqual(builder.validate_statistics(stats, spec), 93064)

    def test_joint_wrong_schema_timing_dimension_and_noise_are_rejected(self):
        mutations = (
            lambda c: c["model"].update(state_only_schema_id="wholebody.state44.action76.legacy"),
            lambda c: c["model"].update(expected_action_horizon=40),
            lambda c: c["model"].update(action_stride=1),
            lambda c: c["model"].update(reference_fps=50),
            lambda c: c["model"].update(statistics_sha256="0" * 64),
            lambda c: c["model"].update(instruction_sha256="0" * 64),
            lambda c: c["model"].update(orientation_target="external"),
            lambda c: c["model"]["action_dit_config"].update(action_dim=76),
            lambda c: c["data"]["train"].update(body_target="measured"),
            lambda c: c["data"]["train"].update(past_action_size=40),
            lambda c: c["data"]["train"].update(future_image_offset=16),
            lambda c: c["model"]["loss"]["action_a2a"].update(source_noise_dim_mask=[True] * 76),
            lambda c: c["model"]["loss"]["action_a2a"]["source_noise_dim_mask"].__setitem__(29, False),
            lambda c: c["model"]["loss"]["action_a2a"].update(source_noise_std=0.0),
        )
        for i, mutate in enumerate(mutations):
            with self.subTest(case=i):
                cfg = fixture_joint_config(self.root, 47)
                mutate(cfg)
                with self.assertRaises(RuntimeError): trainer.validate_config(cfg, smoke=False)

    def test_joint_statistics_timing_and_orientation_validity_must_be_pinned(self):
        changes = [("action_horizon", 40), ("past_action_size", 40), ("future_image_offset", 16),
                   ("fps", 50), ("action_stride", 1), ("action_dim", 76), ("schema", builder.STATS_SCHEMA),
                   ("instruction", "a different task")]
        for dim in (41, 47):
            for key, value in changes:
                with self.subTest(dim=dim, key=key):
                    cfg = fixture_joint_config(self.root, dim)
                    path = Path(cfg["data"]["train"]["stats_path"])
                    stats = valid_joint_stats(dim); stats[key] = value
                    path.write_text(json.dumps(stats))
                    cfg["wholebody_contract"]["statistics_sha256"] = builder.sha256(path)
                    cfg["model"]["statistics_sha256"] = builder.sha256(path)
                    with self.assertRaises(RuntimeError): trainer.validate_config(cfg, smoke=False)
        cfg = fixture_joint_config(self.root, 47)
        path = Path(cfg["data"]["train"]["stats_path"])
        stats = valid_joint_stats(47)
        del stats["orientation_validity_rule"]
        path.write_text(json.dumps(stats))
        cfg["wholebody_contract"]["statistics_sha256"] = builder.sha256(path)
        cfg["model"]["statistics_sha256"] = builder.sha256(path)
        with self.assertRaises(RuntimeError): trainer.validate_config(cfg, smoke=False)

    def test_joint_smoke_gate_rejects_report_from_different_action_dimension(self):
        cfg = fixture_joint_config(self.root, 47)
        smoke = self.root / "smoke"
        smoke.mkdir()
        report = {"status": "FIVE_UPDATES_VERIFIED", "actual_optimizer_updates": 5,
            "skipped_updates": 0, "world_size": 8, "microbatch": 8, "accumulation": 2,
            "statistics_sha256": cfg["wholebody_contract"]["statistics_sha256"],
            "action_initializer_sha256": cfg["model"]["action_initializer_sha256"],
            "schedule_total_updates": 43680, "action_dim": 47, "action_horizon": 16}
        for rank in range(8):
            (smoke / f"smoke_rank_{rank}.json").write_text(json.dumps(report))
        trainer.require_completed_smoke(cfg)
        report["action_dim"] = 41
        (smoke / "smoke_rank_7.json").write_text(json.dumps(report))
        with self.assertRaisesRegex(RuntimeError, "horizon"):
            trainer.require_completed_smoke(cfg)


@unittest.skipUnless(importlib.util.find_spec("omegaconf"), "Full config integration needs the repository's OmegaConf dependency")
class ConfigBuildTests(unittest.TestCase):
    def test_complete_resolved_yaml_and_reject_overwrite(self):
        from omegaconf import OmegaConf
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("dataset", "textcache", "fluxsrc", "qwen"):
                (root / name).mkdir()
            for name in ("flux-2-klein-base-4b.safetensors", "ae.safetensors", "init.pt"):
                (root / name).write_bytes(b"testasset")
            stats = valid_stats()
            stats.update(train_frames=567, train_episode_ids=[0, 1])
            (root / "stats.json").write_text(json.dumps(stats))
            args = SimpleNamespace(dataset_dir=root / "dataset", stats_path=root / "stats.json",
                text_cache_dir=root / "textcache", output_dir=root / "newrun", flux2_src=root / "fluxsrc",
                flux2_model=root / "flux-2-klein-base-4b.safetensors", ae_model=root / "ae.safetensors",
                qwen_model=root / "qwen", action_initializer=root / "init.pt",
                epochs=7, batch_size=3, accumulation=2, num_processes=4)
            def measured(cfg):
                self.assertEqual(cfg.model.action_initializer_sha256, builder.sha256(root / "init.pt"))
                self.assertEqual(cfg.data.train._target_, builder.DATASET_TARGET)
                return {"dataset_frames": 567, "raw_loader_batches": 189,
                        "per_rank_microbatches": 48, "world_size": 4, "microbatch": 3,
                        "gradient_accumulation": 2, "dataloader_drop_last": False, "source": "test fixture"}
            receipt = builder.build(args, measure_loader=measured)
            self.assertEqual((receipt["updates_per_epoch"], receipt["total_updates"], receipt["warmup_updates"]), (24, 168, 8))
            production = OmegaConf.load(root / "newrun/production.yaml")
            smoke = OmegaConf.load(root / "newrun/smoke.yaml")
            self.assertEqual(production.data.train.action_horizon,16)
            self.assertEqual(production.data.train.past_action_size,16)
            self.assertEqual(production.model.expected_action_horizon,16)
            self.assertEqual(production.EVALUATION.action_horizon,16)
            self.assertEqual(receipt['future_action_offsets'],[0,15])
            trainer.validate_config(production, smoke=False)
            trainer.validate_config(smoke, smoke=True)
            self.assertEqual(smoke.lr_scheduler_total_steps, production.lr_scheduler_total_steps)
            self.assertEqual(smoke.num_workers, production.num_workers)
            self.assertFalse((root / "newrun/production").exists())
            self.assertFalse((root / "newrun/smoke").exists())
            with self.assertRaises(ValueError):
                builder.build(args, measure_loader=measured)

    def test_complete_joint_yaml_routes_all_dimensions_sources_and_stride(self):
        from omegaconf import OmegaConf
        for dim, body in ((41, "wbc"), (47, "wbc"), (41, "measured"), (47, "measured")):
            with self.subTest(dim=dim, body=body), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                for name in ("dataset", "textcache", "fluxsrc", "qwen"): (root / name).mkdir()
                for name in ("flux.safetensors", "ae.safetensors", "init.pt"):
                    (root / name).write_bytes(b"testasset")
                stats = valid_joint_stats(dim, body)
                stats.update(train_frames=567, train_episode_ids=[0, 1])
                if dim == 47:
                    stats.update(orientation_statistics_valid_frames=560, orientation_statistics_invalid_frames=7)
                (root / "stats.json").write_text(json.dumps(stats))
                args = SimpleNamespace(dataset_dir=root / "dataset", stats_path=root / "stats.json",
                    text_cache_dir=root / "textcache", output_dir=root / "newrun", flux2_src=root / "fluxsrc",
                    flux2_model=root / "flux.safetensors", ae_model=root / "ae.safetensors", qwen_model=root / "qwen",
                    action_initializer=root / "init.pt", representation=f"joint{dim}", horizon=16,
                    epochs=7, batch_size=3, accumulation=2, num_processes=4)
                def measured(cfg):
                    self.assertEqual(cfg.model._target_, builder.JOINT_MODEL_TARGET)
                    self.assertEqual(cfg.data.train._target_, builder.JOINT_DATASET_TARGET)
                    self.assertEqual(cfg.model.expected_action_dim, dim)
                    self.assertEqual(cfg.model.action_dit_config.action_dim, dim)
                    self.assertEqual(cfg.model.body_target, body)
                    self.assertEqual(list(cfg.model.loss.action_a2a.source_noise_dim_mask), [True] * dim)
                    return {"dataset_frames": 567, "raw_loader_batches": 189, "per_rank_microbatches": 48,
                            "world_size": 4, "microbatch": 3, "gradient_accumulation": 2,
                            "dataloader_drop_last": False, "source": "mock native loader measurement; no iteration"}
                receipt = builder.build(args, measure_loader=measured)
                self.assertEqual((receipt["action_dim"], receipt["action_stride"], receipt["future_image_offset"]), (dim, 2, 32))
                self.assertEqual(receipt["future_action_offsets"], [0, 30])
                self.assertEqual(receipt["past_action_offsets"], [-32, -2])
                production = OmegaConf.load(root / "newrun/production.yaml")
                smoke = OmegaConf.load(root / "newrun/smoke.yaml")
                trainer.validate_config(production, smoke=False)
                trainer.validate_config(smoke, smoke=True)
                self.assertEqual(production.model.state_only_schema_id,
                                 f"wholebody.joint_reference{dim}.{body}29.state44.p25.v1")
                self.assertEqual(production.model.statistics_sha256, builder.sha256(root / "stats.json"))
                self.assertEqual(production.model.instruction_sha256, hashlib.sha256(stats["instruction"].encode()).hexdigest())
                self.assertEqual(production.data.train.action_stride, 2)
                self.assertEqual(production.model.reference_fps, 30)
                self.assertEqual(smoke.lr_scheduler_total_steps, production.lr_scheduler_total_steps)
                self.assertFalse((root / "newrun/production").exists())
                self.assertFalse((root / "newrun/smoke").exists())

    def test_joint_recipe_rejects_legacy_horizon_wrong_statistics_and_missing_validity(self):
        for dim in (41, 47):
            for key, value in [("schema", builder.STATS_SCHEMA), ("action_dim", 76),
                    ("action_horizon", 40), ("past_action_size", 40), ("future_image_offset", 16),
                    ("fps", 50), ("action_stride", 1), ("instruction", "")]:
                with self.subTest(dim=dim, key=key), self.assertRaises(ValueError):
                    stats = valid_joint_stats(dim)
                    stats[key] = value
                    spec = builder.recipe_spec(stats, f"joint{dim}")
                    builder.validate_statistics(stats, spec)
        stats = valid_joint_stats(47)
        del stats["orientation_validity_rule"]
        with self.assertRaises(ValueError):
            builder.validate_statistics(stats, builder.recipe_spec(stats, "joint47"))


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ("torch", "omegaconf", "hydra", "accelerate")),
                     "Native CPU loader integration needs the installed repository dependencies")
class NativeLoaderTests(unittest.TestCase):
    def test_real_native_loader_measurement_without_iteration_or_gpu(self):
        # The native-loss AST fixtures intentionally register a minimal imagewam
        # namespace. A fresh CPU process avoids letting those fixture modules
        # replace the actual trainer import during test discovery.
        code = '''
import importlib.util,json
from unittest import mock
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from pathlib import Path
spec=importlib.util.spec_from_file_location("build_config",Path("scripts/wholebody/build_config.py"))
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
class UniteratedDataset(Dataset):
 def __len__(self):return 93064
 def __getitem__(self,index):raise AssertionError("Loader measurement must never iterate/decode")
cfg=OmegaConf.create(dict(batch_size=8,num_workers=0,prefetch_factor=2,persistent_workers=False,
 seed=42,gradient_accumulation_steps=2,data=dict(train={}),wholebody_contract=dict(world_size=8)))
with mock.patch("hydra.utils.instantiate",return_value=UniteratedDataset()):
 measured=builder.measure_native_loader(cfg)
print("MEASUREMENT="+json.dumps(measured))
'''
        env=dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONPATH=str(REPO/"src"))
        result=subprocess.run([sys.executable,"-c",code],cwd=REPO,env=env,text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        measured=json.loads(next(row.removeprefix("MEASUREMENT=") for row in result.stdout.splitlines()
                                 if row.startswith("MEASUREMENT=")))
        self.assertEqual((measured["raw_loader_batches"], measured["per_rank_microbatches"]), (11633, 1455))
        self.assertEqual(builder.derive_schedule(measured, 60, 2)["total_updates"], 43680)


if __name__ == "__main__":
    unittest.main()
