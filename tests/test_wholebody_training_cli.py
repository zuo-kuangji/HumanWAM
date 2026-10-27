"""CPU tests for portable config/schedule/contracts; no model or CUDA allocation."""
from __future__ import annotations

import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
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
            trainer.validate_config(production, smoke=False)
            trainer.validate_config(smoke, smoke=True)
            self.assertEqual(smoke.lr_scheduler_total_steps, production.lr_scheduler_total_steps)
            self.assertEqual(smoke.num_workers, production.num_workers)
            self.assertFalse((root / "newrun/production").exists())
            self.assertFalse((root / "newrun/smoke").exists())
            with self.assertRaises(ValueError):
                builder.build(args, measure_loader=measured)


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ("torch", "omegaconf", "hydra", "accelerate")),
                     "Native CPU loader integration needs the installed repository dependencies")
class NativeLoaderTests(unittest.TestCase):
    def test_real_native_loader_measurement_without_iteration_or_gpu(self):
        from omegaconf import OmegaConf
        from torch.utils.data import Dataset
        class UniteratedDataset(Dataset):
            def __len__(self):
                return 93064
            def __getitem__(self, index):
                raise AssertionError("Config generation must never iterate/decode data")
        cfg = OmegaConf.create({"batch_size": 8, "num_workers": 0, "prefetch_factor": 2,
            "persistent_workers": False, "seed": 42, "gradient_accumulation_steps": 2,
            "data": {"train": {}}, "wholebody_contract": {"world_size": 8}})
        with mock.patch("hydra.utils.instantiate", return_value=UniteratedDataset()):
            measured = builder.measure_native_loader(cfg)
        self.assertEqual((measured["raw_loader_batches"], measured["per_rank_microbatches"]), (11633, 1455))
        self.assertEqual(builder.derive_schedule(measured, 60, 2)["total_updates"], 43680)


if __name__ == "__main__":
    unittest.main()
