"""Exercise fresh joint configuration generation and reject temporal/provenance drift."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from imagewam.joint_reference_contract import (
    NORMALIZATION_V2,
    expected_schema,
    make_joint_recipe,
)

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(
        name + "_jointv2_test", ROOT / "scripts/wholebody" / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder, trainer = script("build_config"), script("train")


def statistics():
    source = json.loads(
        (
            ROOT / "configs/wholebody/source_contracts/g1_existing_measured_v1.json"
        ).read_text()
    )
    recipe = make_joint_recipe(source)
    return dict(
        schema=expected_schema("measured", "predicted_rotation6d"),
        normalization_strategy=NORMALIZATION_V2,
        recipe_version="v2",
        body_target="measured",
        orientation_target="predicted_rotation6d",
        action_dim=47,
        proprio_dim=44,
        train_split_mode="all_episodes",
        heldout_episode_ids=[],
        train_episode_ids=[0, 1],
        train_frames=567,
        instruction="fixture demonstration",
        hand_target_time_offset=0,
        fps=30,
        action_stride=2,
        action_horizon=40,
        past_action_size=16,
        execution_horizon=16,
        future_image_offset=32,
        reference_source_contract=source,
        reference_source_contract_sha256=recipe["reference_source_contract_sha256"],
        joint_recipe=recipe,
        orientation_prediction_frame="current_heading_relative",
        orientation_anchor_source_key="observation.root_orientation",
    )


class JointV2ConfigTests(unittest.TestCase):
    def build(self, root):
        from omegaconf import OmegaConf

        for name in ("dataset", "cache", "flux", "qwen"):
            (root / name).mkdir()
        for name in ("flux.safetensors", "ae.safetensors", "initializer.pt"):
            (root / name).write_bytes(b"fixture asset, never loaded")
        (root / "stats.json").write_text(json.dumps(statistics()))
        args = SimpleNamespace(
            dataset_dir=root / "dataset",
            stats_path=root / "stats.json",
            text_cache_dir=root / "cache",
            output_dir=root / "run",
            flux2_src=root / "flux",
            flux2_model=root / "flux.safetensors",
            ae_model=root / "ae.safetensors",
            qwen_model=root / "qwen",
            action_initializer=root / "initializer.pt",
            representation="joint",
            horizon=None,
            execution_horizon=16,
            epochs=7,
            batch_size=3,
            accumulation=2,
            num_processes=4,
        )

        def measure(cfg):
            self.assertEqual(
                (cfg.data.train.action_horizon, cfg.data.train.past_action_size),
                (40, 16),
            )
            return dict(
                dataset_frames=567,
                raw_loader_batches=189,
                per_rank_microbatches=48,
                world_size=4,
                microbatch=3,
                gradient_accumulation=2,
                dataloader_drop_last=False,
            )

        receipt = builder.build(args, measure_loader=measure)
        production = OmegaConf.load(root / "run/production.yaml")
        smoke = OmegaConf.load(root / "run/smoke.yaml")
        return receipt, production, smoke

    def test_two_public_modes_have_distinct_representations(self):
        self.assertEqual(builder.canonical_representation("motion_token"), "motion76")
        self.assertEqual(builder.canonical_representation("joint"), "joint47")
        self.assertEqual(
            builder.recipe_spec(statistics(), "joint")["recipe_version"], "v2"
        )

    def test_generated_joint_recipe_keeps_execution16_and_real_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt, production, smoke = self.build(Path(directory))
            self.assertEqual(
                (
                    receipt["action_horizon"],
                    receipt["past_action_horizon"],
                    receipt["execution_horizon"],
                    receipt["lookahead_tail_horizon"],
                ),
                (40, 16, 16, 24),
            )
            self.assertEqual(receipt["future_action_offsets"], [0, 78])
            self.assertEqual(receipt["future_image_offset"], 32)
            self.assertEqual(receipt["past_action_offsets"], [-32, -2])
            self.assertEqual(
                trainer.validate_config(production, smoke=False)["total_updates"], 168
            )
            trainer.validate_config(smoke, smoke=True)
            self.assertFalse(Path(production.output_dir).exists())

    def test_source_timing_heading_and_model_drift_are_rejected(self):
        from omegaconf import OmegaConf

        with tempfile.TemporaryDirectory() as directory:
            _, production, _ = self.build(Path(directory))
            original = OmegaConf.to_container(production, resolve=True)
            changes = [
                lambda c: c["model"].update(expected_action_horizon=16),
                lambda c: c["model"].update(past_action_horizon=40),
                lambda c: c["model"].update(execution_horizon=40),
                lambda c: c["model"].update(source_contract_sha256="0" * 64),
                lambda c: c["data"]["train"].update(past_action_size=40),
                lambda c: c["wholebody_contract"]["joint_recipe"].update(
                    orientation_prediction_frame="world"
                ),
                lambda c: c["model"].update(body_target="wbc"),
            ]
            for index, mutate in enumerate(changes):
                with self.subTest(index=index):
                    altered = copy.deepcopy(original)
                    mutate(altered)
                    with self.assertRaises((ValueError, RuntimeError)):
                        trainer.validate_config(altered, smoke=False)

    def test_relabeling_statistics_cannot_make_decoder_targets_direct_reference(self):
        stats = statistics()
        stats["reference_source_contract"]["body_target"] = "direct_reference"
        stats["reference_source_contract"]["label_semantics"] = "direct_joint_reference"
        with self.assertRaises(ValueError):
            builder.recipe_spec(stats, "joint")


if __name__ == "__main__":
    unittest.main()
