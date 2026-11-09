"""CPU tests of joint targets, causal sampling and immutable preparation."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from imagewam.datasets import wholebody_joint_reference as md
from imagewam.joint_reference_contract import make_joint_recipe, source_contract_sha256, NORMALIZATION_V2
from imagewam.joint_reference_rotation import (
    heading_from_quaternion_wxyz, world_quaternion_to_heading_relative_rotation6d,
    heading_relative_rotation6d_to_world_quaternion, quaternion_matrix,
)


def raw_columns(n=100, bias=0.0):
    time = np.linspace(0, 1, n, dtype=np.float32)[:, None]
    def field(dim):
        return time + np.arange(dim, dtype=np.float32)[None] * 0.01 + bias
    theta = np.arange(n, dtype=np.float64) / 100
    quat = np.stack([np.cos(theta / 2), np.zeros(n), np.zeros(n), np.sin(theta / 2)], axis=-1)
    return {"observation.state": field(43), "action.wbc": field(43) + 10,
            "observation.inspire_left_hand_state": time * 0.6 + np.arange(6)[None] * 0.01,
            "observation.inspire_right_hand_state": 0.9 - time * 0.5 + np.arange(6)[None] * 0.01,
            "observation.projected_gravity": field(3), "teleop.body_quat_w": quat,
            "teleop.smpl_frame_index": np.arange(n, dtype=np.int64) + 1,
            "teleop.stream_mode": np.ones(n, dtype=np.int32),
            "action.motion_token": field(64) + 1000,
            "teleop.target_body_orientation": field(6) + 10000,
            "teleop.inspire_left_hand_cmd": np.full((n, 6), 99999)}


class JointReferenceNumericTests(unittest.TestCase):
    def test_body_mapping_is_issued_wbc_not_motion_or_placeholder_hand(self):
        raw = raw_columns()
        state, action = md.map_numeric(raw)
        self.assertEqual(state.shape, (100, 44))
        self.assertEqual(action.shape, (100, 41))
        np.testing.assert_array_equal(action[:, :29], raw["action.wbc"][:, md.BODY_INDICES])
        np.testing.assert_array_equal(state[:, :29], raw["observation.state"][:, md.BODY_INDICES])
        np.testing.assert_array_equal(action[:, 29:41], state[:, 29:41])
        np.testing.assert_array_equal(action[:, 29:35], raw["observation.inspire_left_hand_state"].astype(np.float32))
        self.assertNotIn("action.motion_token", md.numeric_columns())
        self.assertNotIn(md.ORIENTATION_SOURCE_KEY, md.numeric_columns())
        changed = deepcopy(raw)
        changed["teleop.body_quat_w"][:] = np.nan
        np.testing.assert_array_equal(md.map_numeric(changed)[1], action)

    def test_missing_wbc_never_falls_back_but_measured_is_explicit(self):
        raw = raw_columns()
        del raw["action.wbc"]
        with self.assertRaisesRegex(ValueError, "no implicit source fallback"):
            md.map_numeric(raw)
        state, measured = md.map_numeric(raw, body_target="measured")
        np.testing.assert_array_equal(measured[:, :41], state[:, :41])
        self.assertNotIn("action.wbc", md.numeric_columns(body_target="measured"))
        stats = md.fit_statistics({0: (state, measured)}, [0], body_target="measured")
        self.assertEqual(stats["body_target_source"], "observation.state")
        self.assertNotEqual(stats["schema"], md.schema_for("wbc", "external"))
        md.JointReferenceNormalizer(stats)

    def test_quaternion_rotation_columns_sign_invariant_and_not_transpose(self):
        q = np.array([[1, 0, 0, 0], [np.sqrt(0.5), 0, 0, np.sqrt(0.5)], [0, 1, 0, 0]])
        expected = [[1, 0, 0, 0, 1, 0], [0, 1, 0, -1, 0, 0], [1, 0, 0, 0, -1, 0]]
        np.testing.assert_allclose(md.quaternion_wxyz_to_rotation6d(q), expected, atol=1e-6)
        np.testing.assert_array_equal(md.quaternion_wxyz_to_rotation6d(q), md.quaternion_wxyz_to_rotation6d(-q))
        np.testing.assert_allclose(md.quaternion_wxyz_to_rotation6d(q * 7), expected, atol=1e-6)
        for wrong in [np.zeros((1, 4)), np.full((1, 4), np.nan), np.zeros((1, 3))]:
            with self.assertRaises(ValueError): md.quaternion_wxyz_to_rotation6d(wrong)
        raw = raw_columns()
        _, action = md.map_numeric(raw, orientation_target="predicted_rotation6d")
        self.assertEqual(action.shape, (100, 47))
        np.testing.assert_array_equal(action[:, 41:], md.quaternion_wxyz_to_rotation6d(raw[md.ORIENTATION_SOURCE_KEY].astype(np.float32)))

    def test_stride_two_causal_history_one_second_future_and_tail_masks(self):
        mid = md.window_indices(50, 100)
        np.testing.assert_array_equal(mid["action"], np.arange(50, 82, 2))
        np.testing.assert_array_equal(mid["past_action"], np.arange(18, 50, 2))
        np.testing.assert_array_equal(mid["image"], [50, 82])
        self.assertTrue((mid["past_action"] < 50).all())
        self.assertEqual(md.timing_configuration()["future_action_span_seconds"], 1.0)
        self.assertTrue(md.window_indices(0, 100)["past_action_is_pad"].all())
        tail = md.window_indices(99, 100)
        self.assertEqual(int(tail["action_is_pad"].sum()), 15)
        self.assertTrue((tail["action"] == 99).all())
        self.assertTrue(tail["image_is_pad"][1])
        for bad in [0, -1, True, 2.5]:
            with self.assertRaises(ValueError): md.window_indices(50, 100, action_stride=bad)

    def test_collection_quantiles_physical_hands_and_explicit_sources(self):
        arrays = {0: md.map_numeric(raw_columns()), 1: md.map_numeric(raw_columns(bias=100))}
        stats = md.fit_statistics(arrays, [0, 1])
        self.assertEqual(stats["action_dim"], 41)
        self.assertEqual(stats["body_target_source"], "action.wbc")
        self.assertIn("not original teleop", stats["body_target_semantics"])
        self.assertIsNone(stats["orientation_source_key"])
        self.assertEqual(stats["train_frames"], 200)
        self.assertEqual(stats["heldout_episode_ids"], [])
        all_action = np.concatenate([arrays[0][1], arrays[1][1]])
        np.testing.assert_allclose(stats["action"][md.ACTION_KEY]["global_q01"],
                                   np.quantile(all_action.astype(np.float64), .01, axis=0))
        normalizer = md.JointReferenceNormalizer(stats)
        physical = all_action.copy()
        physical[:, 29:41] = .25
        normalized = normalizer.forward(physical, "action")
        np.testing.assert_array_equal(normalized[:, 29:41], -.5)
        np.testing.assert_allclose(normalizer.backward(normalized), physical, atol=2e-5)
        for ids in [[0], [1, 0], [0, 0, 1]]:
            with self.assertRaises(ValueError): md.fit_statistics(arrays, ids)
        wrong = deepcopy(stats); wrong["schema"] = "wholebody-state44-action76-sameframe-all-episodes-v1"
        with self.assertRaises(ValueError): md.JointReferenceNormalizer(wrong)

    def test_orientation47_stats_and_current_state_independence(self):
        raw = raw_columns()
        state, action = md.map_numeric(raw, orientation_target="predicted_rotation6d")
        stats = md.fit_statistics({0: (state, action, md.orientation_validity(raw))}, [0], orientation_target="predicted_rotation6d")
        self.assertEqual(stats["action_dim"], 47)
        self.assertEqual(stats["rotation6d_order"], ["R00", "R10", "R20", "R01", "R11", "R21"])
        self.assertFalse(stats["orientation_condition_in_model_input"])
        normalizer = md.JointReferenceNormalizer(stats)
        self.assertTrue(np.isfinite(normalizer.forward(action, "action")).all())
        raw[md.ORIENTATION_SOURCE_KEY] *= -1
        np.testing.assert_array_equal(md.map_numeric(raw, orientation_target="predicted_rotation6d")[0], state)
        wrong = deepcopy(stats); wrong["rotation6d_order"].reverse()
        with self.assertRaisesRegex(ValueError, "rotation6d convention"):
            md.JointReferenceNormalizer(wrong)

    def test_orientation_validity_is_explicit_and_quantiles_exclude_fallback(self):
        raw = raw_columns()
        raw["teleop.smpl_frame_index"][::3] = 0
        raw["teleop.stream_mode"][::7] = 5
        valid = md.orientation_validity(raw)
        self.assertFalse(valid[0])
        self.assertTrue(valid[1])
        state, action = md.map_numeric(raw, orientation_target="predicted_rotation6d")
        stats = md.fit_statistics({0: (state, action, valid)}, [0], orientation_target="predicted_rotation6d")
        expected = np.quantile(action[valid, 41:].astype(np.float64), .01, axis=0)
        np.testing.assert_array_equal(stats["action"][md.ACTION_KEY]["global_q01"][41:], expected)
        self.assertEqual(stats["orientation_statistics_valid_frames"], int(valid.sum()))
        self.assertEqual(stats["orientation_statistics_invalid_frames"], int((~valid).sum()))
        with self.assertRaisesRegex(ValueError, "explicit per-row orientation validity"):
            md.fit_statistics({0: (state, action)}, [0], orientation_target="predicted_rotation6d")
        with self.assertRaisesRegex(ValueError, "at least one valid orientation"):
            md.fit_statistics({0: (state, action, np.zeros(100, dtype=bool))}, [0], orientation_target="predicted_rotation6d")
        del raw["teleop.smpl_frame_index"]
        with self.assertRaisesRegex(ValueError, "validity source"):
            md.map_numeric(raw, orientation_target="predicted_rotation6d")
        self.assertEqual(md.map_numeric(raw)[1].shape, (100, 41))


@unittest.skipUnless(importlib.util.find_spec("pyarrow"), "PyArrow required for on-disk fixtures")
class JointReferenceDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.root, self.output, self.cache = self.base / "raw", self.base / "stats", self.base / "text"
        self.raws = {}
        self.write_fixture()

    def tearDown(self):
        self.temp.cleanup()

    def write_fixture(self, count=2, length=100):
        import pyarrow as pa
        import pyarrow.parquet as pq
        (self.root / "meta").mkdir(parents=True)
        (self.root / "data/chunk-000").mkdir(parents=True)
        self.cache.mkdir()
        raw_names = ["unused_dex3_hand_slot"] * 43
        for i, name in zip(md.BODY_INDICES, md.BODY_JOINT_NAMES): raw_names[i] = name
        features = {key: {"shape": [dim], "dtype": "float32"} for key, dim in {
            "observation.state": 43, "action.wbc": 43, md.ORIENTATION_SOURCE_KEY: 4,
            "teleop.smpl_frame_index": 1, "teleop.stream_mode": 1,
            "observation.inspire_left_hand_state": 6, "observation.inspire_right_hand_state": 6,
            "observation.projected_gravity": 3}.items()}
        for key in ("observation.state", "action.wbc"): features[key]["names"] = raw_names
        for side in ("left", "right"):
            features[f"observation.inspire_{side}_hand_state"]["names"] = [f"{side}_{x}" for x in md.HAND_FIELD_ORDER]
        features.update({cam: {"shape": [480, 640, 3], "dtype": "video"} for cam in md.CAMERAS})
        info = {"codebase_version": "v2.1", "fps": 30, "total_episodes": count,
                "total_frames": count * length, "total_videos": count * 2,
                "chunks_size": 1000, "splits": {"train": f"0:{count}"}, "features": features}
        (self.root / "meta/info.json").write_text(json.dumps(info))
        episodes, stats = [], []
        self.instruction = "move the pillow to the bedhead"
        for ep in range(count):
            raw = raw_columns(length, ep * 2)
            self.raws[ep] = raw
            cols = {key: raw[key].tolist() for key in md.numeric_columns(orientation_target="predicted_rotation6d") if key in raw}
            cols.update(timestamp=(np.arange(length) / 30).astype(np.float32).tolist(), frame_index=list(range(length)),
                        index=list(range(ep * length, (ep + 1) * length)), episode_index=[ep] * length, task_index=[0] * length)
            pq.write_table(pa.Table.from_pydict(cols), self.root / f"data/chunk-000/episode_{ep:06d}.parquet")
            episodes.append({"episode_index": ep, "length": length, "tasks": [self.instruction]})
            stats.append({"episode_index": ep, "stats": {key: {"count": [length]} for key in cols}})
            for camera in md.CAMERAS:
                directory = self.root / "videos/chunk-000" / camera
                directory.mkdir(parents=True, exist_ok=True)
                (directory / f"episode_{ep:06d}.mp4").write_bytes(b"synthetic decoder fixture")
        for name, rows in [("episodes.jsonl", episodes), ("episodes_stats.jsonl", stats),
                           ("tasks.jsonl", [{"task_index": 0, "task": self.instruction}])]:
            (self.root / "meta" / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
        prompt = md.DEFAULT_PROMPT.format(task=self.instruction)
        filename = hashlib.sha256(prompt.encode()).hexdigest() + ".qwen3_flux2_len4.pt"
        torch.save({"text_hidden_states": torch.zeros(4, 8), "text_attention_mask": torch.ones(4, dtype=torch.bool)}, self.cache / filename)

    def prepare(self, **kwargs):
        return md.prepare_statistics(str(self.root), str(self.output), **kwargs)

    def dataset(self, **kwargs):
        self.decoder_calls = []
        def decoder(path, indices):
            self.decoder_calls.append((path, indices))
            return {i: torch.full((3, 224, 224), i / 200) for i in indices}
        return md.JointReferenceLeRobotDataset(str(self.root), str(self.output / "dataset_stats.json"),
            str(self.cache), qwen_context_len=4, qwen_text_dim=8, decoder=decoder, **kwargs)

    def test_immutable_preparation_and_all_episode_sha_no_validation(self):
        before = {str(p.relative_to(self.root)): md.sha256(p) for p in self.root.rglob("*") if p.is_file()}
        audit = self.prepare()
        after = {str(p.relative_to(self.root)): md.sha256(p) for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(audit["episodes"], 2)
        self.assertEqual(len(audit["metadata_sha256"]), 4)
        self.assertEqual(len(audit["parquet_sha256"]), 2)
        self.assertEqual(audit["validation_episodes"], [])
        self.assertFalse(audit["tail_samples_are_complete_deployment_futures"])
        with self.assertRaises(FileExistsError): self.prepare()
        with self.assertRaises(ValueError): md.prepare_statistics(str(self.root), str(self.root / "cache"))

    def test_stride_current_condition_and_masked_history_not_future_labels(self):
        self.prepare()
        ds = self.dataset()
        sample = ds[50]
        state, action = md.map_numeric(self.raws[0])
        self.assertEqual(tuple(sample["action"].shape), (16, 41))
        self.assertEqual(tuple(sample["video"].shape), (3, 2, 224, 448))
        np.testing.assert_array_equal(sample["action"], ds.normalizer.forward(action[50:82:2], "action"))
        np.testing.assert_array_equal(sample["past_action"], ds.normalizer.forward(action[18:50:2], "action"))
        np.testing.assert_array_equal(sample["proprio"], np.repeat(ds.normalizer.forward(state[50], "state")[None], 16, axis=0))
        self.assertTrue(all(indices == [50, 82] for path, indices in self.decoder_calls))
        self.assertTrue((ds[0]["past_action"] == 0).all())
        partial = ds[3]
        self.assertEqual(int(partial["past_action_is_pad"].sum()), 15)
        self.assertTrue((partial["past_action"][:15] == 0).all())
        np.testing.assert_array_equal(partial["past_action"][-1], ds.normalizer.forward(action[1], "action"))
        self.assertEqual(int(ds[99]["action_is_pad"].sum()), 15)
        self.assertEqual(ds[100]["episode_index"], 1)
        self.assertTrue((ds[100]["past_action"] == 0).all())
        with self.assertRaises(ValueError): self.dataset(action_stride=1)
        with self.assertRaises(ValueError): self.dataset(split="validation")

    def test_orientation47_no_orientation_in_proprio_text_or_image_input(self):
        self.prepare(orientation_target="predicted_rotation6d")
        ds = self.dataset(orientation_target="predicted_rotation6d")
        a = ds[50]
        before = {key: a[key].clone() for key in ("proprio", "video", "text_hidden_states", "text_attention_mask")}
        state, action, valid = ds.episodes[0]
        action[50:, 41:] = -action[50:, 41:]
        state[51:] = 999  # Future body/hand/gravity observations are targets only.
        b = ds[50]
        self.assertEqual(tuple(b["action"].shape), (16, 47))
        self.assertFalse(torch.equal(a["action"][:, 41:], b["action"][:, 41:]))
        for key, value in before.items(): torch.testing.assert_close(value, b[key], rtol=0, atol=0)
        with self.assertRaises(ValueError): self.dataset(orientation_target="external")

    def test_invalid_orientation_masks_only_last_six_and_zeros_history(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        path = self.root / "data/chunk-000/episode_000000.parquet"
        columns = pq.read_table(path).to_pydict()
        for i in [18, 50, 54]: columns["teleop.smpl_frame_index"][i] = 0
        pq.write_table(pa.Table.from_pydict(columns), path)
        self.prepare(orientation_target="predicted_rotation6d")
        ds = self.dataset(orientation_target="predicted_rotation6d")
        row = ds[50]
        self.assertEqual(tuple(row["action_dim_is_pad"].shape), (47,))
        self.assertEqual(row["joint_orientation_is_pad"].nonzero().flatten().tolist(), [0, 2])
        self.assertEqual(row["past_joint_orientation_is_pad"].nonzero().flatten().tolist(), [0])
        self.assertTrue((row["action"][[0, 2], 41:] == 0).all())
        self.assertTrue((row["past_action"][0, 41:] == 0).all())
        actions = ds.episodes[0][1]
        np.testing.assert_array_equal(row["action"][:, :41], ds.normalizer.forward(actions[50:82:2], "action")[:, :41])
        self.assertFalse(row["action_is_pad"].any())
        self.assertTrue(ds[99]["joint_orientation_is_pad"][1:].all())

    def test_47_missing_validity_metadata_fails_41_is_unaffected(self):
        path = self.root / "meta/info.json"
        info = json.loads(path.read_text())
        del info["features"]["teleop.stream_mode"]
        path.write_text(json.dumps(info))
        with self.assertRaisesRegex(ValueError, "teleop.stream_mode"):
            md.load_metadata(self.root, orientation_target="predicted_rotation6d")
        self.assertEqual(md.load_metadata(self.root)["train"], [0, 1])

    def test_missing_required_wbc_and_reordered_names_fail_early(self):
        path = self.root / "meta/info.json"
        original = json.loads(path.read_text())
        wrong = deepcopy(original); del wrong["features"]["action.wbc"]
        path.write_text(json.dumps(wrong))
        with self.assertRaisesRegex(ValueError, "no implicit source fallback"):
            md.load_metadata(self.root)
        self.assertEqual(md.load_metadata(self.root, body_target="measured")["train"], [0, 1])
        wrong = deepcopy(original)
        wrong["features"]["action.wbc"]["names"][0] = "right_hip_pitch_joint"
        path.write_text(json.dumps(wrong))
        with self.assertRaisesRegex(ValueError, "canonical motor names"):
            md.load_metadata(self.root)

    def test_statistics_reject_parquet_mutation(self):
        self.prepare()
        path = self.root / "data/chunk-000/episode_000001.parquet"
        with path.open("ab") as stream: stream.write(b"mutated")
        with self.assertRaisesRegex(ValueError, "Parquet changed"):
            self.dataset()


def measured_source_contract():
    path=Path(__file__).resolve().parents[1]/"configs/wholebody/source_contracts/g1_existing_measured_v1.json"
    return json.loads(path.read_text())


class JointV2NumericTests(unittest.TestCase):
    def test_explicit_provenance_hash_and_no_direct_reference_masquerade(self):
        contract=measured_source_contract()
        with self.assertRaisesRegex(ValueError,"source_contract"):
            md.map_numeric(raw_columns(),body_target="measured",recipe_version="v2")
        self.assertEqual(make_joint_recipe(contract)["prediction_horizon"],40)
        wrong=deepcopy(contract);wrong["body_target"]="direct_reference";wrong["label_semantics"]="direct_joint_reference"
        with self.assertRaisesRegex(ValueError,"not direct reference"):
            make_joint_recipe(wrong)
        changed=deepcopy(contract);changed["provenance"]["revision"]+=" changed"
        self.assertNotEqual(source_contract_sha256(contract),source_contract_sha256(changed))

    def test_prediction_execution_and_true_history_have_independent_clocks(self):
        timing=md.timing_configuration(40,16,2,32,recipe_version="v2")
        self.assertEqual(timing["execution_horizon"],16)
        self.assertEqual(timing["lookahead_tail_horizon"],24)
        self.assertEqual(timing["future_action_offsets"],list(range(0,80,2)))
        self.assertAlmostEqual(timing["future_action_span_seconds"],2.6)
        self.assertEqual(timing["source_history_rule"],"history16_then_hold_last24")
        window=md.window_indices(50,150,40,16,2,32,recipe_version="v2")
        np.testing.assert_array_equal(window["past_action"],np.arange(18,50,2))
        np.testing.assert_array_equal(window["action"],np.arange(50,130,2))
        np.testing.assert_array_equal(window["image"],[50,82])
        with self.assertRaises(ValueError):md.timing_configuration(32,16,2,32,recipe_version="v2")

    def test_heading_conversion_roundtrips_and_removes_common_world_yaw(self):
        angles=np.asarray([.6,.7,3.1,-2.9])
        quat=np.stack((np.cos(angles/2),np.zeros(4),np.zeros(4),np.sin(angles/2)),axis=-1)
        relative=world_quaternion_to_heading_relative_rotation6d(quat,quat[0])
        np.testing.assert_allclose(relative[0],[1,0,0,0,1,0],atol=1e-6)
        result=heading_relative_rotation6d_to_world_quaternion(relative,quat[0])
        np.testing.assert_allclose(quaternion_matrix(result),quaternion_matrix(quat),atol=1e-6)
        np.testing.assert_array_equal(relative,world_quaternion_to_heading_relative_rotation6d(-quat,-quat[0]))
        with self.assertRaises(ValueError):heading_from_quaternion_wxyz(np.zeros(4))
        with self.assertRaises(ValueError):heading_relative_rotation6d_to_world_quaternion(np.zeros((4,6)),quat[0])


@unittest.skipUnless(importlib.util.find_spec("pyarrow"),"PyArrow required for immutable on-disk v2 checks")
class JointV2DatasetTests(unittest.TestCase):
    setUp=JointReferenceDatasetTests.setUp
    tearDown=JointReferenceDatasetTests.tearDown
    prepare=JointReferenceDatasetTests.prepare
    dataset=JointReferenceDatasetTests.dataset

    def write_fixture(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        JointReferenceDatasetTests.write_fixture(self,length=150)
        info=json.loads((self.root/"meta/info.json").read_text())
        info["features"]["observation.root_orientation"]={"shape":[4],"dtype":"float64","names":["base_qw","base_qx","base_qy","base_qz"]}
        info["features"]["action.original_joint_reference"]={"shape":[29],"dtype":"float32","names":list(md.BODY_JOINT_NAMES)}
        (self.root/"meta/info.json").write_text(json.dumps(info))
        for ep in [0,1]:
            path=self.root/f"data/chunk-000/episode_{ep:06d}.parquet"
            cols=pq.read_table(path).to_pydict();theta=np.arange(150)/50+.3
            cols["observation.root_orientation"]=np.stack((np.cos(theta/2),np.zeros(150),np.zeros(150),np.sin(theta/2)),axis=-1).tolist()
            cols["action.original_joint_reference"]=(np.asarray(cols["observation.state"])[:,md.BODY_INDICES]+.5).tolist()
            pq.write_table(pa.Table.from_pydict(cols),path)
        self.kwargs=dict(body_target="measured",orientation_target="predicted_rotation6d",action_horizon=40,
                         past_action_size=16,recipe_version="v2",execution_horizon=16,
                         source_contract=measured_source_contract())

    def test_preparation_hashes_all_sources_and_keeps_raw_files_unchanged(self):
        before={str(p):md.sha256(p) for p in self.root.rglob("*") if p.is_file()}
        audit=self.prepare(**self.kwargs)
        self.assertEqual(before,{str(p):md.sha256(p) for p in self.root.rglob("*") if p.is_file()})
        stats=json.loads((self.output/"dataset_stats.json").read_text())
        self.assertEqual(stats["normalization_strategy"],NORMALIZATION_V2)
        self.assertEqual(stats["orientation_prediction_frame"],"current_heading_relative")
        self.assertEqual(stats["action"][md.ACTION_KEY]["global_q01"][41:],[-1.]*6)
        self.assertFalse(audit["source_provenance_deployment_ready"])
        self.assertFalse(audit["original_sonic_reference_equivalence_verified"])
        self.assertEqual(stats["joint_recipe"],make_joint_recipe(measured_source_contract()))

    def test_real_history16_target40_current_heading_and_tail_masks(self):
        self.prepare(**self.kwargs);dataset=self.dataset(**self.kwargs);row=dataset[50]
        self.assertEqual(tuple(row["action"].shape),(40,47))
        self.assertEqual(tuple(row["past_action"].shape),(16,47))
        np.testing.assert_allclose(row["action"][0,41:],[1,0,0,0,1,0],atol=1e-6)
        self.assertTrue(all(indices==[50,82] for _,indices in self.decoder_calls))
        self.assertTrue(torch.equal(row["proprio"][0],row["proprio"][-1]))
        self.assertFalse(row["action_is_pad"].any())
        self.assertTrue(dataset[149]["action_is_pad"][1:].all())
        self.assertTrue((dataset[0]["past_action"]==0).all())

    def test_contract_mutation_or_wrong_target_fails_before_loading(self):
        self.prepare(**self.kwargs)
        wrong=deepcopy(self.kwargs);wrong["source_contract"]["provenance"]["revision"]+=" altered"
        with self.assertRaisesRegex(ValueError,"source|contract|configuration"):
            self.dataset(**wrong)

    @unittest.skipUnless(importlib.util.find_spec("hydra"),"Hydra required for real instantiate regression")
    def test_real_hydra_instantiation_accepts_nested_contract_without_convert_all(self):
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        self.prepare(**self.kwargs)
        cfg=OmegaConf.create({"_target_":"imagewam.datasets.wholebody_joint_reference.JointReferenceLeRobotDataset",
            "dataset_dir":str(self.root),"stats_path":str(self.output/"dataset_stats.json"),
            "qwen_text_cache_dir":str(self.cache),"qwen_context_len":4,"qwen_text_dim":8,**self.kwargs})
        dataset=instantiate(cfg)
        self.assertEqual(dataset.source_contract,measured_source_contract())
        self.assertEqual(dataset.timing["prediction_horizon"],40)
        self.assertEqual(dataset.timing["execution_horizon"],16)
        self.assertEqual(len(dataset),300)

    def test_direct_reference_reads_declared29_field_not_wbc_or_measured(self):
        contract=measured_source_contract();contract.update(body_target="direct_reference",
            label_semantics="direct_joint_reference",joint_position_key="action.original_joint_reference",
            joint_position_indices=list(range(29)))
        contract["provenance"]["synchronization_verified"]=True
        kwargs=dict(self.kwargs,body_target="direct_reference",source_contract=contract)
        self.prepare(**kwargs);dataset=self.dataset(**kwargs)
        state,action=dataset.episodes[0][:2]
        np.testing.assert_allclose(action[:,:29],state[:,:29]+.5,atol=1e-6)
        self.assertNotIn("action.wbc",md.numeric_columns("direct_reference","predicted_rotation6d",recipe_version="v2",source_contract=contract))


if __name__ == "__main__":
    unittest.main()
