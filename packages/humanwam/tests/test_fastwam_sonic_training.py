"""CPU checks of actual pinned Hydra configuration and real compact-data I/O."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from humanwam.backends import fastwam_sonic_data as data
from humanwam.backends import fastwam_sonic_training as training

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "third_party/FastWAM"


def columns(n=3):
    values = {key: np.zeros((n, width), dtype=np.float32) for key, width in data.RAW_DIMS.items()}
    values["observation.state"][:] = np.arange(43)
    values["action.motion_token"][:] = np.arange(64) + 100
    values["observation.inspire_left_hand_state"][:] = np.arange(6) / 10
    values["observation.inspire_right_hand_state"][:] = np.arange(6) / 10 + 0.3
    values["observation.projected_gravity"][:] = [0.0, 0.0, -1.0]
    return values


def info_fixture():
    return {
        "codebase_version": "v2.1", "fps": 30, "total_frames": 3,
        "total_episodes": 1, "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "features": {
            **{key: {"shape": [width], "dtype": "float32"} for key, width in data.RAW_DIMS.items()},
            **{"observation.images." + key: {"dtype": "video", "shape": [224, 224, 3]} for key in ("ego_view", "chest")},
        },
    }


def write_compact_metadata(root):
    (root / "meta").mkdir(parents=True)
    info = info_fixture()
    info["features"]["action"] = {"shape": [76], "dtype": "float32"}
    info["features"]["observation.state"] = {"shape": [44], "dtype": "float32"}
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta" / data.RECEIPT).write_text(json.dumps({
        "version": 1, "profile": "feedback76_h16", "mapping": data.MAPPING,
        "action_frequency_hz": 30,
    }))


def test_measured_mapping_matches_groot_and_never_uses_binary_command_hands():
    values = columns()
    values["action"] = np.full((3, 78), 99.0)
    actions, states = data.compact_feedback76(values)
    assert actions.shape == (3, 76)
    assert states.shape == (3, 44)
    np.testing.assert_array_equal(actions[:, :64], values["action.motion_token"])
    np.testing.assert_array_equal(actions[:, 64:70], values["observation.inspire_left_hand_state"])
    np.testing.assert_array_equal(states[:, :22], values["observation.state"][:, :22])
    np.testing.assert_array_equal(states[:, 22:29], values["observation.state"][:, 29:36])
    np.testing.assert_array_equal(states[:, 29:41], actions[:, 64:76])
    np.testing.assert_array_equal(states[:, 41:], values["observation.projected_gravity"])


@pytest.mark.parametrize("key,bad", [
    ("observation.state", np.zeros((3, 44))),
    ("action.motion_token", np.zeros((2, 64))),
    ("observation.inspire_left_hand_state", np.ones((3, 6)) * 1000),
    ("observation.projected_gravity", np.full((3, 3), np.nan)),
])
def test_mapping_rejects_shape_nonfinite_and_implicit_hand_scaling(key, bad):
    values = columns()
    values[key] = bad
    with pytest.raises(ValueError):
        data.compact_feedback76(values)


def test_compact_metadata_requires_recorded_semantics_not_just_width(tmp_path):
    write_compact_metadata(tmp_path)
    assert data.audit_compact_dataset(tmp_path)["action_frequency_hz"] == 30
    receipt = tmp_path / "meta" / data.RECEIPT
    receipt.write_text(json.dumps({"version": 1, "profile": "feedback76_h16"}))
    with pytest.raises(ValueError, match="receipt"):
        data.audit_compact_dataset(tmp_path)


@pytest.mark.skipif(not (SOURCE / "configs/train.yaml").is_file(), reason="Pinned source absent")
def test_actual_native_config_composes_as_76_44_h16_without_torch():
    from omegaconf import OmegaConf

    cfg = training.compose_sonic_config(SOURCE)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert resolved["model"]["_target_"] == "fastwam.runtime.create_fastwam"
    assert resolved["model"]["proprio_dim"] == 44
    assert resolved["model"]["action_dit_config"]["action_dim"] == 76
    train = resolved["data"]["train"]
    assert train["num_frames"] - 1 == 16
    assert train["action_video_freq_ratio"] == 4
    assert (train["num_frames"] - 1) // train["action_video_freq_ratio"] == 4
    assert train["processor"]["action_output_dim"] == 76
    assert train["processor"]["proprio_output_dim"] == 44
    assert not any(train["processor"]["delta_action_dim_mask"]["default"])
    assert [x["key"] for x in train["shape_meta"]["images"]] == ["ego_view", "chest"]
    assert resolved["model"]["action_scheduler"]["train_shift"] == 1.0
    assert resolved["model"]["action_scheduler"]["infer_shift"] == 1.0
    script = "from humanwam.backends.fastwam_sonic_training import compose_sonic_config; import sys; compose_sonic_config(" + repr(str(SOURCE)) + "); assert 'torch' not in sys.modules and 'fastwam' not in sys.modules"
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True)


@pytest.mark.skipif(not (SOURCE / "configs/train.yaml").is_file(), reason="Pinned source absent")
@pytest.mark.parametrize("key,value", [("data.train.num_frames", 33), ("data.train.processor.action_output_dim", 7), ("model.action_scheduler.infer_shift", 5.0)])
def test_protected_recipe_rejects_partial_relabeling(key, value):
    with pytest.raises(ValueError, match="Protected"):
        training.compose_sonic_config(SOURCE, {key: value})


def test_launch_constructs_real_accelerate_module_without_writing_output(tmp_path):
    source = tmp_path / "native"
    (source / "scripts").mkdir(parents=True)
    (source / "scripts/train.py").touch()
    (source / "configs").mkdir()
    (source / "configs/train.yaml").touch()
    prepared = tmp_path / "prepared"
    write_compact_metadata(prepared)
    initializer = tmp_path / "init.pt"
    initializer.write_bytes(b"not-model-weights")
    launcher = tmp_path / "train.json"
    launcher.write_text("{}")
    text = tmp_path / "text"
    text.mkdir()
    options = {
        "dataset_dir": str(prepared), "text_embedding_cache_dir": str(text),
        "action_init": str(initializer), "output_dir": str(tmp_path / "new-output"),
        "launcher_config": str(launcher), "model_cache_root": str(tmp_path),
        "num_processes": 2,
    }
    result = training.build_train_command(options, source)
    assert result["argv"][1:3] == ["-m", "accelerate.commands.launch"]
    assert "humanwam.backends.fastwam_sonic_training" in result["argv"]
    assert result["env"]["DIFFSYNTH_SKIP_DOWNLOAD"] == "true"
    assert not (tmp_path / "new-output").exists()
    for extras in ({"method": "delta_flow"}, {"overrides": {"model.proprio_dim": 8}}, {"horizon": 16}, {"num_processes": True}):
        with pytest.raises(ValueError):
            training.build_train_command({**options, **extras}, source)


@pytest.mark.skipif(importlib.util.find_spec("pyarrow") is None, reason="Optional parquet I/O dependency absent")
def test_real_parquet_conversion_is_separate_and_corrects_features_stats_and_media(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    source = tmp_path / "source"
    (source / "meta").mkdir(parents=True)
    info = info_fixture()
    (source / "meta/info.json").write_text(json.dumps(info))
    (source / "meta/episodes.jsonl").write_text(json.dumps({"episode_index": 0, "length": 3}) + "\n")
    (source / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "put pillow away"}) + "\n")
    (source / "meta/episodes_stats.jsonl").write_text(json.dumps({"episode_index": 0, "stats": {"unchanged": {"min": [2]}}}) + "\n")
    (source / "meta/stats.json").write_text('{"stale":true}')
    (source / "videos").mkdir()
    table = pa.table({key: value.tolist() for key, value in columns().items()})
    hf = {"info": {"features": {key: {"_type": "Sequence"} for key in columns()}}}
    table = table.replace_schema_metadata({b"huggingface": json.dumps(hf).encode()})
    relative = Path("data/chunk-000/episode_000000.parquet")
    (source / relative).parent.mkdir(parents=True)
    pq.write_table(table, source / relative)
    before = (source / relative).read_bytes()
    output = tmp_path / "prepared"
    receipt = data.prepare_dataset(source, output)
    assert receipt["action_frequency_hz"] == 30
    assert (source / relative).read_bytes() == before
    assert (output / "videos").is_symlink()
    assert not (output / "meta/stats.json").exists()
    table_out = pq.read_table(output / relative)
    expected_actions, expected_states = data.compact_feedback76(columns())
    np.testing.assert_array_equal(table_out["action"].to_pylist(), expected_actions)
    np.testing.assert_array_equal(table_out["observation.state"].to_pylist(), expected_states)
    updated = json.loads(table_out.schema.metadata[b"huggingface"])
    assert updated["info"]["features"]["action"]["length"] == 76
    new_stats = json.loads((output / "meta/episodes_stats.jsonl").read_text())
    assert len(new_stats["stats"]["action"]["mean"]) == 76
    assert new_stats["stats"]["unchanged"] == {"min": [2]}
    data.audit_compact_dataset(output)
    with pytest.raises(ValueError, match="fresh"):
        data.prepare_dataset(source, output)
