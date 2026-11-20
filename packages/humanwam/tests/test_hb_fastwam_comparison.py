"""Matched B6-v6 launch plans are explicit artifacts, never job submissions."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("hb_fastwam_compare", ROOT / "hb_training/fastwam_tools/hb_fastwam_compare.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.fixture
def inputs(tmp_path):
    source = tmp_path / "FastWAM"
    config = source / "scripts/accelerate_configs/accelerate_zero1_ds.yaml"
    config.parent.mkdir(parents=True)
    config.touch()
    data = tmp_path / "b6v6"
    (data / "meta").mkdir(parents=True)
    (data / "meta/hb_fastwam_interface.json").write_text(json.dumps({"profile": "g1", "state_dim": 44, "action_dim": 76, "cameras": ["ego_view", "chest"]}))
    cache = tmp_path / "models"
    text = tmp_path / "text"
    cache.mkdir()
    text.mkdir()
    init = tmp_path / "init.pt"
    init.write_bytes(b"fixture-not-model-weights")
    chosen = tmp_path / "chosen.json"
    chosen.write_text('{"per_gpu_batch":8}')
    (tmp_path / "fastwam.ok").touch()
    return dict(python=sys.executable, fastwam_source=source, data_root=data, action_init=init,
                model_cache=cache, text_cache=text, chosen=chosen, output_root=tmp_path / "new-comparison",
                master_addr="allocation-master.example")


def test_matched_plans_share_inputs_training_budget_and_seed_without_writes(inputs):
    plan = MODULE.build_comparison_plan(**inputs)
    assert plan["task"] == "b6v6"
    runs = plan["runs"]
    assert set(runs) == {"hb_v2", "unified"}
    assert not inputs["output_root"].exists()
    assert plan["global_batch"] == 256
    left = runs["hb_v2"]["ranks"][0]
    right = runs["unified"]["ranks"][0]
    assert left["env"]["HB_AUG_PROFILE"] == "hb_v2"
    assert right["env"]["HB_AUG_PROFILE"] == "unified"
    assert left["env"]["HB_INLOOP_EVERY"] == right["env"]["HB_INLOOP_EVERY"] == "0"
    assert all(left["env"][name] == right["env"][name] == "" for name in ("HB_MODEL_YAML", "HB_COMPOSE_ONLY", "HB_DATA_SAMPLE"))
    assert left["env"]["HB_AUG_SEED"] == right["env"]["HB_AUG_SEED"] == "42"
    assert left["unset_env"] == right["unset_env"]
    assert {"RANK", "MASTER_ADDR", "MASTER_PORT"} <= set(left["unset_env"])
    assert left["unset_env_prefixes"] == right["unset_env_prefixes"] == ["PET_"]
    for profile, run in runs.items():
        assert len(run["ranks"]) == 4
        assert run["output_dir"] == str(inputs["output_root"] / profile)
        for rank, command in enumerate(run["ranks"]):
            args = command["argv"]
            assert args[args.index("--machine_rank") + 1] == str(rank)
            assert "max_steps=4000" in args and "seed=42" in args and "batch_size=8" in args
            assert "save_every=2000" in args
    def strip_run_specific(args):
        port_index = args.index("--main_process_port") + 1
        return [value for index, value in enumerate(args) if index != port_index and not value.startswith("output_dir=")]
    assert strip_run_specific(left["argv"]) == strip_run_specific(right["argv"])


def test_output_reuse_and_wrong_interface_are_rejected(inputs):
    target = inputs["output_root"] / "unified"
    target.mkdir(parents=True)
    with pytest.raises(ValueError, match="fresh"):
        MODULE.build_comparison_plan(**inputs)
    target.rmdir()
    (inputs["data_root"] / "meta/hb_fastwam_interface.json").write_text('{"profile":"g1","state_dim":46}')
    with pytest.raises(ValueError, match="interface"):
        MODULE.build_comparison_plan(**inputs)


def test_plan_requires_passed_batch_benchmark(inputs):
    inputs["chosen"].with_name("fastwam.ok").unlink()
    with pytest.raises(ValueError, match="benchmark"):
        MODULE.build_comparison_plan(**inputs)


def test_plan_keeps_virtualenv_python_symlink(inputs, tmp_path):
    python = tmp_path / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    plan = MODULE.build_comparison_plan(**{**inputs, "python": python})
    assert plan["runs"]["unified"]["ranks"][0]["argv"][0] == str(python)
    assert plan["runs"]["hb_v2"]["ranks"][0]["env"]["PATH"].split(MODULE.os.pathsep)[0] == str(python.parent)
