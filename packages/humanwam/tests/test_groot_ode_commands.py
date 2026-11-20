"""Native ODE G1 CLI/config checks; no model or robot execution."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from humanwam.backends import groot_ode_commands as commands

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "packages/ode_g1"


@pytest.fixture
def options(tmp_path):
    native = tmp_path / "native config.yaml"
    native.write_text(yaml.safe_dump({
        "device": "cpu", "paths": {"dataset_root": "raw", "prepared": "prepared", "feature_cache": "features", "output": "train"},
        "base_model": {"path": "/local/gr00t", "cosmos_path": "/local/cosmos", "allow_download": False},
    }))
    wrapper = tmp_path / "wrapper.json"
    wrapper.write_text("{}")
    return {"native_config": str(native), "launcher_config": str(wrapper), "python": sys.executable}


def test_native_validator_preserves_local_path_resolution_without_model_import(options, tmp_path):
    cfg = commands.load_native_config(options, SOURCE)
    assert cfg["paths"]["dataset_root"] == str(tmp_path / "raw")
    assert cfg["head"]["horizon"] == 16
    assert (cfg["head"]["body_dim"], cfg["head"]["hand_dim"], cfg["head"]["state_dim"]) == (64, 12, 44)
    script = "import sys; from humanwam.backends.groot_ode_commands import load_native_config; load_native_config(" + repr(options) + "," + repr(str(SOURCE)) + "); assert 'torch' not in sys.modules and 'ode_g1' not in sys.modules"
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True)


@pytest.mark.parametrize("stage", ["inspect", "prepare", "features", "train"])
def test_stages_use_the_local_module_and_no_scheduler_or_shell(options, stage):
    command = commands.build_native_command(options, SOURCE, stage)
    assert command["argv"] == [sys.executable, "-m", "ode_g1", stage, "--config", options["native_config"]]
    assert command["cwd"] == str(SOURCE)
    assert command["env"]["HF_HUB_OFFLINE"] == "1"
    first = ROOT / "third_party/Isaac-GR00T-ODE" if stage in {"features", "train"} else SOURCE / "src"
    assert command["env"]["PYTHONPATH"].split(os.pathsep)[0] == str(first)


def test_built_inspect_command_runs_against_local_cli(options):
    command = commands.build_native_command(options, SOURCE, "inspect")
    result = subprocess.run(command["argv"], cwd=command["cwd"], env={**os.environ, **command["env"]}, capture_output=True, text=True, check=True)
    actual = json.loads(result.stdout)
    assert actual["head"]["horizon"] == 16
    assert actual["base_model"]["allow_download"] is False


def test_overrides_are_serialized_as_single_native_arguments_and_validated(options):
    options["native_overrides"] = {"training.batch_size": 8, "dataset.instruction": "place 'object' $(unchanged)", "device": "cpu"}
    command = commands.build_train_command(options, SOURCE)
    assert 'training.batch_size=8' in command["argv"]
    assert 'dataset.instruction="place \'object\' $(unchanged)"' in command["argv"]
    cfg = commands.load_native_config(options, SOURCE)
    assert cfg["training"]["batch_size"] == 8
    assert cfg["dataset"]["instruction"] == "place 'object' $(unchanged)"
    for values in ({"head.hand_dim": 14}, {"head.horizon": 8}, {"dataset.camera_keys": ["observation.images.chest", "observation.images.ego_view"]}, {"base_model.allow_download": True}, {"--device": "cpu"}, {"does.not.exist": 1}, {"training.learning_rate": float("nan")}):
        with pytest.raises(ValueError):
            commands.build_train_command({**options, "native_overrides": values}, SOURCE)


def test_single_process_recipe_rejects_ddp_and_robot_execution(options):
    for count in (0, 2, True, "1", 1.0):
        with pytest.raises(ValueError, match="one optimizer"):
            commands.build_train_command({**options, "num_processes": count}, SOURCE)
    with pytest.raises(ValueError):
        commands.build_native_command(options, SOURCE, "deploy")
    check = commands.build_native_command(options, SOURCE, "deploy-check")
    assert check["argv"][3] == "deploy"
    assert check["argv"][-1] == "--check"


def test_model_stages_reject_the_wrong_gr00t_revision_or_changed_tracked_source(options, monkeypatch):
    def wrong_commit(argv, **kwargs):
        value = "7aa0bca50f1097f8b17f5c274c98264b72a1419f"
        return subprocess.CompletedProcess(argv, 0, stdout=value + "\n" if "rev-parse" in argv else "")

    monkeypatch.setattr(commands.subprocess, "run", wrong_commit)
    with pytest.raises(ValueError, match="separate Isaac-GR00T"):
        commands.build_train_command(options, SOURCE)

    def dirty_source(argv, **kwargs):
        value = commands.GR00T_SOURCE_COMMIT + "\n" if "rev-parse" in argv else " M gr00t/model/gr00t_n1d7/gr00t_n1d7.py\n"
        return subprocess.CompletedProcess(argv, 0, stdout=value)

    monkeypatch.setattr(commands.subprocess, "run", dirty_source)
    with pytest.raises(ValueError, match="local changes"):
        commands.build_native_command(options, SOURCE, "features")


def test_local_package_can_be_edited_and_used_without_an_external_git_checkout(options, tmp_path):
    local = tmp_path / "our-package"
    shutil.copytree(SOURCE / "src", local / "src")
    config_file = local / "src/ode_g1/config.py"
    config_file.write_text(config_file.read_text() + "\n# Local development belongs to HumanWAM.\n")
    assert not (local / ".git").exists()
    command = commands.build_native_command(options, local, "inspect")
    result = subprocess.run(command["argv"], cwd=command["cwd"], env={**os.environ, **command["env"]}, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["head"]["horizon"] == 16
    assert command["env"]["PYTHONPATH"].split(os.pathsep)[0] == str(local / "src")


def test_serve_uses_policy_rpc_and_loopback_default(options):
    command = commands.build_serve_command(options, SOURCE)
    assert command["argv"] == [sys.executable, "-m", "humanwam.sonic_policy_server", "--config", options["launcher_config"], "--host", "127.0.0.1", "--port", "5558"]
    for port in (0, 65536, True, "5558"):
        with pytest.raises(ValueError, match="port"):
            commands.build_serve_command({**options, "port": port}, SOURCE)


def test_offline_infer_uses_native_episode_and_camera_flags_without_issuing_actions(options, tmp_path):
    observation = tmp_path / "current.npz"
    observation.write_bytes(b"not loaded during pure command construction")
    image = tmp_path / "ego.png"
    image.touch()
    target = tmp_path / "future prediction.npz"
    config = {**options, "input": str(observation), "output": str(target), "episode_start": True,
              "images": {"observation.images.ego_view": str(image)}, "instruction": "put pillow away"}
    command = commands.build_native_command(config, SOURCE, "infer")
    assert command["argv"][3] == "infer"
    assert "--episode-start" in command["argv"]
    assert "observation.images.ego_view=" + str(image) in command["argv"]
    assert command["argv"][-2:] == ["--instruction", "put pillow away"]
    assert not target.exists()
    target.touch()
    with pytest.raises(ValueError, match="new inference"):
        commands.build_native_command(config, SOURCE, "infer")


def test_shipped_environment_config_is_valid_and_offline(tmp_path, monkeypatch):
    for key, value in {
        "GROOT_ODE_DATA_ROOT": str(tmp_path / "raw"), "GROOT_ODE_RUN_ROOT": str(tmp_path / "run"),
        "GROOT_ODE_BASE_MODEL": str(tmp_path / "gr00t"), "GROOT_ODE_COSMOS": str(tmp_path / "cosmos"),
    }.items():
        monkeypatch.setenv(key, value)
    cfg = commands.load_native_config({"native_config": str(ROOT / "configs/humanwam/groot-ode.yaml")}, SOURCE)
    assert cfg["paths"]["output"] == str(tmp_path / "run/train")
    assert cfg["base_model"]["allow_download"] is False
    assert cfg["inference"]["verify_head_sha256"] is True
