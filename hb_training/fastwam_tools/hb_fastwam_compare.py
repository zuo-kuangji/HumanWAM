"""Build matched B6-v6 FastWAM augmentation plans without starting any jobs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

DISTRIBUTED_ENV = (
    "RANK", "WORLD_SIZE", "LOCAL_RANK", "GROUP_RANK", "ROLE_RANK", "LOCAL_WORLD_SIZE",
    "MASTER_ADDR", "MASTER_PORT", "NPROC_PER_NODE", "WORLD_SIZE_WITH_STANDBY", "TORCHELASTIC_RUN_ID",
)


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def build_comparison_plan(*, python, fastwam_source, data_root, action_init, model_cache,
                          text_cache, chosen, output_root, master_addr, machines=4,
                          port=31300, steps=4000, seed=42):
    """Read the B6 prepared interface/benchmark and return explicit per-rank argv.

    This function neither creates run directories nor contacts a cluster. A
    path/metadata preflight is not evidence of a GPU allocation or model load.
    """
    _positive(machines, "machines")
    _positive(steps, "steps")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(port) is not int or not 1 <= port <= 65525:
        raise ValueError("port and port+10 must be valid TCP ports")
    if not isinstance(master_addr, str) or not master_addr.strip():
        raise ValueError("Provide the allocated cluster's master address")
    paths = {name: Path(value).expanduser().resolve(strict=True) for name, value in {
        "fastwam_source": fastwam_source, "data_root": data_root,
        "action_init": action_init, "model_cache": model_cache, "text_cache": text_cache,
        "chosen": chosen,
    }.items()}
    # The invoked location of a symlinked virtualenv interpreter selects its
    # pyvenv.cfg. Resolving it would launch the system interpreter instead.
    paths["python"] = Path(python).expanduser().absolute()
    if not os.access(paths["python"], os.X_OK):
        raise ValueError("python must be an executable interpreter")
    for name in ("python", "action_init", "chosen"):
        if not paths[name].is_file():
            raise ValueError(f"{name} must be a file")
    for name in ("fastwam_source", "data_root", "model_cache", "text_cache"):
        if not paths[name].is_dir():
            raise ValueError(f"{name} must be a directory")
    if not (paths["fastwam_source"] / "scripts/accelerate_configs/accelerate_zero1_ds.yaml").is_file():
        raise ValueError("Native FastWAM ZeRO-1 launch configuration missing")
    if not (paths["chosen"].parent / "fastwam.ok").is_file():
        raise ValueError("The FastWAM batch benchmark must have passed (fastwam.ok)")
    benchmark = json.loads(paths["chosen"].read_text())
    batch = _positive(benchmark["per_gpu_batch"], "benchmarked per_gpu_batch")
    interface = json.loads((paths["data_root"] / "meta/hb_fastwam_interface.json").read_text())
    expected = {"profile": "g1", "state_dim": 44, "action_dim": 76, "cameras": ["ego_view", "chest"]}
    if any(interface.get(key) != value for key, value in expected.items()):
        raise ValueError("Comparison requires the prepared B6-v6 G1 state44/action76/ego+chest interface")
    if paths["data_root"].name != "b6v6":
        raise ValueError("Use the existing prepared b6v6 dataset for this comparison")
    out = Path(output_root).expanduser().resolve()
    outputs = {profile: out / profile for profile in ("hb_v2", "unified")}
    for target in outputs.values():
        if target.exists():
            raise ValueError(f"Comparison outputs must be fresh: {target}")
        if paths["data_root"] in target.parents or target in paths["data_root"].parents:
            raise ValueError("Comparison outputs must be outside the dataset")
    entry = Path(__file__).resolve().with_name("hb_fastwam_train.py")
    hb_root = entry.parents[1]
    package_src = hb_root.parent / "packages/humanwam/src"
    shared = [
        f"batch_size={batch}", f"max_steps={steps}", f"seed={seed}", "log_every=20",
        f"save_every={min(2000, steps)}", "eval_every=0", "num_workers=8",
        "wandb.enabled=false", "gradient_accumulation_steps=1",
        "data.train.dataset_dirs=" + json.dumps([str(paths["data_root"])]),
        "data.train.text_embedding_cache_dir=" + json.dumps(str(paths["text_cache"])),
        "model.action_dit_pretrained_path=" + json.dumps(str(paths["action_init"])),
    ]
    runs = {}
    for offset, profile in enumerate(outputs):
        environment = {
            "FASTWAM_SRC": str(paths["fastwam_source"]), "HB_FW_PROFILE": "g1", "HB_FW_JOINTS": "29",
            "HB_FW_AUG": "1", "HB_AUG_PROFILE": profile, "HB_AUG_SEED": str(seed),
            "HB_INLOOP_EVERY": "0", "HB_MODEL_YAML": "", "HB_COMPOSE_ONLY": "", "HB_DATA_SAMPLE": "",
            "HB_AUG_DIR": str(hb_root / "hb_groot"),
            "DIFFSYNTH_MODEL_BASE_PATH": str(paths["model_cache"]), "DIFFSYNTH_SKIP_DOWNLOAD": "true",
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "WANDB_MODE": "disabled",
            "PYTHONPATH": os.pathsep.join([str(package_src), str(paths["fastwam_source"] / "src")]),
            "PATH": str(paths["python"].parent) + os.pathsep + os.environ.get("PATH", ""),
        }
        ranks = []
        for rank in range(machines):
            argv = [
                str(paths["python"]), "-m", "accelerate.commands.launch", "--config_file",
                str(paths["fastwam_source"] / "scripts/accelerate_configs/accelerate_zero1_ds.yaml"),
                "--deepspeed_multinode_launcher", "standard", "--num_machines", str(machines),
                "--machine_rank", str(rank), "--main_process_ip", master_addr, "--main_process_port", str(port + offset * 10),
                "--num_processes", str(machines * 8), str(entry),
                "output_dir=" + json.dumps(str(outputs[profile])), *shared,
            ]
            ranks.append({
                "argv": argv, "cwd": str(paths["fastwam_source"]), "env": environment,
                "unset_env": list(DISTRIBUTED_ENV), "unset_env_prefixes": ["PET_"],
            })
        runs[profile] = {"output_dir": str(outputs[profile]), "ranks": ranks}
    return {
        "task": "b6v6", "model_profile": "g1", "horizon": 16,
        "per_gpu_batch": batch, "global_batch": batch * machines * 8,
        "max_steps": steps, "model_seed": seed, "data": interface,
        "same_input_paths": {key: str(value) for key, value in paths.items()},
        "runs": runs, "jobs_started": False, "models_loaded": False, "cluster_allocation_verified": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("python", "fastwam-source", "data-root", "action-init", "model-cache", "text-cache", "chosen", "output-root", "master-addr"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--machines", type=int, default=4)
    parser.add_argument("--port", type=int, default=31300)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    print(json.dumps(build_comparison_plan(**vars(args)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
