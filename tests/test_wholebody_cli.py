"""CPU augmentation provenance, raw joint smoke windows and epoch-prefetch checks."""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

from imagewam.augmentation_config import (
    augmentation_sha256,
    default_augmentation,
    normalize_augmentation,
)

ROOT = Path(__file__).resolve().parents[1]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load_file("build_aug_test", ROOT / "scripts/wholebody/build_config.py")
training = load_file("train_aug_test", ROOT / "scripts/wholebody/train.py")
fixtures = load_file(
    "cli_fixtures_aug_test", ROOT / "tests/test_wholebody_training_cli.py"
)


def with_profile(cfg, profile=None):
    profile = (
        default_augmentation() if profile is None else normalize_augmentation(profile)
    )
    cfg["data"]["train"]["augmentation"] = copy.deepcopy(profile)
    cfg["wholebody_contract"].update(
        training_augmentation=copy.deepcopy(profile),
        training_augmentation_sha256=augmentation_sha256(profile),
    )
    return cfg


def test_legacy_configs_without_any_new_fields_remain_disabled(tmp_path):
    cfg = fixtures.fixture_config(tmp_path)
    result = training.training_augmentation_contract(cfg)
    assert result["legacy"] and not result["config"]["enabled"]
    assert result["sha256"] == augmentation_sha256(None)
    training.validate_config(cfg, smoke=False)
    assert not training.training_augmentation_contract(cfg, dataset=SimpleNamespace())[
        "config"
    ]["enabled"]


@pytest.mark.parametrize(
    "drift", ["hash", "data", "partial", "missing", "noncanonical"]
)
def test_profile_hash_and_dataset_copy_cannot_silently_diverge(tmp_path, drift):
    cfg = with_profile(fixtures.fixture_config(tmp_path))
    if drift == "hash":
        cfg["wholebody_contract"]["training_augmentation_sha256"] = "0" * 64
    elif drift == "data":
        cfg["data"]["train"]["augmentation"]["proprio"]["hand_noise_std"] = 0.009
    elif drift == "partial":
        del cfg["wholebody_contract"]["training_augmentation"]
    elif drift == "missing":
        del cfg["data"]["train"]["augmentation"]
    else:
        cfg["data"]["train"]["augmentation"] = {"enabled": True}
    with pytest.raises(RuntimeError, match="augmentation"):
        training.validate_config(cfg, smoke=False)


def test_actual_dataset_configuration_and_enabled_flag_are_verified(tmp_path):
    cfg = with_profile(fixtures.fixture_config(tmp_path))
    profile = cfg["data"]["train"]["augmentation"]
    actual = SimpleNamespace(config=copy.deepcopy(profile), enabled=True)
    dataset = SimpleNamespace(augmentation=actual)
    assert (
        training.training_augmentation_contract(cfg, dataset=dataset)["config"]
        == profile
    )
    actual.enabled = False
    with pytest.raises(RuntimeError, match="Actual dataset"):
        training.training_augmentation_contract(cfg, dataset=dataset)
    actual.enabled = True
    actual.config["visual"]["blur"]["probability"] = 0.0
    with pytest.raises(RuntimeError, match="Actual dataset"):
        training.training_augmentation_contract(cfg, dataset=dataset)
    with pytest.raises(RuntimeError, match="Actual dataset"):
        training.training_augmentation_contract(cfg, dataset=SimpleNamespace())


def build_fixture(root, **options):
    from omegaconf import OmegaConf

    for directory in ("dataset", "cache", "flux", "qwen"):
        (root / directory).mkdir()
    for filename in ("flux.safetensors", "ae.safetensors", "initializer.pt"):
        (root / filename).write_bytes(
            b"Synthetic config fixture; never loaded as a model"
        )
    stats = fixtures.valid_stats()
    stats.update(train_frames=567, train_episode_ids=[0, 1])
    (root / "stats.json").write_text(json.dumps(stats))
    args = SimpleNamespace(
        dataset_dir=root / "dataset",
        stats_path=root / "stats.json",
        text_cache_dir=root / "cache",
        output_dir=root / "newrun",
        flux2_src=root / "flux",
        flux2_model=root / "flux.safetensors",
        ae_model=root / "ae.safetensors",
        qwen_model=root / "qwen",
        action_initializer=root / "initializer.pt",
        epochs=7,
        batch_size=3,
        accumulation=2,
        num_processes=4,
        **options,
    )

    def measure(cfg):
        profile = OmegaConf.to_container(cfg.data.train.augmentation, resolve=True)
        assert profile == OmegaConf.to_container(
            cfg.wholebody_contract.training_augmentation, resolve=True
        )
        assert (
            augmentation_sha256(profile)
            == cfg.wholebody_contract.training_augmentation_sha256
        )
        return {
            "dataset_frames": 567,
            "raw_loader_batches": 189,
            "per_rank_microbatches": 48,
            "world_size": 4,
            "microbatch": 3,
            "gradient_accumulation": 2,
            "dataloader_drop_last": False,
        }

    receipt = builder.build(args, measure_loader=measure)
    production = OmegaConf.load(root / "newrun/production.yaml")
    smoke = OmegaConf.load(root / "newrun/smoke.yaml")
    training.validate_config(production, smoke=False)
    training.validate_config(smoke, smoke=True)
    return receipt, production, smoke


def test_new_builder_defaults_to_the_shared_enabled_profile_and_binds_both_configs(
    tmp_path,
):
    from omegaconf import OmegaConf

    receipt, production, smoke = build_fixture(tmp_path)
    shared = normalize_augmentation(
        OmegaConf.to_container(
            OmegaConf.load(ROOT / "configs/wholebody/augmentation_v1.yaml"),
            resolve=True,
        )
    )
    assert shared == default_augmentation()
    assert receipt["training_augmentation"] == shared
    assert receipt["training_augmentation_sha256"] == augmentation_sha256(shared)
    assert Path(receipt["training_augmentation_profile"]).name == "augmentation_v1.yaml"
    assert production.data.train.augmentation == smoke.data.train.augmentation == shared


def test_builder_explicit_opt_out_keeps_full_disabled_provenance(tmp_path):
    receipt, production, _ = build_fixture(tmp_path, no_augmentation=True)
    assert not production.data.train.augmentation.enabled
    assert receipt["training_augmentation"] == normalize_augmentation(None)
    assert receipt["training_augmentation_profile"] is None


def test_custom_profile_is_canonicalized_and_cli_options_are_mutually_exclusive(
    tmp_path,
):
    custom = tmp_path / "custom.yaml"
    custom.write_text(
        "enabled: true\nseed: 18\nvisual:\n  camera_mode: shared\nproprio:\n  enabled: false\n"
    )
    receipt, production, _ = build_fixture(tmp_path, augmentation_config=custom)
    assert production.data.train.augmentation.seed == 18
    assert production.data.train.augmentation.visual.camera_mode == "shared"
    assert not production.data.train.augmentation.proprio.enabled
    assert receipt["training_augmentation_sha256"] == augmentation_sha256(
        receipt["training_augmentation"]
    )
    with pytest.raises(ValueError, match="Choose"):
        builder.resolve_training_augmentation(
            SimpleNamespace(augmentation_config=custom, no_augmentation=True)
        )
    arguments = [
        "--dataset-dir",
        "/data",
        "--stats-path",
        "/stats",
        "--text-cache-dir",
        "/cache",
        "--output-dir",
        "/new",
        "--flux2-src",
        "/flux",
        "--flux2-model",
        "/model",
        "--ae-model",
        "/ae",
        "--qwen-model",
        "/qwen",
        "--action-initializer",
        "/init",
    ]
    with pytest.raises(SystemExit):
        builder.argument_parser().parse_args(
            arguments + ["--no-augmentation", "--augmentation-config", str(custom)]
        )


def smoke_report(cfg):
    return {
        "status": "FIVE_UPDATES_VERIFIED",
        "actual_optimizer_updates": 5,
        "skipped_updates": 0,
        "world_size": cfg["wholebody_contract"]["world_size"],
        "microbatch": cfg["batch_size"],
        "accumulation": cfg["gradient_accumulation_steps"],
        "action_horizon": cfg["model"]["expected_action_horizon"],
        "action_dim": cfg["model"]["expected_action_dim"],
        "statistics_sha256": cfg["wholebody_contract"]["statistics_sha256"],
        "action_initializer_sha256": cfg["model"]["action_initializer_sha256"],
        "schedule_total_updates": cfg["lr_scheduler_total_steps"],
    }


def test_new_smoke_gate_binds_augmentation_but_legacy_disabled_receipts_still_work(
    tmp_path,
):
    cfg = fixtures.fixture_config(tmp_path)
    directory = Path(cfg["wholebody_contract"]["smoke_output_dir"])
    directory.mkdir()
    report = smoke_report(cfg)

    def save():
        for rank in range(cfg["wholebody_contract"]["world_size"]):
            (directory / f"smoke_rank_{rank}.json").write_text(json.dumps(report))

    save()
    training.require_completed_smoke(cfg)
    with_profile(cfg)
    with pytest.raises(RuntimeError, match="augmentation"):
        training.require_completed_smoke(cfg)
    report.update(
        training_augmentation=cfg["data"]["train"]["augmentation"],
        training_augmentation_sha256=cfg["wholebody_contract"][
            "training_augmentation_sha256"
        ],
    )
    save()
    training.require_completed_smoke(cfg)
    report["training_augmentation_sha256"] = "f" * 64
    save()
    with pytest.raises(RuntimeError, match="augmentation"):
        training.require_completed_smoke(cfg)


def trainer_methods():
    """Execute the real iterator/train methods without importing heavy model dependencies."""
    tree = ast.parse((ROOT / "src/imagewam/trainer.py").read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Wan22Trainer"
    )
    functions = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_training_data_iterator", "train"}
    ]
    namespace = {
        "logger": SimpleNamespace(info=lambda *args, **kwargs: None),
        "time": SimpleNamespace(perf_counter=lambda: 0.0),
    }
    exec(  # noqa: S102 - Execute only the two fixed local repository AST methods above.
        compile(
            ast.Module(body=functions, type_ignores=[]),
            "native_trainer_methods",
            "exec",
        ),
        namespace,
    )
    return namespace


def test_dataset_epoch_is_published_before_first_prefetch_and_epoch_rollover():
    methods = trainer_methods()
    events = []

    class FinishedProbe(Exception):
        pass

    class Loader:
        def __iter__(self):
            events.append(("iterator", trainer.epoch))
            if trainer.epoch:
                raise FinishedProbe
            return iter(())

    trainer = SimpleNamespace(
        train_dataset=SimpleNamespace(
            set_epoch=lambda epoch: events.append(("epoch", epoch))
        ),
        train_loader=Loader(),
        epoch=0,
        max_steps=1,
        global_step=0,
        model=object(),
        rank_timer_every=0,
        batch_in_epoch=0,
        _set_dit_only_train_mode=lambda: None,
        _rank_timer_sync=lambda active: None,
        accelerator=SimpleNamespace(unwrap_model=lambda model: model),
        train_sampler=SimpleNamespace(clear_resume_batch_offset=lambda: None),
    )
    trainer._training_data_iterator = MethodType(
        methods["_training_data_iterator"], trainer
    )
    with pytest.raises(FinishedProbe):
        methods["train"](trainer)
    assert events == [("epoch", 0), ("iterator", 0), ("epoch", 1), ("iterator", 1)]
    plain = SimpleNamespace(train_dataset=object(), train_loader=[1, 2], epoch=4)
    assert list(methods["_training_data_iterator"](plain)) == [1, 2]


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="Native smoke schema needs CPU PyTorch",
)
def test_joint_smoke_uses_prediction40_and_unexpanded_history16_and_current_state_once():
    import torch

    scalar = torch.zeros(1)
    cfg = {
        "model": {"expected_action_horizon": 40, "expected_action_dim": 47},
        "data": {"train": {"past_action_size": 16}},
    }
    sample = {
        "video": scalar.expand(1, 3, 2, 224, 448),
        "action": scalar.expand(1, 40, 47),
        "past_action": scalar.expand(1, 16, 47),
        "proprio": scalar.expand(1, 40, 44),
        "text_hidden_states": scalar.expand(1, 128, 7680),
    }
    training.validate_smoke_sample(sample, cfg)
    with pytest.raises(RuntimeError, match="history"):
        training.validate_smoke_sample(
            dict(sample, past_action=scalar.expand(1, 40, 47)), cfg
        )
    future = torch.arange(40, dtype=torch.float32)[None, :, None].expand(1, 40, 44)
    with pytest.raises(RuntimeError, match="current state"):
        training.validate_smoke_sample(dict(sample, proprio=future), cfg)


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="Native trainer wrapper imports CPU PyTorch",
)
def test_training_wrapper_attaches_bound_augmentation_before_native_initialization(
    tmp_path, monkeypatch
):
    class StopAfterInitialization(Exception):
        pass

    class FakeTrainer:
        def __init__(self, **kwargs):
            raise StopAfterInitialization

        def train(self):
            pass

    cfg = with_profile(fixtures.fixture_config(tmp_path))
    monkeypatch.setitem(
        __import__("sys").modules,
        "imagewam.trainer",
        SimpleNamespace(Wan22Trainer=FakeTrainer),
    )
    monkeypatch.setattr(training, "initializer_audit", lambda model, cfg: {})
    monkeypatch.setattr(training, "_INSTALLED", False)
    training.install_training_contract(smoke=False)
    model = SimpleNamespace()
    with pytest.raises(StopAfterInitialization):
        FakeTrainer(model=model, cfg=cfg)
    assert model.training_augmentation_config == cfg["data"]["train"]["augmentation"]
    assert (
        model.training_augmentation_config is not cfg["data"]["train"]["augmentation"]
    )
