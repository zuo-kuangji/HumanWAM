#!/usr/bin/env python3
"""Launch the audited native whole-body recipe, optionally as a five-update smoke.

Run with Accelerate/DeepSpeed ZeRO-1, not plain Python. The smoke starts from
the base and saves no model/optimizer checkpoint. Production requires its
matching smoke receipts and starts independently from the base again.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

STATS_SCHEMA = "wholebody-state44-action76-sameframe-all-episodes-v1"
NORMALIZATION = "fit_wholebody_motion64_body29_gravity3_fixed_inspire_hands"
MODEL_TARGET = "imagewam.wholebody_delta.create_imagewam_flux2_wholebody_delta"
DATASET_TARGET = "imagewam.datasets.wholebody.WholeBodyLeRobotDataset"
_INSTALLED = False


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def expected_microbatches(loader_length, accumulation, optimizer_updates):
    """Count consumed microbatches, including partial accumulation at epoch end."""
    require(loader_length > 0 and accumulation > 0 and optimizer_updates > 0,
            "Loader length, accumulation and optimizer updates must be positive")
    updates_per_epoch = math.ceil(loader_length / accumulation)
    complete_epochs, remaining_updates = divmod(optimizer_updates, updates_per_epoch)
    return complete_epochs * loader_length + min(remaining_updates * accumulation, loader_length)


def validate_config(cfg, *, smoke):
    """Check the resolved, fresh-run method before constructing a CUDA model."""
    c, model, data = cfg["wholebody_contract"], cfg["model"], cfg["data"]
    require(cfg["resume"] is None and not model.get("parent_checkpoint")
            and model.get("additional_training_epochs") is None, "This entry is fresh-base training, not task-checkpoint continuation")
    require(model["_target_"] == MODEL_TARGET and data["train"]["_target_"] == DATASET_TARGET,
            "Use the dedicated whole-body model and dataset")
    require(c["fresh_base"] is True and c["zero_stage"] == 1 and int(c["world_size"]) > 0,
            "Fresh-base ZeRO-1 distribution contract missing")
    require(cfg["learning_rate"] == 1e-4 and cfg["weight_decay"] == 0.01
            and cfg["mixed_precision"] == "bf16" and cfg["seed"] == 42,
            "The published fresh full-finetuning recipe uses LR1e-4, WD0.01, BF16 and seed42")
    require(data["val"] is None and cfg["eval_every"] == 0
            and data["train"]["train_split_mode"] == "all_episodes", "This recipe has no held-out validation split")
    require(not model["flux2_lora_config"]["enabled"] and not model["load_text_encoder"]
            and not model["pack_proprio_after_text"], "Full DiT finetuning uses cached pure text, without state tokens")
    require(model["action_state_dim"] == model["proprio_dim"] == 44
            and model["expected_action_dim"] == model["action_dit_config"]["action_dim"] == 76
            and model["expected_action_horizon"] == 40, "Expected current state44, action76 and horizon40")
    ds = data["train"]
    require(ds["action_horizon"] == ds["past_action_size"] == 40
            and ds["qwen_context_len"] == 128 and ds["qwen_text_dim"] == 7680,
            "Dataset history/horizon/text dimensions changed")
    context = model["flux2_action_image_context"]
    require(context["mode"] == "noise_idm" and context["include_future_tokens"] is True
            and context["knowledge_insulation"] is True and context["reuse_image_fm_workspace"] is False,
            "Action conditioning must use insulated visual workspaces")
    loss, image = model["loss"], model["loss"]["image_i2i"]
    action = loss["action_a2a"]
    require(loss["lambda_video"] == 0.5 and loss["lambda_action"] == 1.0
            and action["enabled"] and action["fm_weight"] == 1.0 and action["ic_weight"] == 0.5
            and action["ic_steps"] == 1 and action["ic_loss"] == "l1"
            and action["source_noise_std"] == 0.5 and list(action["source_noise_dim_mask"]) == [True] * 76
            and not action["aura"]["enabled"], "Action source/FM/IC or all-76 noise mask changed")
    require(image["enabled"] and image["source_mode"] == "gaussian" and image["variance_normalize"]
            and math.isclose(image["current_residual_scale"], 1 / math.sqrt(3), rel_tol=1e-12)
            and image["ic_weight"] == 1.0 and image["ic_steps"] == 1 and image["ic_loss"] == "l1",
            "Visual source must be 0.5current + sqrt0.75noise with native visual FM/IC")
    measurement = c["measured_loader"]
    require(measurement["world_size"] == c["world_size"] and measurement["microbatch"] == cfg["batch_size"]
            and measurement["gradient_accumulation"] == cfg["gradient_accumulation_steps"], "Loader distribution changed after config generation")
    updates = math.ceil(measurement["per_rank_microbatches"] / cfg["gradient_accumulation_steps"])
    total = updates * cfg["num_epochs"]
    require(total >= 5 and c["updates_per_epoch"] == updates and cfg["lr_scheduler_total_steps"] == total
            and cfg["warmup_steps"] == max(1, int(total * .05)), "Rebuild config after changing dataset/epochs/batch/world")
    require(cfg["keep_latest_weights"] == 4 and cfg["keep_latest_state_only"] is True
            and not cfg["save_steps"], "Retain four weights and one training state")
    require(cfg["max_steps"] == (5 if smoke else total)
            and cfg["save_every"] == (0 if smoke else updates)
            and cfg["save_at_end"] is (not smoke), "Smoke/production stopping or saving configuration mismatch")
    expected_output = c["smoke_output_dir"] if smoke else c["production_output_dir"]
    require(Path(cfg["output_dir"]).resolve() == Path(expected_output).resolve(), "Wrong smoke/production output directory")
    stats_path = Path(ds["stats_path"])
    require(sha256(stats_path) == c["statistics_sha256"], "Statistics changed after config generation")
    stats = json.loads(stats_path.read_text())
    require(stats.get("schema") == STATS_SCHEMA and stats.get("normalization_strategy") == NORMALIZATION
            and stats.get("train_frames") == measurement["dataset_frames"]
            and stats.get("action_dim") == 76 and stats.get("proprio_dim") == 44,
            "Statistics schema/dimensions/count do not match this dataset")
    require(stats.get("train_split_mode") == "all_episodes" and stats.get("heldout_episode_ids") == []
            and stats.get("hand_target_time_offset") == 0, "Statistics contain a held-out split or shifted hand labels")
    require(sha256(model["action_dit_pretrained_path"]) == model["action_initializer_sha256"], "Action initializer changed")
    return {"updates_per_epoch": updates, "total_updates": total, "warmup_updates": cfg["warmup_steps"]}


def require_completed_smoke(cfg):
    """Production is allowed only after all ranks completed the same five-update smoke."""
    c = cfg["wholebody_contract"]
    root = Path(c["smoke_output_dir"])
    for rank in range(int(c["world_size"])):
        report = json.loads((root / f"smoke_rank_{rank}.json").read_text())
        require(report.get("status") == "FIVE_UPDATES_VERIFIED" and report.get("actual_optimizer_updates") == 5
                and report.get("skipped_updates") == 0 and report.get("world_size") == c["world_size"]
                and report.get("microbatch") == cfg["batch_size"]
                and report.get("accumulation") == cfg["gradient_accumulation_steps"]
                and report.get("statistics_sha256") == c["statistics_sha256"]
                and report.get("action_initializer_sha256") == cfg["model"]["action_initializer_sha256"]
                and report.get("schedule_total_updates") == cfg["lr_scheduler_total_steps"],
                f"Rank {rank} has no matching completed five-update smoke; run --smoke first")


def claim_output(output, config_path):
    """One rank creates a brand-new run; other ranks join that exact launch claim."""
    output, config_path = Path(output), Path(config_path).resolve(strict=True)
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    claim = output / "launch_claim.json"
    expected = {"config_sha256": sha256(config_path), "config_path": str(config_path)}
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
        temporary = output / ("launch_claim.tmp." + str(os.getpid()) + ".json")
        write_json_new(temporary, expected)
        temporary.replace(claim)
    else:
        deadline = time.monotonic() + 60
        while not claim.is_file() and time.monotonic() < deadline:
            time.sleep(.1)
        require(claim.is_file(), "Rank0 did not create the fresh-run launch claim")
        require(json.loads(claim.read_text()) == expected, "Another launch owns this output directory")


def initializer_audit(model, cfg):
    audit = model.compatible_action_initialization_audit
    require(audit.get("status") == "shared_layers_loaded_exactly_new_projections_recorded"
            and audit.get("source_format") == "flux2_shared_partial_initializer_v1"
            and audit.get("source_action_dim") is None and audit.get("retained_tensor_count") == 114
            and audit.get("target_action_dim") == 76 and audit.get("current_state_dim") == 44,
            "Only the base-derived 114-key shared initializer is supported by this fresh recipe")
    require(Path(audit["source"]).resolve() == Path(cfg.model.action_dit_pretrained_path).resolve()
            and audit.get("source_sha256") == cfg.model.action_initializer_sha256, "Factory loaded a different initializer")
    expected = {"action_encoder.weight", "action_encoder.bias", "head.linear.weight",
                "state_encoder.weight", "state_encoder.bias", "head.adaLN_modulation.1.weight"}
    require(set(audit.get("newly_initialized", ())) == expected, "New input/output/state projections were not recorded explicitly")
    require(not audit.get("optimizer_loaded") and not audit.get("parent_checkpoint"), "Task checkpoint/optimizer inheritance detected")
    return audit


def install_training_contract(*, smoke):
    global _INSTALLED
    require(not _INSTALLED, "Training contract already installed")
    _INSTALLED = True
    import torch
    from imagewam.trainer import Wan22Trainer
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    original_init, original_train = Wan22Trainer.__init__, Wan22Trainer.train

    def checked_init(self, *args, **kwargs):
        cfg, before = kwargs.get("cfg"), kwargs.get("model")
        require(cfg is not None and before is not None, "Native trainer requires explicit model/config")
        schedule = validate_config(cfg, smoke=smoke)
        initializer_audit(before, cfg)
        original_init(self, *args, **kwargs)
        model = self.accelerator.unwrap_model(self.model)
        c = self.cfg.wholebody_contract
        protocol = model.validate_state_only_route()
        audit = initializer_audit(model, self.cfg)
        require(self.accelerator.num_processes == int(c.world_size)
                and self.batch_size == int(cfg.batch_size)
                and self.gradient_accumulation_steps == int(cfg.gradient_accumulation_steps), "Actual distributed batch differs from the measured recipe")
        plugin = self.accelerator.state.deepspeed_plugin
        require(plugin is not None and int(plugin.deepspeed_config["zero_optimization"]["stage"]) == 1, "Launch with DeepSpeed ZeRO-1")
        require(self.mixed_precision == "bf16" and self.resume is None and not self._weights_resume_loaded_before_prepare
                and self.global_step == self.epoch == self.batch_in_epoch == 0, "Fresh BF16 run must start at optimizer step/epoch0")
        actual = len(self.train_loader)
        require(actual == c.measured_loader.per_rank_microbatches
                and len(self.train_dataset) == c.measured_loader.dataset_frames, "Native runtime loader differs from CPU measurement")
        require(math.ceil(actual / self.gradient_accumulation_steps) == schedule["updates_per_epoch"], "Updates per epoch changed")
        native = getattr(self.scheduler, "scheduler", self.scheduler)
        require(isinstance(native, SequentialLR) and len(native._schedulers) == 2, "Expected native linear warmup + cosine scheduler")
        warmup, cosine = native._schedulers
        require(isinstance(warmup, LinearLR) and isinstance(cosine, CosineAnnealingLR)
                and warmup.total_iters == schedule["warmup_updates"]
                and math.isclose(warmup.start_factor, 1 / schedule["warmup_updates"], rel_tol=1e-12)
                and warmup.end_factor == 1 and cosine.T_max == schedule["total_updates"] - schedule["warmup_updates"]
                and math.isclose(cosine.eta_min, 1e-6, rel_tol=1e-12), "Native warmup/cosine schedule differs")
        require(math.isclose(float(self.optimizer.param_groups[0]["lr"]), 1e-4 / schedule["warmup_updates"], rel_tol=1e-7, abs_tol=1e-14), "Unexpected first warmup LR")
        require(model.proprio_encoder is None and model.proprio_dim is None and not hasattr(model, "pose_head"), "State/C-Pred entered the visual branch")
        require(protocol["visual_state_route"] == protocol["text_state_route"] == "none"
                and protocol["action_to_visual_gradients"] is False and protocol["knowledge_insulation"] is True
                and protocol["action_dim"] == 76 and protocol["current_state_dim"] == 44
                and protocol["action_horizon"] == 40, "Current-state/stop-gradient protocol changed")
        require(model.training_objective_count() == 3 and model.text_encoder is None
                and all(not p.requires_grad for p in model.vae.parameters()), "Use frozen VAE, cached Qwen and the three native objective events")
        require(not any(isinstance(m, torch.nn.Dropout) and m.p > 0 for m in model.modules()), "Condition/model dropout is not part of this recipe")
        engine = self.accelerator.deepspeed_engine_wrapped.engine
        require(engine.global_steps == engine.skipped_steps == 0, "Optimizer counters were inherited")
        optimized = [p for group in engine.optimizer.bit16_groups for p in group]
        ids = [id(p) for p in optimized]
        require(len(ids) == len(set(ids)) and set(ids) == {id(p) for p in model.parameters() if p.requires_grad}, "Trainable parameter optimizer coverage is not exact")
        require(all(p.requires_grad for p in model.action_expert.state_encoder.parameters()), "Action-side state encoder is frozen")
        write_json_new(Path(self.output_dir) / f"native_contract_rank_{self.accelerator.process_index}.json",
            {"status": "ACTUAL_NATIVE_TRAINER_CONTRACT_PASS", "rank": self.accelerator.process_index,
             "mode": "five_update_smoke" if smoke else "fresh_production", "world_size": self.accelerator.num_processes,
             "dataset_frames": len(self.train_dataset), "actual_loader_microbatches_per_epoch": actual,
             "microbatch": self.batch_size, "accumulation": self.gradient_accumulation_steps,
             "nominal_global_batch": self.batch_size * self.gradient_accumulation_steps * self.accelerator.num_processes,
             **schedule, "stopping_updates": self.max_steps, "fresh_initialization": True,
             "parent_weights_loaded": False, "parent_optimizer_loaded": False, "smoke_weights_loaded": False,
             "state_encoder_in_optimizer": True, "initial_lr": float(self.optimizer.param_groups[0]["lr"]),
             "peak_lr": self.learning_rate, "cosine_min_lr": cosine.eta_min,
             "protocol": protocol, "initialization_audit": audit})

    def checked_train(self):
        original_train(self)
        engine = self.accelerator.deepspeed_engine_wrapped.engine
        require(self.global_step == self.max_steps == engine.global_steps and engine.skipped_steps == 0, "Native training did not finish its exact optimizer-update count")
        self.accelerator.wait_for_everyone()
        weights = sorted(Path(self.weights_dir).glob("*.pt"))
        states = sorted(p for p in Path(self.state_dir).iterdir() if p.is_dir())
        require((not weights and not list(Path(self.state_dir).iterdir())) if smoke else
                (0 < len(weights) <= 4 and len(states) == 1), "Checkpoint retention/save contract failed")
        write_json_new(Path(self.output_dir) / f"training_finished_rank_{self.accelerator.process_index}.json",
            {"status": "FIVE_UPDATES_NATIVE_TRAINING_FINISHED" if smoke else "EPOCH_TRAINING_FINISHED",
             "optimizer_updates": self.global_step, "skipped_updates": engine.skipped_steps,
             "training_epochs": self.num_epochs, "fresh_base": True, "smoke_weights_loaded": False,
             "weight_files": [p.name for p in weights], "training_state_directories": [p.name for p in states],
             "task_success_evaluated": False})

    Wan22Trainer.__init__, Wan22Trainer.train = checked_init, checked_train


def install_smoke_hooks():
    """Observe actual native losses/backwards and FP32 optimizer updates on every rank."""
    import torch
    import torch.distributed as dist
    from torch.utils.data._utils.collate import default_collate
    from imagewam.models.backbones.imagewam import ImageWAM
    from imagewam.wholebody_delta import StateOnlyDeltaImageWAM, _CURRENT_ACTION_STATE
    from imagewam.trainer import Wan22Trainer
    evidence = {"objective_events": [], "workspace_checks": 0, "action_source_checks": 0, "image_source_checks": 0}
    phase = {"name": None, "visual": 0., "action": 0., "state": 0.}

    def graph_free(value):
        if torch.is_tensor(value):
            require(not value.requires_grad and value.grad_fn is None, "Action visual workspace retains a backward graph")
        elif isinstance(value, dict):
            for item in value.values():
                graph_free(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                graph_free(item)

    original_pack = ImageWAM._pack_flux2_action_workspace
    def checked_pack(self, *args, **kwargs):
        value = original_pack(self, *args, **kwargs)
        graph_free(value)
        evidence["workspace_checks"] += 1
        return value
    ImageWAM._pack_flux2_action_workspace = checked_pack

    original_action = ImageWAM._apply_action_a2a_source_noise
    def checked_action(self, source, *args, **kwargs):
        value = original_action(self, source, *args, **kwargs)
        require(tuple(source.shape[1:]) == (40, 76) and tuple(self.action_a2a_source_noise_dim_mask) == (True,) * 76,
                "Wrong action-history shape/noise mask")
        residual = value.float() - source.float()
        require(bool(torch.isfinite(value).all()) and .35 < residual.std(unbiased=False).item() < .65
                and .30 < residual[..., 64:].std(unbiased=False).item() < .70, "Expected0.5 Gaussian noise on all action dimensions, including hands")
        evidence["action_source_checks"] += 1
        return value
    ImageWAM._apply_action_a2a_source_noise = checked_action

    original_image = ImageWAM._apply_image_i2i_source_noise
    def checked_image(self, source, *args, **kwargs):
        value = original_image(self, source, *args, **kwargs)
        scale = 1 / math.sqrt(3)
        residual = value.float() * math.sqrt(1 + scale**2) - scale * source.float()
        require(bool(torch.isfinite(value).all()) and .85 < residual.std(unbiased=False).item() < 1.15, "Visual source formula/noise differs")
        evidence["image_source_checks"] += 1
        return value
    ImageWAM._apply_image_i2i_source_noise = checked_image

    original_iter = StateOnlyDeltaImageWAM.iter_training_losses
    def checked_iter(self, *args, **kwargs):
        for loss, metrics in original_iter(self, *args, **kwargs):
            require(bool(torch.isfinite(loss).all()) and all(math.isfinite(float(v)) for v in metrics.values()), "Nonfinite native smoke loss")
            require(_CURRENT_ACTION_STATE.get() is None, "Action-state context leaked across objective yield/backward")
            name = "image_ic" if "loss_image_ic" in metrics else ("action_ic" if "loss_ic" in metrics and "loss_video" not in metrics else "main_fm")
            phase.update(name=name, visual=0., action=0., state=0.)
            yield loss, metrics
            event = dict(phase, metrics={k: float(v) for k, v in metrics.items()})
            if name == "image_ic":
                require(event["visual"] > 0 and event["action"] == event["state"] == 0, "Visual IC gradient route failed")
            elif name == "action_ic":
                require(event["visual"] == 0 and event["action"] > 0 and event["state"] > 0, "Action IC stop-gradient/current-state route failed")
            else:
                require(event["visual"] > 0 and event["action"] > 0 and event["state"] > 0, "Main visual/action FM gradients missing")
            evidence["objective_events"].append(event)
            phase["name"] = None
    StateOnlyDeltaImageWAM.iter_training_losses = checked_iter

    original_init = Wan22Trainer.__init__
    def checked_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        model = self.accelerator.unwrap_model(self.model)
        sample = default_collate([self.train_dataset[0]])
        require(tuple(sample["video"].shape) == (1, 3, 2, 224, 448)
                and tuple(sample["action"].shape) == tuple(sample["past_action"].shape) == (1, 40, 76)
                and tuple(sample["proprio"].shape) == (1, 40, 44)
                and tuple(sample["text_hidden_states"].shape) == (1, 128, 7680), "Actual sample schema differs")
        require(torch.equal(sample["proprio"], sample["proprio"][:, :1].expand_as(sample["proprio"])), "Dataset supplied future state instead of repeated current state")
        with torch.no_grad(), self.accelerator.autocast():
            first = model.build_inputs_flux2(sample)
            second = model.build_inputs_flux2(dict(sample, proprio=sample["proprio"] + .25))
            for key in ("text_hidden_states", "text_attention_mask", "ref_image_latents", "target_latent", "ref_img_ids", "target_img_ids"):
                require(torch.equal(first[key], second[key]), "State leaked into visual/text input: " + key)
            require(first["text_hidden_states"].shape[1] == 128, "State text token detected")
        del first, second, sample
        evidence.update(state_perturbation_visual_invariance=True, current_only_state_time_axis=True,
                        text_tokens_without_state=128, no_heldout_dataset=True)
        for label, module in (("visual", model.video_expert), ("action", model.action_expert), ("state", model.action_expert.state_encoder)):
            candidates = [(n, p) for n, p in module.named_parameters() if p.requires_grad]
            require(bool(candidates), "No trainable parameters in " + label)
            for name, param in (candidates if label == "state" else candidates[:4] + candidates[-4:]):
                def hook(grad, group=label):
                    require(phase["name"] is not None and bool(torch.isfinite(grad).all()), "Nonfinite or out-of-phase native gradient")
                    phase[group] += float(grad.detach().float().norm())
                    return grad
                param.register_hook(hook)
        engine = self.accelerator.deepspeed_engine_wrapped.engine
        evidence["master_probe_initial"] = [p.detach().flatten()[:65536].cpu().clone() for p in engine.optimizer.single_partition_of_fp32_groups]
        torch.cuda.reset_peak_memory_stats()
    Wan22Trainer.__init__ = checked_init

    original_train = Wan22Trainer.train
    def checked_train(self):
        original_train(self)
        engine = self.accelerator.deepspeed_engine_wrapped.engine
        require(self.global_step == engine.global_steps == 5 and engine.skipped_steps == 0, "Smoke needs exactly five real optimizer updates")
        micros = expected_microbatches(len(self.train_loader), self.gradient_accumulation_steps, 5)
        require(len(evidence["objective_events"]) == micros * 3, "Smoke native objective count differs")
        for name in ("image_ic", "main_fm", "action_ic"):
            require(sum(v["name"] == name for v in evidence["objective_events"]) == micros, "Missing native objective: " + name)
        require(evidence["workspace_checks"] >= micros and evidence["action_source_checks"] >= 2 * micros
                and evidence["image_source_checks"] >= 2 * micros, "Smoke did not observe actual source/workspace construction")
        old = evidence.pop("master_probe_initial")
        change = sum(float((p.detach().flatten()[:65536].cpu() - value).abs().sum())
                     for p, value in zip(engine.optimizer.single_partition_of_fp32_groups, old))
        require(change > 0 and math.isfinite(change), "No actual finite FP32 optimizer update observed")
        module = self.accelerator.unwrap_model(self.model).action_expert.state_encoder
        digest = hashlib.sha256()
        for name, value in module.state_dict().items():
            digest.update(name.encode())
            digest.update(value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
        hashes = [None] * self.accelerator.num_processes
        dist.all_gather_object(hashes, digest.hexdigest())
        require(len(set(hashes)) == 1, "Action-side state encoder differs across ranks")
        evidence.update(status="FIVE_UPDATES_VERIFIED", rank=self.accelerator.process_index,
            actual_optimizer_updates=engine.global_steps, skipped_updates=engine.skipped_steps,
            world_size=self.accelerator.num_processes, microbatch=self.batch_size, accumulation=self.gradient_accumulation_steps,
            consumed_microbatches=micros,
            schedule_total_updates=self.cfg.lr_scheduler_total_steps,
            statistics_sha256=self.cfg.wholebody_contract.statistics_sha256,
            action_initializer_sha256=self.cfg.model.action_initializer_sha256,
            fp32_master_probe_absolute_change=change, state_encoder_hash=digest.hexdigest(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            no_weight_or_optimizer_checkpoints_saved=True, production_must_reload_fresh_base=True,
            action_to_visual_workspace_graph_free=True, action_ic_stop_gradient_verified=True,
            action_fm_verification_scope="All native action visual workspaces detached; main FM gradient event combines visual and action losses")
        write_json_new(Path(self.output_dir) / f"smoke_rank_{self.accelerator.process_index}.json", evidence)
    Wan22Trainer.train = checked_train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    from omegaconf import OmegaConf
    from imagewam.runtime import run_training
    config_path = args.config.resolve(strict=True)
    cfg = OmegaConf.load(config_path)
    validate_config(cfg, smoke=args.smoke)
    receipt = json.loads((config_path.parent / "build_receipt.json").read_text())
    require(receipt["config_sha256"].get(config_path.name) == sha256(config_path), "Resolved config changed; rebuild into a new run directory")
    if not args.smoke:
        require_completed_smoke(cfg)
    claim_output(cfg.output_dir, config_path)
    install_training_contract(smoke=args.smoke)
    if args.smoke:
        install_smoke_hooks()
    run_training(cfg)


if __name__ == "__main__":
    main()
