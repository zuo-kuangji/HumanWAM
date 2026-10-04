"""Offline right-arm prediction example. Does not command a physical robot."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from PIL import Image
import torch
from torchvision.transforms import functional as TF
from hydra.utils import instantiate
from omegaconf import OmegaConf
from imagewam.datasets.lerobot.utils.normalizer import LinearNormalizer, load_dataset_stats_from_json


def pack_images(head, wrist):
    images = [TF.resize(TF.to_tensor(Image.open(p).convert("RGB")), [224, 224], antialias=True)
              for p in [head, wrist]]
    return torch.cat(images, dim=-1).unsqueeze(0) * 2 - 1


def normalize_inputs(normalizer, past, proprio):
    past, proprio = torch.as_tensor(past, dtype=torch.float32), torch.as_tensor(proprio, dtype=torch.float32)
    if past.shape != (16, 7) or proprio.shape != (7,):
        raise ValueError("Expected raw past_action [16,7] and current proprio [7]")
    if not torch.isfinite(past).all() or not torch.isfinite(proprio).all():
        raise ValueError("Inputs must be finite")
    sample = normalizer.forward({
        "action": {"right_arm": past[:, :6], "right_gripper": past[:, 6:]},
        "state": {"right_arm": proprio[:6], "right_gripper": proprio[6:]}})
    return (torch.cat([sample["action"][k] for k in ["right_arm", "right_gripper"]], -1).unsqueeze(0),
            torch.cat([sample["state"][k] for k in ["right_arm", "right_gripper"]], -1).unsqueeze(0))


def release_settings(release, cfg, instruction=None):
    manifest = json.loads((release / "manifest.json").read_text())
    step = int(manifest["checkpoint_step"])
    if not manifest.get("inference_only") or manifest.get("optimizer_included"):
        raise ValueError("Expected an inference-only release manifest")
    dataset = manifest["dataset"]
    supported = {"at237299966/r1lite_place_multi_cups": 15280,
                 "at237299966/r1lite_insert_flower": 11440}
    if supported.get(dataset) != step:
        raise ValueError("Unsupported dataset/final checkpoint combination")
    filename = f"step_{step:06d}.pt"
    if filename not in {item["path"] for item in manifest["files"]}:
        raise ValueError("Final checkpoint absent from release manifest")
    default = cfg.data.train.get("override_instruction") or "place multi cups"
    if dataset.endswith("r1lite_insert_flower") and not cfg.data.train.get("override_instruction"):
        raise ValueError("Flower release must contain its training instruction override")
    return release / filename, step, default if instruction is None else instruction


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--release", type=Path, required=True)
    p.add_argument("--head", type=Path, required=True)
    p.add_argument("--right-wrist", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True, help="NPZ: raw past_action [16,7], proprio [7]")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--flux2-src", required=True)
    p.add_argument("--flux2-weights", required=True)
    p.add_argument("--ae-weights", required=True)
    p.add_argument("--qwen", default="Qwen/Qwen3-4B")
    p.add_argument("--text-cache", type=Path, help="Optional matching prompt's .qwen3_flux2_len128.pt")
    p.add_argument("--instruction", help="Defaults to the selected release's training instruction")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=None, help="Default: fresh noise on every call")
    args = p.parse_args()
    cfg = OmegaConf.load(args.release / "config.yaml")
    checkpoint, expected_step, instruction = release_settings(args.release, cfg, args.instruction)
    prompt = "A video recorded from a robot's point of view executing the following instruction: " + instruction
    if args.text_cache:
        expected = hashlib.sha256(prompt.encode("utf-8")).hexdigest() + ".qwen3_flux2_len128.pt"
        if args.text_cache.name != expected:
            p.error("Text cache filename must match the exact instruction's SHA256 and context length 128")
    if args.output.exists():
        p.error("Refusing to overwrite prediction output")
    model_cfg = cfg.model
    model_cfg.flux2_src_path = args.flux2_src
    model_cfg.flux2_model_path = args.flux2_weights
    model_cfg.ae_model_path = args.ae_weights
    model_cfg.action_dit_pretrained_path = None
    model_cfg.qwen3_model_spec = args.qwen
    model_cfg.load_text_encoder = args.text_cache is None
    model_cfg.mot_checkpoint_mixed_attn = False
    model = instantiate(model_cfg, model_dtype=torch.bfloat16, device=args.device).eval()
    # Fail closed on a mismatched full checkpoint; native load_checkpoint is permissive.
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if payload.get("step") != expected_step or "optimizer" in payload:
        raise ValueError("Expected the selected release's final inference checkpoint")
    model.mot.load_state_dict(payload["mot"], strict=True)
    model.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
    del payload
    stats = load_dataset_stats_from_json(str(args.release / "dataset_stats.json"))
    normalizer = LinearNormalizer(cfg.data.train.shape_meta, False, "q01/q99", None, stats)
    with np.load(args.inputs, allow_pickle=False) as inputs:
        past, proprio = normalize_inputs(normalizer, inputs["past_action"], inputs["proprio"])
    text = {}
    if args.text_cache:
        cached = torch.load(args.text_cache, map_location="cpu", weights_only=True)
        text = {"context": cached["text_hidden_states"].unsqueeze(0),
                "context_mask": cached["text_attention_mask"].unsqueeze(0)}
    with torch.inference_mode():
        result = model.infer_action_flux2(prompt=prompt, input_image=pack_images(args.head, args.right_wrist),
            action_horizon=16, proprio=proprio, num_inference_steps=1, image_num_inference_steps=1,
            action_init=past, action_init_noise_strength=.5, action_init_noise_type="additive",
            action_init_noise_dim_mask=[True] * 7, clamp_action_init_after_noise=False,
            seed=args.seed, **text)
    normalized = result["action"].cpu()
    raw = torch.cat([normalizer.normalizers["action"][key].backward(part)
                     for key, part in [("right_arm", normalized[:, :6]), ("right_gripper", normalized[:, 6:])]], -1)
    if not torch.isfinite(raw).all():
        raise ValueError("Non-finite predicted actions")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, action=raw.numpy(), normalized_action=normalized.numpy())
    print(f"Saved {tuple(raw.shape)} absolute right-arm targets to {args.output}; not sent to hardware.")


if __name__ == "__main__":
    main()
