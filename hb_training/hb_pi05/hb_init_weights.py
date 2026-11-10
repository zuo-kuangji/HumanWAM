"""pi05_base (JAX) -> PyTorch -> 76-D G1 init checkpoint.

  convert : runs OpenPI's own converter examples/convert_jax_model_to_pytorch.py:main (pi05 path, float32, stock
            32-D config `pi05_droid_finetune` = Pi0Config(pi05=True, action_dim=32, action_horizon=16)). That converter
            calls PI0Pytorch.load_state_dict(..., strict=False) and ignores the result; we record it and FAIL if anything
            except the unused/tied `lm_head` weights was not loaded.
  init76  : builds PI0Pytorch for action_dim=76 in float32 (one checkpoint shared by all tasks: G1 uses 76 dims, H2 a18
            uses 75 + 1 zero pad dim; max_token_len does not affect parameters), copies every converted tensor whose name and
            shape match, keeps the fresh nn.Linear init for whole layers that mismatch (only action_in_proj and
            action_out_proj are allowed to), logs every fresh/unloaded key, FAILS on any other mismatch, and writes
            model.safetensors that train_pytorch.py loads strictly via TrainConfig.pytorch_weight_path.
  --lerobot: GCS (gs://openpi-assets) is ~100 KB/s from the DLC nodes, so the base can instead be HF
            lerobot/pi05_base @ 22c30e5 (public, fp32 PyTorch port of the same weights); its names are remapped to
            PI0Pytorch and the same strict checks apply.
Usage:  python hb_init_weights.py convert --jax-dir $HB_PI05_BASE_JAX --out $HB_PI05_BASE_PT
        python hb_init_weights.py init76  --base $HB_PI05_BASE_PT   --out $HB_PI05_INIT
        python hb_init_weights.py init76  --base $HB_G/models/lerobot-pi05_base --lerobot --out $HB_PI05_INIT
"""
import argparse
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import safetensors.torch
import torch

ALLOWED_FRESH_LAYERS = {"action_in_proj", "action_out_proj"}
ALLOWED_UNLOADED = re.compile(os.environ.get("HB_ALLOW_UNLOADED", r".*lm_head\.weight$"))


def cmd_convert(a):
    import openpi.models_pytorch.pi0_pytorch as p0

    jax_dir, out = Path(a.jax_dir).resolve(), Path(a.out).resolve()
    if "pi05" not in str(jax_dir):
        sys.exit(f"converter selects the pi05 (adaRMS) layout only if 'pi05' is in the checkpoint path: {jax_dir}")
    if not (jax_dir / "params").is_dir():
        sys.exit(f"{jax_dir}/params missing (download gs://openpi-assets/checkpoints/pi05_base/ there)")
    results = []
    orig = torch.nn.Module.load_state_dict

    def recording_load(self, state_dict, strict=True, assign=False):
        res = orig(self, state_dict, strict=strict, assign=assign)
        results.append(res)
        return res

    p0.PI0Pytorch.load_state_dict = recording_load
    path = Path(os.environ["HB_OPENPI"]) / "examples/convert_jax_model_to_pytorch.py"
    spec = importlib.util.spec_from_file_location("openpi_convert_jax_model_to_pytorch", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main(checkpoint_dir=str(jax_dir), config_name="pi05_droid_finetune", output_path=str(out), precision="float32")
    p0.PI0Pytorch.load_state_dict = orig
    assert len(results) == 1, results
    missing, unexpected = list(results[0].missing_keys), list(results[0].unexpected_keys)
    bad = [k for k in missing if not ALLOWED_UNLOADED.fullmatch(k)]
    report = {"jax_dir": str(jax_dir), "missing_keys": missing, "unexpected_keys": unexpected, "bad": bad}
    (out / "hb_convert_report.json").write_text(json.dumps(report, indent=2))
    print("converter load_state_dict: missing", missing, "unexpected", unexpected)
    if bad or unexpected:
        sys.exit(f"CONVERT_FAIL: unloaded {bad} unexpected {unexpected}")
    print("CONVERT_OK", out)


def remap_lerobot(src, target_keys):
    """lerobot/pi05_base (LeRobot PI05Policy, PyTorch port of OpenPI's PI0Pytorch) -> OpenPI PI0Pytorch names.
    Exact name, then without the policy's leading 'model.', then a unique suffix match. Normalizer buffers are dropped;
    anything else unmatched fails, so every tensor is either mapped 1:1 or reported."""
    tset = set(target_keys)
    out, dropped, unmatched, renamed = {}, [], [], {}
    for k, v in src.items():
        cands = [k] + ([k[len("model."):]] if k.startswith("model.") else [])
        hit = next((c for c in cands if c in tset), None)
        if hit is None:
            b = cands[-1]
            suf = [t for t in tset if t.endswith("." + b) or b.endswith("." + t)]
            hit = suf[0] if len(suf) == 1 else None
        if hit is None:
            (dropped if re.search(r"normaliz", k) else unmatched).append(k)
            continue
        if hit in out:
            unmatched.append(f"duplicate target {hit} <- {k}")
            continue
        out[hit] = v
        if hit != k:
            renamed[k] = hit
    return out, dropped, unmatched, renamed


def cmd_init76(a):
    import openpi.models_pytorch.pi0_pytorch as p0

    import hb_g1_config as C

    base, out = Path(a.base), Path(a.out)
    src = safetensors.torch.load_file(str(base / "model.safetensors"))
    cfg = C.model_config(dtype="float32", pytorch_compile_mode=None)
    torch.manual_seed(a.seed)
    model = p0.PI0Pytorch(cfg)
    target = model.state_dict()
    lerobot_info = None
    if a.lerobot:
        src = {k: v.float() for k, v in src.items()}
        src, dropped, unmatched, renamed = remap_lerobot(src, target.keys())
        print(f"lerobot remap: {len(src)} mapped, {len(renamed)} renamed, dropped {dropped}")
        if unmatched:
            sys.exit("INIT76_FAIL lerobot keys without an OpenPI match:\n" + "\n".join(unmatched[:200]))
        lerobot_info = {"dropped": dropped, "renamed_count": len(renamed),
                        "renamed_examples": dict(list(renamed.items())[:20])}
    # tied parameters share one tensor; save_model keeps only one of their names
    alias = {}
    for name, p in model.named_parameters(remove_duplicate=False):
        alias.setdefault(id(p), []).append(name)
    tied = {n: names for names in alias.values() if len(names) > 1 for n in names}

    fresh_layers, problems, unloaded = set(), [], []
    for k, v in target.items():
        layer = k.rsplit(".", 1)[0]
        if k in src:
            if tuple(src[k].shape) != tuple(v.shape):
                if layer in ALLOWED_FRESH_LAYERS:
                    fresh_layers.add(layer)
                else:
                    problems.append(f"shape {k}: base {tuple(src[k].shape)} vs new {tuple(v.shape)}")
        elif any(t in src for t in tied.get(k, [])):
            continue  # loaded through its tied alias
        elif ALLOWED_UNLOADED.fullmatch(k):
            unloaded.append(k)
        else:
            problems.append(f"missing in base: {k}")
    problems += [f"not in new model: {k}" for k in src if k not in target]
    if problems:
        sys.exit("INIT76_FAIL\n" + "\n".join(problems))
    load = {k: t for k, t in src.items() if k.rsplit(".", 1)[0] not in fresh_layers}
    res = model.load_state_dict(load, strict=False)
    fresh_keys = sorted(k for k in target if k.rsplit(".", 1)[0] in fresh_layers)
    allowed_missing = set(fresh_keys) | set(unloaded) | {k for k in tied if k not in src}
    assert not res.unexpected_keys and set(res.missing_keys) <= allowed_missing, res
    for k in fresh_keys:
        print(f"FRESH {k}: new {tuple(target[k].shape)} (base {tuple(src[k].shape) if k in src else None})")
    for k in unloaded:
        print(f"UNLOADED (kept init, unused) {k}")
    out.mkdir(parents=True, exist_ok=True)
    safetensors.torch.save_model(model, str(out / "model.safetensors"))
    report = {"base": str(base), "base_format": "lerobot" if a.lerobot else "openpi", "lerobot_remap": lerobot_info,
              "seed": a.seed, "model_config": {"pi05": cfg.pi05, "action_dim": cfg.action_dim,
              "action_horizon": cfg.action_horizon, "max_token_len": cfg.max_token_len}, "fresh_keys": fresh_keys,
              "unloaded_keys": unloaded, "loaded_tensors": len(load), "total_tensors": len(target)}
    (out / "hb_init_report.json").write_text(json.dumps(report, indent=2))
    print(f"INIT76_OK {out}: loaded {len(load)} tensors, fresh {fresh_keys}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert")
    c.add_argument("--jax-dir", required=True)
    c.add_argument("--out", required=True)
    i = sub.add_parser("init76")
    i.add_argument("--base", required=True)
    i.add_argument("--out", required=True)
    i.add_argument("--seed", type=int, default=0)
    i.add_argument("--lerobot", action="store_true",
                   help="--base is lerobot/pi05_base (HF, PyTorch); remap its names to OpenPI PI0Pytorch")
    a = ap.parse_args()
    {"convert": cmd_convert, "init76": cmd_init76}[a.cmd](a)


if __name__ == "__main__":
    main()
