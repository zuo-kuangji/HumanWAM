"""Assemble the diffusers Cosmos-Predict2.5-2B "base/post-trained" pipeline directory that DiT4DiT loads
(framework.cosmos25.base_model -> diffusers.Cosmos2_5_PredictBasePipeline.from_pretrained) from PUBLIC sources.

The official diffusers copy (nvidia/Cosmos-Predict2.5-2B, revision diffusers/base/post-trained) is gated. It was produced
by diffusers' own scripts/convert_cosmos_to_diffusers.py (--transformer_type Cosmos-2.5-Predict-Base-2B --save_pipeline),
whose inputs are:
  transformer  base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt   (gated repo; we take the byte-size
               downloaded from the official gated nvidia/Cosmos-Predict2.5-2B with the user's HF token by fetch_models.sh)
  text encoder nvidia/Cosmos-Reason1-7B (public; shard sizes identical to the official text_encoder/)
  tokenizer    Qwen/Qwen2.5-VL-7B-Instruct (public; the script's default)
  vae          Wan-AI/Wan2.1-T2V-1.3B-Diffusers subfolder vae (public; byte size identical to the official vae/)
  scheduler    UniPCMultistepScheduler(use_karras_sigmas, use_flow_sigmas, flow_prediction, sigma_max 200, sigma_min 0.01)
This file re-implements that script's Cosmos-2.5 branch (diffusers @ 3996788b, Apache-2.0) with one difference: the
checkpoint is read with torch.load(weights_only=True) (its pickle only references OrderedDict, _rebuild_tensor_v2 and
BFloat16/Byte/Long storages, so no code from the mirror is executed).

Usage: python hb_build_cosmos.py --transformer-ckpt X.pt --text-encoder DIR --tokenizer DIR --wan-root DIR --out DIR
Prints COSMOS_BUILD_OK. Licence: NVIDIA Open Model License ("Built on NVIDIA Cosmos") -> NOTICE file in the output.
"""
import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import torch

RENAME = {  # TRANSFORMER_KEYS_RENAME_DICT_COSMOS_2_0 (order matters)
    "t_embedder.1": "time_embed.t_embedder",
    "t_embedding_norm": "time_embed.norm",
    "blocks": "transformer_blocks",
    "adaln_modulation_self_attn.1": "norm1.linear_1",
    "adaln_modulation_self_attn.2": "norm1.linear_2",
    "adaln_modulation_cross_attn.1": "norm2.linear_1",
    "adaln_modulation_cross_attn.2": "norm2.linear_2",
    "adaln_modulation_mlp.1": "norm3.linear_1",
    "adaln_modulation_mlp.2": "norm3.linear_2",
    "self_attn": "attn1",
    "cross_attn": "attn2",
    "q_proj": "to_q",
    "k_proj": "to_k",
    "v_proj": "to_v",
    "output_proj": "to_out.0",
    "q_norm": "norm_q",
    "k_norm": "norm_k",
    "mlp.layer1": "ff.net.0.proj",
    "mlp.layer2": "ff.net.2",
    "x_embedder.proj.1": "patch_embed.proj",
    "final_layer.adaln_modulation.1": "norm_out.linear_1",
    "final_layer.adaln_modulation.2": "norm_out.linear_2",
    "final_layer.linear": "proj_out",
}
DROP = ("accum_video_sample_counter", "accum_image_sample_counter", "accum_iteration", "accum_train_in_hours",
        "pos_embedder.seq", "pos_embedder.dim_spatial_range", "pos_embedder.dim_temporal_range", "_extra_state")
CONFIG = {  # TRANSFORMER_CONFIGS["Cosmos-2.5-Predict-Base-2B"]
    "in_channels": 16 + 1, "out_channels": 16, "num_attention_heads": 16, "attention_head_dim": 128, "num_layers": 28,
    "mlp_ratio": 4.0, "text_embed_dim": 1024, "adaln_lora_dim": 256, "max_size": (128, 240, 240),
    "patch_size": (1, 2, 2), "rope_scale": (1.0, 3.0, 3.0), "concat_padding_mask": True, "extra_pos_embed_type": None,
    "use_crossattn_projection": True, "crossattn_proj_in_channels": 100352, "encoder_hidden_states_channels": 1024,
}


class _NoSafety:  # DiT4DiT also replaces the guardrail with a dummy (Cosmos25.py:_DefaultDummySafetyChecker)
    def __init__(self, *a, **k):
        pass

    def to(self, *a, **k):
        return self

    def check_text_safety(self, text):  # noqa: ARG002
        return True

    def check_video_safety(self, video):
        return video


def sha256(path, bs=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(bs):
            h.update(b)
    return h.hexdigest()


def convert_transformer(ckpt):
    from accelerate import init_empty_weights
    from diffusers import CosmosTransformer3DModel

    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    for k in ("model", "module", "state_dict"):
        if isinstance(sd, dict) and k in sd:
            sd = sd[k]
    new = {}
    for key, v in sd.items():
        nk = key.removeprefix("net.")
        for a, b in RENAME.items():
            nk = nk.replace(a, b)
        if any(d in nk for d in DROP):
            continue
        new[nk] = v
    with init_empty_weights():
        model = CosmosTransformer3DModel(**CONFIG)
    expected = set(model.state_dict())
    missing, unexpected = sorted(expected - set(new)), sorted(set(new) - expected)
    if missing or unexpected:
        sys.exit(f"transformer key mismatch: missing {missing[:10]} ({len(missing)}), unexpected {unexpected[:10]} "
                 f"({len(unexpected)})")
    model.load_state_dict(new, strict=True, assign=True)
    return model.to(dtype=torch.bfloat16), {k: new[k] for k in list(new)[:: max(1, len(new) // 6)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transformer-ckpt", required=True)
    ap.add_argument("--text-encoder", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--wan-root", required=True, help="Wan-AI/Wan2.1-T2V-1.3B-Diffusers snapshot (uses subfolder vae)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--provenance", default="{}", help="JSON with source repos / revisions / hashes")
    a = ap.parse_args()

    import diffusers
    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler
    from diffusers.pipelines.cosmos.pipeline_cosmos2_5_predict import Cosmos2_5_PredictBasePipeline
    from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

    out = Path(a.out)
    tmp = out.parent / f".{out.name}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    t0 = time.time()
    transformer, probe = convert_transformer(a.transformer_ckpt)
    print(f"transformer converted ({sum(p.numel() for p in transformer.parameters()) / 1e9:.3f} B params)", flush=True)
    vae = AutoencoderKLWan.from_pretrained(a.wan_root, subfolder="vae", torch_dtype=torch.float32)
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(a.text_encoder, torch_dtype="auto", device_map="cpu")
    tokenizer = AutoTokenizer.from_pretrained(a.tokenizer)
    scheduler = UniPCMultistepScheduler(use_karras_sigmas=True, use_flow_sigmas=True, prediction_type="flow_prediction",
                                        sigma_max=200.0, sigma_min=0.01)
    pipe = Cosmos2_5_PredictBasePipeline(text_encoder=text_encoder, tokenizer=tokenizer, transformer=transformer, vae=vae,
                                         scheduler=scheduler, safety_checker=lambda *args, **kwargs: None)
    pipe.save_pretrained(str(tmp), safe_serialization=True, max_shard_size="5GB")
    del pipe, text_encoder, vae

    # reload exactly as DiT4DiT does and spot-check tensors
    re = Cosmos2_5_PredictBasePipeline.from_pretrained(str(tmp), torch_dtype=torch.bfloat16, local_files_only=True,
                                                       safety_checker=_NoSafety())
    tsd = re.transformer.state_dict()
    for k, v in probe.items():
        if not torch.equal(tsd[k].float(), v.to(torch.bfloat16).float()):
            sys.exit(f"reload mismatch for transformer.{k}")
    n_blocks = len(re.transformer.transformer_blocks)
    assert n_blocks == 28 and re.transformer.config.in_channels == 17, (n_blocks, re.transformer.config.in_channels)
    assert list(re.vae.temperal_downsample) == [False, True, True], re.vae.temperal_downsample
    prov = json.loads(a.provenance)
    prov.update(built=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), diffusers=diffusers.__version__,
                torch=torch.__version__, seconds=round(time.time() - t0),
                equivalent_to="nvidia/Cosmos-Predict2.5-2B @ diffusers/base/post-trained (same recipe, public inputs)",
                transformer_blocks=n_blocks, text_encoder_class=type(re.text_encoder).__name__,
                tokenizer_class=type(re.tokenizer).__name__)
    del re
    (tmp / "HB_PROVENANCE.json").write_text(json.dumps(prov, indent=2))
    (tmp / "NOTICE").write_text("Licensed by NVIDIA Corporation under the NVIDIA Open Model License\nBuilt on NVIDIA Cosmos\n"
                                "Text encoder: nvidia/Cosmos-Reason1-7B; tokenizer: Qwen/Qwen2.5-VL-7B-Instruct (Apache-2.0); "
                                "VAE: Wan-AI/Wan2.1-T2V-1.3B-Diffusers (Apache-2.0).\n")
    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)
    print(f"COSMOS_BUILD_OK {out} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
