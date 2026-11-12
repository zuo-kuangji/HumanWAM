"""FLUX.2 Qwen3 instruction cache for Delta-flow tasks that have no previous experiment to copy it from.

Same computation as HumanWAM scripts/flux2/precompute_flux2_qwen3_embeds.py (chat template, enable_thinking False,
padding to max_length 128, hidden states of flux2.text_encoder.OUTPUT_LAYERS_QWEN3 concatenated, stored as bf16 +
bool mask), on CPU in bfloat16 (the GPU original also runs the model in bfloat16). make() first recomputes a reference
prompt whose cache already exists (A3) and refuses to write anything unless the result matches it.
"""
import hashlib
import os
import sys
from pathlib import Path

import torch

_MODEL = {}


def _load(code, qwen_path, flux2_src):
    if "m" not in _MODEL:
        sys.path.insert(0, str(Path(code) / "src"))
        from imagewam.models.backbones.flux2_imports import ensure_flux2_importable
        ensure_flux2_importable(str(flux2_src))
        from flux2.text_encoder import OUTPUT_LAYERS_QWEN3
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(min(32, os.cpu_count() or 8))
        tok = AutoTokenizer.from_pretrained(str(qwen_path))
        model = AutoModelForCausalLM.from_pretrained(str(qwen_path), torch_dtype=torch.bfloat16).eval()
        _MODEL["m"] = (tok, model, tuple(OUTPUT_LAYERS_QWEN3))
    return _MODEL["m"]


def embed(prompt, code, qwen_path, flux2_src, context_len=128):
    tok, model, layers = _load(code, qwen_path, flux2_src)
    rendered = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
    enc = tok([rendered], return_tensors="pt", padding="max_length", truncation=True, max_length=context_len)
    with torch.no_grad():
        out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, output_hidden_states=True,
                    use_cache=False)
    hidden = torch.stack([out.hidden_states[k] for k in layers], dim=1)       # [1, c, l, d]
    hidden = hidden.permute(0, 2, 1, 3).reshape(1, context_len, -1)           # b l (c d)
    return dict(text_hidden_states=hidden[0].to(torch.bfloat16).clone(),
                text_attention_mask=enc.attention_mask[0].to(torch.bool).clone())


def cache_name(prompt, context_len=128):
    return f"{hashlib.sha256(prompt.encode('utf-8')).hexdigest()}.qwen3_flux2_len{context_len}.pt"


def check_reference(ref_prompt, ref_dir, code, qwen_path, flux2_src):
    ref = torch.load(Path(ref_dir) / cache_name(ref_prompt), map_location="cpu", weights_only=True)
    new = embed(ref_prompt, code, qwen_path, flux2_src)
    if not torch.equal(ref["text_attention_mask"].bool(), new["text_attention_mask"]):
        raise SystemExit("qwen cache: attention mask differs from the reference cache")
    m = ref["text_attention_mask"].bool()
    a, b = ref["text_hidden_states"][m].float(), new["text_hidden_states"][m].float()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    rel = ((a - b).norm() / a.norm()).item()
    info = dict(min_token_cosine=float(cos.min()), mean_token_cosine=float(cos.mean()), relative_l2=rel,
                tokens=int(m.sum()))
    if info["min_token_cosine"] < 0.99 or rel > 0.05:
        raise SystemExit(f"qwen cache: CPU recomputation does not match the reference cache: {info}")
    return info


def make(prompt, out_dir, code, qwen_path, flux2_src, ref_prompt, ref_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / cache_name(prompt)
    if path.is_file():
        return path, None
    info = check_reference(ref_prompt, ref_dir, code, qwen_path, flux2_src)
    payload = embed(prompt, code, qwen_path, flux2_src)
    if not torch.isfinite(payload["text_hidden_states"].float()).all() or not payload["text_attention_mask"].any():
        raise SystemExit("qwen cache: invalid embedding")
    tmp = out_dir / f".{path.name}.tmp{os.getpid()}"
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path, info
