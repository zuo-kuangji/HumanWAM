"""CPU check that GR00T's Cosmos-Reason2-2B pieces load offline from $G/hf-home exactly as groot_node.sh will.

Usage: HF_HOME=$G/hf-home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $G/envs/groot-py312/bin/python check_reason2_offline.py
"""
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hb_offline_fix  # noqa: E402,F401
from transformers import AutoConfig, Qwen3VLForConditionalGeneration, Qwen3VLProcessor

assert os.environ.get("HF_HUB_OFFLINE") == "1", "run with HF_HUB_OFFLINE=1"
name = "nvidia/Cosmos-Reason2-2B"
torch.set_num_threads(8)
proc = Qwen3VLProcessor.from_pretrained(name, trust_remote_code=True)
cfg = AutoConfig.from_pretrained(name, trust_remote_code=True)
ids = proc.tokenizer("pick up the trash and drop it in the bin")["input_ids"]
model = Qwen3VLForConditionalGeneration.from_pretrained(name, torch_dtype=torch.bfloat16)
n = sum(p.numel() for p in model.parameters())
print("REASON2_OFFLINE_OK", json.dumps({"processor": type(proc).__name__, "tokens": len(ids),
                                        "model_type": cfg.model_type, "params_M": round(n / 1e6, 1),
                                        "llm_layers": len(model.model.language_model.layers)}), flush=True)
