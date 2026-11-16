"""GR00T N1.7 adapter: Gr00tPolicy.get_action exactly as gr00t/eval/open_loop_eval.py (eval processor, no HB patch).

ckpt = .../groot-runs-*/<task>/production/checkpoint-<step> (or the production dir itself = final weights).
The policy's own processor (statistics.json in the checkpoint) normalizes state / unnormalizes actions.
"""
import json
import sys
from pathlib import Path

import numpy as np

import hb_eval_common as E

# canonical benchmark state vector -> GR00T state groups (prepare_groot_data.py GROUPS / H2_GROUPS)
G1_STATE = {"left_leg": (0, 6), "right_leg": (6, 12), "waist": (12, 15), "left_arm": (15, 22), "right_arm": (22, 29),
            "left_hand": (29, 35), "right_hand": (35, 41), "projected_gravity": (41, 44)}
H2_STATE = {"left_leg": (0, 6), "right_leg": (6, 12), "waist": (12, 15), "left_arm": (15, 22), "right_arm": (22, 29),
            "head": (29, 31), "left_hand": (31, 53), "right_hand": (53, 75), "projected_gravity": (75, 78)}


class Adapter:
    def __init__(self, ckpt, task, split, meta, device="cuda:0", steps=None, prompt_view=None):
        sys.path.insert(0, str(E.G / "hb_groot"))
        import hb_offline_fix  # noqa: F401  tokenizer load under HF_HUB_OFFLINE=1 (same as training)
        import torch
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.policy.gr00t_policy import Gr00tPolicy
        self.attn = "native(flash_attention_2)"
        if not torch.cuda.is_available():  # CPU smoke test only: flash-attn 2 has no CPU kernel -> sdpa
            from transformers import Qwen3VLForConditionalGeneration as Q
            orig = Q.from_pretrained.__func__

            def _fp(cls, *a, **k):
                k["attn_implementation"] = "sdpa"
                return orig(cls, *a, **k)
            Q.from_pretrained = classmethod(_fp)
            self.attn = "sdpa (CPU smoke test)"
        self.policy = Gr00tPolicy(embodiment_tag=EmbodimentTag.NEW_EMBODIMENT, model_path=str(ckpt),
                                  device=device if torch.cuda.is_available() else "cpu")
        head = self.policy.model.action_head
        self.native_steps = int(head.num_inference_timesteps)
        if steps:
            head.num_inference_timesteps = int(steps)
        self.steps = int(head.num_inference_timesteps)
        self.mod = self.policy.get_modality_config()
        self.state_map = H2_STATE if meta["layout"].startswith("h2") else G1_STATE
        self.video_keys = list(self.mod["video"].modality_keys)
        self.state_keys = list(self.mod["state"].modality_keys)
        self.action_keys = list(self.mod["action"].modality_keys)
        self.lang_key = self.mod["language"].modality_keys[0]
        missing = [k for k in self.state_keys if k not in self.state_map] + \
                  [k for k in self.video_keys if k not in meta["cams"]]
        if missing:
            raise KeyError(f"checkpoint expects {missing}, not in the benchmark state/cache cameras {meta['cams']}")
        # the instruction text GR00T was trained with (its own view's tasks.jsonl)
        view = Path(prompt_view) if prompt_view else E.G / "groot-data" / task
        self.prompts = {str(r["task_index"]): r["task"] for r in E.jl(view / "meta/tasks.jsonl")} \
            if (view / "meta/tasks.jsonl").exists() else {}
        self.ckpt = str(ckpt)

    def info(self):
        return {"model": "groot", "ckpt": self.ckpt, "denoising_steps": self.steps, "native_steps": self.native_steps,
                "attention": self.attn,
                "video_keys": self.video_keys, "state_keys": self.state_keys, "action_keys": self.action_keys}

    def predict(self, obs):
        from gr00t.data.utils import parse_observation_gr00t
        o = {}
        for k in self.state_keys:
            a, b = self.state_map[k]
            o[f"state.{k}"] = np.asarray(obs["state"][a:b], dtype=np.float32)[None]  # (T=1, d)
        for k in self.video_keys:
            o[f"video.{k}"] = obs["images"][k][None]  # (T=1, H, W, 3) uint8
        prompt = self.prompts.get("0", obs["prompt"]) if len(self.prompts) == 1 else obs["prompt"]
        o[self.lang_key] = prompt
        act, _ = self.policy.get_action(parse_observation_gr00t(o, self.mod))
        return np.concatenate([np.asarray(act[k])[0] for k in self.action_keys], axis=-1)  # (16, Da)
