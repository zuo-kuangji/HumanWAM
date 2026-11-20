"""2026-11-20 user: FastWAM continuation, 2000 more steps from a finished 4000-step run: weights only, fresh AdamW,
warmup 5 % to peak lr 3e-5 (0.3 x the original 1e-4), cosine to 1 % (FastWAM's own scheduler, learning_rate=3e-5).
FastWAM's own resume=<file.pt> loads the weights AFTER accelerator.prepare (DeepSpeed ZeRO-1, bf16), where the
optimizer's fp32 master copy would overwrite them at the first step; so the DiT (mot) and proprio encoder weights are
loaded here into the freshly built model, before the trainer and DeepSpeed are created (strict).
Same arguments as hb_fastwam_train.py; env HB_FW_INIT=<runs/<task>/out/checkpoints/weights/step_004000.pt>."""
import json
import os
import runpy
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
INIT = os.environ["HB_FW_INIT"]
sys.path.insert(0, str(Path(os.environ["FASTWAM_SRC"]).resolve() / "src"))
sys.path.insert(0, str(HERE))
import fastwam.runtime as native  # noqa: E402

_Trainer = native.Wan22Trainer


class ContTrainer(_Trainer):
    def __init__(self, cfg, model, train_dataset, val_dataset):
        p = torch.load(INIT, map_location="cpu", mmap=True, weights_only=False)
        model.mot.load_state_dict(p["mot"], strict=True)
        pe = getattr(model, "proprio_encoder", None)
        if pe is not None:
            pe.load_state_dict(p["proprio_encoder"], strict=True)
        print("HBFW_INIT", json.dumps(dict(path=INIT, saved_step=p.get("step"), mot_tensors=len(p["mot"]),
                                           proprio_encoder=pe is not None)), flush=True)
        del p
        super().__init__(cfg=cfg, model=model, train_dataset=train_dataset, val_dataset=val_dataset)


native.Wan22Trainer = ContTrainer
sys.argv[0] = str(HERE / "hb_fastwam_train.py")
runpy.run_path(sys.argv[0], run_name="__main__")
