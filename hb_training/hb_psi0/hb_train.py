"""torchrun entry: register hb_psi0_config, apply hb_patches, run the pinned upstream scripts/train.py.

Usage (via torchrun): hb_train.py <native Psi0 args ...>   (config module name is inserted automatically)
Env: PSI_SRC = Psi0 checkout root (contains scripts/train.py and .env).
     HB_PROBE_OUT = dir -> probe mode: no checkpoints, writes rank<R>.json with peak GPU memory + step time.
"""
import json
import os
import runpy
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main() -> None:
    psi = os.environ["PSI_SRC"]
    import hb_psi0_config
    sys.modules["psi.config.train.hb_psi0_config"] = hb_psi0_config
    import hb_patches
    hb_patches.apply()

    entry = os.path.join(psi, "scripts", "train.py")
    sys.argv = [entry, "hb_psi0_config"] + sys.argv[1:]
    probe_out = os.environ.get("HB_PROBE_OUT")
    t0 = time.time()
    ok, err = True, None
    try:
        runpy.run_path(entry, run_name="__main__")
    except BaseException as e:  # noqa: BLE001 - probe must record OOM etc.
        ok, err = False, f"{type(e).__name__}: {e}"[:4000]
        if not probe_out:
            raise
        traceback.print_exc()
    if probe_out:
        import torch
        rank = int(os.environ.get("RANK", "0"))
        res = {"ok": ok, "error": err, "rank": rank, "wall_s": time.time() - t0}
        if torch.cuda.is_available():
            dev = torch.cuda.current_device()
            res.update(peak_reserved=torch.cuda.max_memory_reserved(dev),
                       peak_allocated=torch.cuda.max_memory_allocated(dev),
                       total=torch.cuda.get_device_properties(dev).total_memory)
            res["peak_frac"] = res["peak_reserved"] / res["total"]
        t = hb_patches.PROBE["times"]
        res["steps_done"] = len(t)
        if len(t) >= 4:
            res["sec_per_step"] = (t[-1] - t[1]) / (len(t) - 2)  # skip the first (warm-up) step
        os.makedirs(probe_out, exist_ok=True)
        with open(os.path.join(probe_out, f"rank{rank}.json"), "w") as f:
            json.dump(res, f, indent=2)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
