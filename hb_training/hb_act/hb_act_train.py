"""torchrun entry: register hb_act_config, apply hb_act_patches, run the pinned upstream Psi0 scripts/train.py.
Env: PSI_SRC = Psi0 checkout; HB_PROBE_OUT = dir -> probe mode (rank<R>.json with peak memory + step time).
"""
import json
import os
import runpy
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "hb_psi0"))


def main() -> None:
    psi = os.environ["PSI_SRC"]
    import hb_act_config
    sys.modules["psi.config.train.hb_act_config"] = hb_act_config
    import hb_act_patches
    hb_act_patches.apply()
    entry = os.path.join(psi, "scripts", "train.py")
    sys.argv = [entry, "hb_act_config"] + sys.argv[1:]
    probe_out = os.environ.get("HB_PROBE_OUT")
    t0 = time.time()
    ok, err = True, None
    try:
        runpy.run_path(entry, run_name="__main__")
    except BaseException as e:  # noqa: BLE001
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
            res.update(peak_reserved=torch.cuda.max_memory_reserved(dev), total=torch.cuda.get_device_properties(dev).total_memory)
            res["peak_frac"] = res["peak_reserved"] / res["total"]
        t = hb_act_patches.PROBE["times"]
        res["steps_done"] = len(t)
        if len(t) >= 4:
            res["sec_per_step"] = (t[-1] - t[1]) / (len(t) - 2)
        os.makedirs(probe_out, exist_ok=True)
        with open(os.path.join(probe_out, f"rank{rank}.json"), "w") as f:
            json.dump(res, f, indent=2)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
