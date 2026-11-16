"""2026-11-14: auto-upload for the basket continuation runs on a20-basket-transfer-pre-reach-trim-expert-v2.
  grootcontA/basket -> groot/<basket dir>/b512-trimv2-from-b512/step-XXXXXX  (groot-runs-cont/from-b512)
  grootcontB/basket -> groot/<basket dir>/b512-trimv2-from-b256/step-XXXXXX  (groot-runs-cont/from-b256)
  psi0cont/basket   -> psi0/<basket dir>/b512-trimv2/step-002000|004000      (psi0-runs/basketv2/cont-aligned)
    python au_patch.py <hb_autoupload.py> <jobs.json>
"""
import json, shutil, sys
from pathlib import Path

f, jf = Path(sys.argv[1]), Path(sys.argv[2])
s = f.read_text()
if "collect_grootcont" not in s:
    shutil.copy2(f, f.with_name(".hb_autoupload.py.bak-cont-20261114"))
    old = '    if model in ("groot", "pi05", "psi0a", "act256", "grootft", "grootft512"):'
    assert s.count(old) == 1
    s = s.replace(old, '    if model in ("groot", "pi05", "psi0a", "act256", "grootft", "grootft512", "grootcontA", "grootcontB", "psi0cont"):')
    add = '''
# 2026-11-14 user: GR00T / Psi-Zero basket continuation, 4000 more steps from the final checkpoints on
# humanoid-benchmark/a20-basket-transfer-pre-reach-trim-expert-v2 @c4df4d40 (pre-reach waits trimmed, 54,433 frames)
TRIMV2 = "humanoid-benchmark/a20-basket-transfer-pre-reach-trim-expert-v2 @ c4df4d40e51c439d7b41697d7c0d0184c844cb93"
SUBDIR.update({("grootcontA", "basket"): "b512-trimv2-from-b512", ("grootcontB", "basket"): "b512-trimv2-from-b256",
               ("psi0cont", "basket"): "b512-trimv2"})
REPO_OF.update(grootcontA="groot", grootcontB="groot", psi0cont="psi0")
MODEL_TEXT.update(grootcontA=MODEL_TEXT["groot"], grootcontB=MODEL_TEXT["groot"], psi0cont=MODEL_TEXT["psi0"])
CODE_DIR.update(grootcontA="hb_groot", grootcontB="hb_groot", psi0cont="hb_psi0")


def collect_grootcont(run_name, src_label):
    run = G / "groot-runs-cont" / run_name
    if not (run / "production-complete.json").is_file():
        return None
    plan = rd(run / "production-complete.json")
    c = collect_groot_from(run, dict(plan, save_steps=plan["save_list"][0]))
    keep = set(c["eps"][-1]["files"])
    c["eps"] = [dict(epoch=None, step=st, name=f"step-{st:06d}",
                     files={k: run / "production" / f"checkpoint-{st}" / k for k in keep
                            if (run / "production" / f"checkpoint-{st}" / k).is_file()}) for st in plan["save_list"]]
    data = G / "groot-data" / "basketv2"
    c["common"].update({"dataset_meta/modality.json": data / "meta/modality.json",
                        "dataset_meta/hb-dataset-receipt.json": data / "hb-dataset-receipt.json"})
    c["proto"].update(epochs=None, epochs_equivalent=plan.get("epochs_equivalent"), recipe=plan.get("recipe"),
                      save_steps=plan["save_list"], gradient_accumulation_steps=plan.get("gradient_accumulation_steps"),
                      trainable="all modules (tune_llm, tune_visual, projector, diffusion action head, VLLN)",
                      training_data=TRIMV2, frames=plan.get("frames"),
                      continued_from=f"groot/<task>/{src_label}/step-004000 (trained 4000 steps on basket-transfer-expert-v1)",
                      continuation="weights only: fresh AdamW state, fresh warmup 5 % + cosine over 4000 steps; the "
                                   "source checkpoint's normalization statistics (statistics.json) are kept")
    return c


def collect_psi0cont(task):
    root = G / "psi0-runs" / "basketv2" / "cont-aligned"
    ex = root / "logs" / "prod-node-0.exit"
    c = collect_psi_like("psi0", "basket", root, ex.is_file() and ex.read_text().strip() == "0")
    if c is None:
        return None
    dec = rd(root / "decision.json")
    c["proto"].update(training_data=TRIMV2, frames=dec.get("frames"),
                      continued_from="psi0/<task>/b512/step-004000 (aligned run, 4000 steps on basket-transfer-expert-v1)",
                      continuation="weights only: fresh optimizer, fresh warmup + cosine over 4000 steps; normalization "
                                   "statistics of the source run (psi0-data/basket/meta/stats_psi0.json) are kept")
    return c


COLLECT.update(grootcontA=lambda t: collect_grootcont("from-b512", "b512"),
               grootcontB=lambda t: collect_grootcont("from-b256", "b256"), psi0cont=collect_psi0cont)

'''
    anchor = "\ndef main():"
    assert s.count(anchor) == 1
    s = s.replace(anchor, add + anchor)
    compile(s, str(f), "exec")
    f.write_text(s)
    print("PATCHED", f)
jobs = json.loads(jf.read_text())
for m in ("grootcontA", "grootcontB", "psi0cont"):
    if {"model": m, "task": "basket"} not in jobs:
        jobs.append({"model": m, "task": "basket"})
jf.write_text(json.dumps(jobs, indent=1) + "\n")
print("JOBS", len(jobs))
