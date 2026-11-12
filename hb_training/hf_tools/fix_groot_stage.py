"""Drop training-only files from the already-staged groot repo (keeps the upload hash cache), refresh protocol.json."""
import json, sys
from pathlib import Path
R = Path(sys.argv[1]) / "groot"
removed = []
for d in sorted(R.glob("*/epoch-*")):
    for name in ("trainer_state.json", "training_args.bin", "training_plan.json"):
        f = d / name
        if f.exists():
            f.unlink(); removed.append(str(f.relative_to(R)))
    pp = d / "protocol.json"
    proto = json.loads(pp.read_text())
    proto.pop("trainer_global_step", None)
    proto["files"] = {str(p.relative_to(d)): p.stat().st_size for p in sorted(d.rglob("*"))
                      if p.is_file() and p.name != "protocol.json"}
    pp.write_text(json.dumps(proto, indent=2, ensure_ascii=False) + "\n")
rd = R / "README.md"
rd.write_text(rd.read_text().replace("Optimizer / DeepSpeed / RNG state is not included.",
                                     "Only what evaluation needs is included (no optimizer / trainer / resume state)."))
files = [p for p in R.rglob("*") if p.is_file() and ".cache" not in p.relative_to(R).parts]
print("REMOVED", len(removed)); print("NOW files", len(files), "GB", round(sum(p.stat().st_size for p in files) / 1e9, 2))
for d in sorted(R.glob("*/epoch-*")):
    print(d.relative_to(R), sorted(json.loads((d / "protocol.json").read_text())["files"]))
    break
