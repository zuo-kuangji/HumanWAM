#!/bin/bash
# 2026-11-16 00:xx user: FastWAM environment for a training benchmark on our data.
# Copy of the Delta-flow venv (Python 3.11, torch 2.7.1+cu128 / torchvision 0.22.1 = FastWAM's pins) + FastWAM's other pins.
P=/mnt/nas/cross/Team/RobotAgent/Personal/baijiaqi/humanoid-benchmark-work/preflight-20261104T1648Z
G=$P/policies-20261109
F=$G/fastwam
E=$G/envs/fastwam
exec >> $F/setup.log 2>&1
echo "$(date) start"
if [ ! -f $E/.hb_ready ]; then
  [ -d $E ] || cp -a $P/venv $E
  $E/bin/python -c "import sys, torch, torchvision; print('ENV_BASE', sys.prefix, sys.version.split()[0], torch.__version__, torchvision.__version__)"
  REQ=$(python3 - "$F/FastWAM/pyproject.toml" <<'PY'
import re, sys
t = open(sys.argv[1]).read()
deps = re.findall(r'"([^"]+)"', t.split('dependencies = [')[1].split(']')[0])
print(' '.join(d for d in deps if not d.startswith(('torch==', 'torchvision=='))))
PY
)
  echo "REQ $REQ"
  $E/bin/python -m ensurepip --upgrade && $E/bin/python -m pip install --no-input --no-build-isolation -i https://mirrors.aliyun.com/pypi/simple/ $REQ \
    && $E/bin/python -m pip install --no-input --no-deps --no-build-isolation -i https://mirrors.aliyun.com/pypi/simple/ -e $F/FastWAM \
    && $E/bin/python -c "import torch, transformers, accelerate, deepspeed, hydra, fastwam; print('ENV_OK', torch.__version__, transformers.__version__, accelerate.__version__, deepspeed.__version__)" \
    && touch $E/.hb_ready || echo ENV_FAILED
fi
echo "$(date) env done"
