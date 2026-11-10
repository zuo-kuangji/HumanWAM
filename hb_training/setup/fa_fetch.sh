#!/bin/bash
# Replace the slow single-connection flash-attn downloads with pdl.py (32 ranges), then resume the Psi0 build/prep
# and install flash-attn into the GR00T env. Usage: setsid nohup bash fa_fetch.sh > $G/fa-fetch.log 2>&1 &
G=$(dirname "$(realpath "$0")")
W=$G/wheels
GP=$G/envs/groot-py312/bin/python
PSI_WHL=$(ls "$W" | grep -o '^flash_attn-2\.7\.4\.post1+cu12torch2\.7cxx11abi[A-Z]*-cp311-cp311-linux_x86_64\.whl' | head -1)
GR_WHL=flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
echo "psi0 wheel: ${PSI_WHL:?psi0 wheel name not found in $W}"
pkill -f "seq 200" ; pkill -f "wheels/flash_attn" ; sleep 3
rm -f "$W/$PSI_WHL" "$W/$GR_WHL"
REL=https://github.com/Dao-AILab/flash-attention/releases/download
( "$GP" -u "$G/pdl.py" "$REL/v2.8.3/${GR_WHL//+/%2B}" "$W/$GR_WHL" 32 \
  && LD_LIBRARY_PATH=$G/envs/groot-py312/lib/hbstub "$GP" -m pip install -q --no-deps "$W/$GR_WHL" \
  && touch "$W/$GR_WHL.done" \
  && LD_LIBRARY_PATH=$G/envs/groot-py312/lib/hbstub "$GP" -c 'import flash_attn; print("GROOT_FLASH_ATTN_OK", flash_attn.__version__)' ) &
"$GP" -u "$G/pdl.py" "$REL/v2.7.4.post1/${PSI_WHL//+/%2B}" "$W/$PSI_WHL" 32 || { echo PSI_WHEEL_FAIL; wait; exit 1; }
pgrep -f "hb_psi0/build_env.sh" > /dev/null && { echo "psi0 build still running?"; }
bash "$G/hb_psi0/build_env.sh" > "$G/hb_psi0_build_env.log" 2>&1
grep -q BUILD_ENV_OK "$G/hb_psi0_build_env.log" && echo PSI0_BUILD_OK || echo PSI0_BUILD_FAIL
bash "$G/hb_psi0/prep_all.sh" > "$G/psi0-prep-all.log" 2>&1
wait
echo FA_FETCH_DONE
