#!/bin/bash
# 2026-11-16: GPU smoke of the FastWAM in-training eval on the REAL training path (parent-session request), run on master-0
# during the MSE hold (relay item 75). Waits until FastWAM a10v3 finished and node 0 claimed item 75, launches the native
# accelerate/ZeRO-1 training on 8 GPUs (per-GPU batch 16 = production), a3v4 data, HB_INLOOP_EVERY=10, max_steps 40,
# save_every 0, and kills it after the step-20 eval + training steps >= 25 (before any checkpoint is written).
# PASS = HB_INLOOP lines with nMSE at steps 10 and 20, training log continues after them, no OOM / FAILED, peak GPU
# memory < 79.5 GB. FAIL -> HB_INLOOP default in hb_fastwam_train.py is switched to "0" (in-loop off, training unchanged).
# Writes smoke-fw/RESULT.json and smoke-fw/DONE (mse_hold.sh waits for DONE before releasing the relay).
E0=$(dirname "$(realpath "$0")"); G=$(dirname "$E0"); F=$G/fastwam; S=$E0/smoke-fw; mkdir -p $S
exec >> $S/driver.log 2>&1
echo "== $(date) waiting for a10v3 complete + node 0 on item 75"
until [ -f $F/runs/a10v3/complete.json ] && [ "$(cat $G/relay/pos-node-0 2>/dev/null)" -ge 75 ] 2>/dev/null; do sleep 30; done
sleep 90
echo "== $(date) launch"
rm -rf $S/out; mkdir -p $S/out
( unset RANK WORLD_SIZE LOCAL_RANK GROUP_RANK ROLE_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
  unset $(compgen -e | grep -E '^PET_') NPROC_PER_NODE WORLD_SIZE_WITH_STANDBY TORCHELASTIC_RUN_ID
  export FASTWAM_SRC=$F/FastWAM DIFFSYNTH_MODEL_BASE_PATH=$F/models DIFFSYNTH_SKIP_DOWNLOAD=true HF_HUB_OFFLINE=1 HB_AUG_DIR=$G/hb_groot HB_FW_PROFILE=g1
  export NCCL_SOCKET_IFNAME=eth GLOO_SOCKET_IFNAME=eth0 NCCL_DEBUG=WARN OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PATH=$G/envs/fastwam/bin:$PATH HB_INLOOP_EVERY=10
  cd $F/FastWAM
  exec setsid $G/envs/fastwam/bin/python -m accelerate.commands.launch --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml \
    --num_machines 1 --num_processes 8 --main_process_port 31399 \
    $F/tools/hb_fastwam_train.py "output_dir=$S/out" batch_size=16 max_steps=40 log_every=5 save_every=0 eval_every=0 \
    num_workers=8 wandb.enabled=false gradient_accumulation_steps=1 "data.train.dataset_dirs=[\"$F/data/a3v4\"]" \
    "data.train.text_embedding_cache_dir=$F/text-cache/all" "model.action_dit_pretrained_path=$F/action-init-g1.pt"
) > $S/train.log 2>&1 &
LP=$!
echo "launcher pid $LP"
peak=0; t0=$(date +%s); verdict=""
while :; do
  sleep 10
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1); [ "${m:-0}" -gt "$peak" ] && peak=$m
  if grep -Eq "OutOfMemory|CUDA out of memory|eval failed|FAILED:|install failed" $S/train.log; then verdict=fail; break; fi
  if grep -Eq "HB_INLOOP fastwam step 20: [^ ]+ nMSE" $S/train.log && grep -Eq "step=(2[5-9]|3[0-9])/40" $S/train.log; then verdict=pass; break; fi
  if ! kill -0 $LP 2>/dev/null; then verdict=fail; echo "launcher exited"; break; fi
  [ $(( $(date +%s) - t0 )) -gt 3600 ] && { verdict=fail; echo "timeout"; break; }
done
sleep 5
pkill -TERM -g $LP 2>/dev/null; kill -TERM $LP 2>/dev/null; sleep 20; pkill -KILL -g $LP 2>/dev/null
for p in $(ps -eo pid,args | grep "$S/out" | grep -v grep | awk '{print $1}'); do kill -KILL $p 2>/dev/null; done
[ "$peak" -ge 81000 ] && verdict=fail
python3 - "$S" "$verdict" "$peak" <<'EOF'
import json, re, sys
S, v, peak = sys.argv[1], sys.argv[2], int(sys.argv[3])
t = open(f"{S}/train.log", errors="replace").read()
il = re.findall(r"HB_INLOOP fastwam step (\d+): \S+ nMSE ([0-9.]+) .*? on (\d+) queries, ([0-9.]+) s", t)
loss = re.findall(r"step=(\d+)/40\s+(?:\S+\.py:\d+\s+)?loss=([0-9.]+)", t)
r = dict(verdict=v, peak_mem_mib=peak, inloop=il, loss=loss[-8:], ready=re.findall(r"HB_INLOOP fastwam ready.*", t)[:1])
json.dump(r, open(f"{S}/RESULT.json", "w"), indent=1); print(r)
EOF
if [ "$verdict" != pass ]; then
  T=$F/tools/hb_fastwam_train.py
  sed -i 's/os.environ.get("HB_INLOOP_EVERY", "250") != "0"/os.environ.get("HB_INLOOP_EVERY", "0") != "0"/' $T
  echo "FAIL -> in-loop default OFF in $T: $(grep -c 'HB_INLOOP_EVERY", "0"' $T)"
fi
rm -rf $S/out
touch $S/DONE; echo "== $(date) done $verdict peak $peak"
