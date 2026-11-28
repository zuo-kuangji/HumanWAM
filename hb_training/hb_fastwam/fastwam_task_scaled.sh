#!/bin/bash
# 2026-11-28 scaled copy: one FastWAM run of HB_STEPS steps (no continuation), checkpoints HB_SAVE + last (user 11-28).
# 2026-11-27 r80 copy (big-model checkpoint rule, protocol.yaml interface.save_rule): checkpoints 3200 and 4000 (the trainer always saves max_steps) only.
# Relay cmd item helper (2026-11-16 user): FastWAM on one benchmark task, 4 nodes x 8 GPUs, per-GPU batch from the
# benchmark ($G/fastwam/chosen.json: largest batch that fits), no gradient accumulation, 4000 steps (user default while
# the batch is unknown), native Gaussian FM recipe (lr 1e-4 cosine, ZeRO-1, bf16), HB augmentation v2 + proprio
# perturbation, our state/action interface. Checkpoints at 2000 and 4000. Skips (exit 3) unless the benchmark passed.
#   fastwam_task.sh <task> <profile g1|h2|h2p> <port>
G=$(dirname "$(dirname "$(realpath "$0")")")
TASK=$1 PROF=$2 PORT=$3 STEPS=${HB_STEPS:?} SAVE=${HB_SAVE:?}
F=$G/fastwam; E=$G/envs/fastwam/bin/python; R=$F/runs/$TASK
RK=${RANK:?}; M0=${MASTER_ADDR:?}
[ -f $F/fastwam.ok ] && [ -s $F/chosen.json ] || { echo "SKIP fastwam $TASK: benchmark not passed"; exit 3; }
[ -f $F/data/$TASK/meta/hb_fastwam_interface.json ] && [ -f $F/action-init-$PROF.pt ] || { echo "SKIP fastwam $TASK: data not prepared"; exit 3; }
BS=$(python3 -c "import json;print(json.load(open('$F/chosen.json'))['per_gpu_batch'])")
mkdir -p $R/logs
exec > >(tee -a "$R/logs/node-$RK.log") 2>&1
date -u; hostname
[ -f $R/complete.json ] && { echo "already complete"; exit 0; }
echo "FASTWAM task=$TASK profile=$PROF rank=$RK master=$M0:$PORT per_gpu=$BS global=$((BS * 32)) steps=$STEPS save=$SAVE"
if ! ldconfig -p | grep -q 'libavutil.so.56'; then
  apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
    libavcodec58 libavformat58 libavutil56 libswscale5 libswresample3 libavdevice58 libavfilter7 || true
fi
unset RANK WORLD_SIZE LOCAL_RANK GROUP_RANK ROLE_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
unset $(compgen -e | grep -E '^PET_') NPROC_PER_NODE WORLD_SIZE_WITH_STANDBY TORCHELASTIC_RUN_ID
export FASTWAM_SRC=$F/FastWAM DIFFSYNTH_MODEL_BASE_PATH=$F/models DIFFSYNTH_SKIP_DOWNLOAD=true HF_HUB_OFFLINE=1 HB_AUG_DIR=$G/hb_groot HB_FW_PROFILE=$PROF
export NCCL_SOCKET_IFNAME=eth GLOO_SOCKET_IFNAME=eth0 NCCL_DEBUG=WARN OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH=$G/envs/fastwam/bin:$PATH  # r2 (11-16 05:3x): r1 failed, accelerate called the missing deepspeed CLI (pdsh launcher)
for i in $(seq 36); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 5; done
cd $F/FastWAM
$E -m accelerate.commands.launch --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml --deepspeed_multinode_launcher standard \
  --num_machines 4 --machine_rank $RK --main_process_ip $M0 --main_process_port $PORT --num_processes 32 \
  $F/tools/hb_fastwam_train.py "output_dir=$R/out" "batch_size=$BS" max_steps=$STEPS log_every=20 save_every=$SAVE eval_every=0 \
  num_workers=8 wandb.enabled=false gradient_accumulation_steps=1 "data.train.dataset_dirs=[\"$F/data/$TASK\"]" \
  "data.train.text_embedding_cache_dir=$F/text-cache/all" "model.action_dit_pretrained_path=$F/action-init-$PROF.pt"
rc=$?
echo "$rc" > $R/logs/exit-node-$RK
if [ "$rc" = 0 ] && [ "$RK" = 0 ]; then
  python3 - "$R/complete.json" "$TASK" "$PROF" "$BS" "$F/data/$TASK/meta/hb_fastwam_interface.json" <<'EOF'
import json, sys, time
out, task, prof, bs, iface = sys.argv[1:6]
d = dict(task=task, profile=prof, per_gpu_batch=int(bs), global_batch=int(bs) * 32, gradient_accumulation_steps=1,
         max_steps=int(__import__("os").environ["HB_STEPS"]), save_every=int(__import__("os").environ["HB_SAVE"]), single_run=True, recipe="native FastWAM Gaussian FM (libero_uncond_2cam224_1e-4 base), HB aug v2 + proprio perturbation",
         interface=json.load(open(iface)), finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
json.dump(d, open(out, 'w'), indent=1); print('COMPLETE', d)
EOF
fi
echo "EXIT $rc"; exit $rc
