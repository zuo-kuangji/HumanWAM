#!/bin/bash
# 2026-11-22 user: C6 (long task, 2 x steps) continuation = 4000 more steps from step_008000 (total 12000), else identical to fastwam_cont.sh
# Relay cmd item helper (2026-11-20 user): FastWAM continuation of a finished 4000-step run, 2000 more steps.
# Weights only (DiT + proprio encoder of runs/<task>/out/checkpoints/weights/step_008000.pt, loaded before DeepSpeed
# init by fastwam/tools/hb_fastwam_cont.py), fresh AdamW, warmup 5 % to peak lr 3e-5 (0.3 x 1e-4), cosine to 1 %;
# same data, normalization, augmentation v2 + proprio perturbation, per-GPU batch, 4 nodes x 8, no accumulation.
# Checkpoints at 1000 and 2000 (= total steps 5000 / 6000). Run dir runs/<task>-cont2k.
#   fastwam_cont.sh <task> <profile g1|h2|h2p> <port>
G=$(dirname "$(dirname "$(realpath "$0")")")
TASK=$1 PROF=$2 PORT=$3 STEPS=4000 SAVE=2000 LR=3e-5
F=$G/fastwam; E=$G/envs/fastwam/bin/python; S=$F/runs/$TASK; R=$F/runs/$TASK-cont4k
INIT=$S/out/checkpoints/weights/step_008000.pt
RK=$RANK; M0=$MASTER_ADDR
[ -n "$RK" ] && [ -n "$M0" ] || { echo "RANK / MASTER_ADDR unset"; exit 1; }
[ -f $S/complete.json ] && [ -s $INIT ] || { echo "SKIP fastwam cont $TASK: no finished 8000-step run"; exit 3; }
[ -f $F/fastwam.ok ] && [ -s $F/chosen.json ] && [ -f $F/data/$TASK/meta/hb_fastwam_interface.json ] && [ -f $F/action-init-$PROF.pt ] || { echo "SKIP fastwam cont $TASK: not prepared"; exit 3; }
BS=$(python3 -c "import json;print(json.load(open('$F/chosen.json'))['per_gpu_batch'])")
mkdir -p $R/logs
[ "$RK" = 0 ] && ln -sfn $TASK $F/data/$TASK-cont4k
exec > >(tee -a "$R/logs/node-$RK.log") 2>&1
date -u; hostname
[ -f $R/complete.json ] && { echo "already complete"; exit 0; }
echo "FASTWAM-CONT task=$TASK profile=$PROF rank=$RK master=$M0:$PORT per_gpu=$BS global=$((BS * 32)) steps=$STEPS save_every=$SAVE lr=$LR init=$INIT"
if ! ldconfig -p | grep -q 'libavutil.so.56'; then apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends libavcodec58 libavformat58 libavutil56 libswscale5 libswresample3 libavdevice58 libavfilter7 || true; fi
unset RANK WORLD_SIZE LOCAL_RANK GROUP_RANK ROLE_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
unset $(compgen -e | grep -E '^PET_') NPROC_PER_NODE WORLD_SIZE_WITH_STANDBY TORCHELASTIC_RUN_ID
export FASTWAM_SRC=$F/FastWAM DIFFSYNTH_MODEL_BASE_PATH=$F/models DIFFSYNTH_SKIP_DOWNLOAD=true HF_HUB_OFFLINE=1 HB_AUG_DIR=$G/hb_groot HB_FW_PROFILE=$PROF HB_FW_INIT=$INIT
export NCCL_SOCKET_IFNAME=eth GLOO_SOCKET_IFNAME=eth0 NCCL_DEBUG=WARN OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH=$G/envs/fastwam/bin:$PATH
for i in $(seq 36); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 5; done
cd $F/FastWAM
$E -m accelerate.commands.launch --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml --deepspeed_multinode_launcher standard --num_machines 4 --machine_rank $RK --main_process_ip $M0 --main_process_port $PORT --num_processes 32 $F/tools/hb_fastwam_cont.py "output_dir=$R/out" "batch_size=$BS" max_steps=$STEPS log_every=20 save_every=$SAVE eval_every=0 num_workers=8 wandb.enabled=false gradient_accumulation_steps=1 learning_rate=$LR 'data.train.dataset_dirs=["'$F/data/$TASK'"]' "data.train.text_embedding_cache_dir=$F/text-cache/all" "model.action_dit_pretrained_path=$F/action-init-$PROF.pt"
rc=$?
echo "$rc" > $R/logs/exit-node-$RK
if [ "$rc" = 0 ] && [ "$RK" = 0 ]; then
  python3 - "$R/complete.json" "$TASK" "$PROF" "$BS" "$F/data/$TASK/meta/hb_fastwam_interface.json" "$INIT" "$LR" <<'PYEOF'
import json, sys, time
out, task, prof, bs, iface, init, lr = sys.argv[1:8]
d = dict(task=task + "-cont4k", base_task=task, profile=prof, per_gpu_batch=int(bs), global_batch=int(bs) * 32,
         gradient_accumulation_steps=1, max_steps=4000, save_every=2000, continued_from=init, continued_from_step=8000,
         total_steps_after=12000, peak_lr=float(lr),
         recipe="native FastWAM Gaussian FM (libero_uncond_2cam224_1e-4 base), HB aug v2 + proprio perturbation; "
                "continuation: weights only, fresh AdamW, warmup 5 % to peak lr 3e-5 (0.3 x 1e-4), cosine to 1 %",
         interface=json.load(open(iface)), finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
json.dump(d, open(out, 'w'), indent=1); print('COMPLETE', d)
PYEOF
fi
echo "EXIT $rc"; exit $rc
