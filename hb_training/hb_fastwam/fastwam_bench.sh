#!/bin/bash
# Relay cmd item (2026-11-16 user): FastWAM training benchmark on 4 nodes x 8 GPUs, A10 v3 on our interface with the HB
# augmentation v2. Per-GPU batch sizes are tried in increasing order, 30 steps each (log every 5, no checkpoint, no
# eval); the first failure (OOM etc.) stops the search; peak GPU memory is sampled by nvidia-smi on every node. Hard cap
# 100 min, then the queue continues; always exits 0. Rank 0 writes $G/fastwam/bench-20261116/summary.txt and, when at
# least one batch trained, $G/fastwam/chosen.json (largest batch whose peak stays <= 76,000 MiB) + $G/fastwam/fastwam.ok,
# which the FastWAM training items wait for.
G=$(dirname "$(dirname "$(realpath "$0")")")
F=$G/fastwam; B=$F/bench-20261116-r2; E=$G/envs/fastwam/bin/python
RK=${RANK:?}; M0=${MASTER_ADDR:?}; T0=$(date +%s)
mkdir -p $B
exec > >(tee -a "$B/node-$RK.log") 2>&1
echo "$(date -u) fastwam_bench rank $RK master $M0"
[ -f $F/bench.ready ] || { echo "SKIP: preparation not ready"; exit 0; }
if ! ldconfig -p | grep -q 'libavutil.so.56'; then
  apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
    libavcodec58 libavformat58 libavutil56 libswscale5 libswresample3 libavdevice58 libavfilter7 || true
fi
unset RANK WORLD_SIZE LOCAL_RANK GROUP_RANK ROLE_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
unset $(compgen -e | grep -E '^PET_') NPROC_PER_NODE WORLD_SIZE_WITH_STANDBY TORCHELASTIC_RUN_ID
export FASTWAM_SRC=$F/FastWAM DIFFSYNTH_MODEL_BASE_PATH=$F/models DIFFSYNTH_SKIP_DOWNLOAD=true HF_HUB_OFFLINE=1 HB_AUG_DIR=$G/hb_groot HB_FW_PROFILE=g1
export NCCL_SOCKET_IFNAME=eth GLOO_SOCKET_IFNAME=eth0 NCCL_DEBUG=WARN OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH=$G/envs/fastwam/bin:$PATH  # r2 (11-16 05:3x): r1 failed, accelerate called the missing deepspeed CLI (pdsh launcher)
D=$F/data/a10v3; TC=$F/text-cache/all; AI=$F/action-init-g1.pt
cd $F/FastWAM
k=0
for bs in 16 32 48 64 96; do
  k=$((k + 1)); T=$B/bs$bs; mkdir -p $T
  [ -f $B/STOP ] && break
  [ $(( $(date +%s) - T0 )) -gt 6000 ] && { echo "time cap"; break; }
  for i in $(seq 36); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 5; done
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -l 3 > $T/mem-$RK.txt &
  SM=$!
  echo "$(date -u) trial bs=$bs per GPU (global $((bs * 32)))"
  timeout 1500 $E -m accelerate.commands.launch --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml --deepspeed_multinode_launcher standard \
    --num_machines 4 --machine_rank $RK --main_process_ip $M0 --main_process_port $((31100 + k)) --num_processes 32 \
    $F/tools/hb_fastwam_train.py "output_dir=$T/out" "batch_size=$bs" max_steps=30 log_every=5 save_every=0 eval_every=0 \
    num_workers=8 wandb.enabled=false "data.train.dataset_dirs=[\"$D\"]" "data.train.text_embedding_cache_dir=$TC" \
    "model.action_dit_pretrained_path=$AI" > $T/train-$RK.log 2>&1
  rc=$?
  kill $SM 2>/dev/null; pkill -f hb_fastwam_train.py 2>/dev/null; sleep 5; pkill -9 -f hb_fastwam_train.py 2>/dev/null
  echo $rc > $T/rc-$RK
  for i in $(seq 120); do [ "$(ls $T/rc-* 2>/dev/null | wc -l)" = 4 ] && break; sleep 5; done
  bad=0
  for f in $T/rc-*; do [ "$(cat $f)" = 0 ] || bad=1; done
  grep -q -i 'out of memory' $T/train-*.log && bad=1
  if [ $RK = 0 ]; then
    sp=$(grep -a -o 'speed=[0-9.]* step/s, [0-9.]* samples/s' $T/train-0.log | tail -2 | tr '\n' ';')
    mx=$(for f in $T/mem-*.txt; do sort -t, -k1 -n $f | tail -1 | cut -d, -f1; done | sort -n | tail -1)
    echo "bs=$bs global=$((bs * 32)) rc=[$(cat $T/rc-* | tr '\n' ' ')] oom=$(grep -l -i 'out of memory' $T/train-*.log | wc -l) peak_mem_MiB=$mx speed=$sp" >> $B/summary.txt
    [ $bad = 0 ] && echo "$bs $mx" >> $B/ok.txt
    [ $bad = 1 ] && touch $B/STOP
  fi
  [ $bad = 1 ] && break
done
if [ $RK = 0 ] && [ -s $B/ok.txt ]; then
  pick=$(awk '$2 <= 76000 {b=$1} END {print b}' $B/ok.txt)
  [ -z "$pick" ] && pick=$(head -1 $B/ok.txt | cut -d' ' -f1)
  echo "{\"per_gpu_batch\": $pick, \"global_batch\": $((pick * 32)), \"source\": \"$B/summary.txt\"}" > $F/chosen.json
  touch $F/fastwam.ok
  echo "CHOSEN per-GPU batch $pick"
fi
echo "$(date -u) fastwam_bench done on rank $RK"
exit 0
