#!/bin/bash
# 2026-11-16: CPU-only preparation of FastWAM for the 12 Delta-flow tasks (no GPU is touched: training is running).
# Data copies on our interface, ActionDiT initializers per profile, one shared T5 text cache, augmented-sample checks.
P=/mnt/nas/cross/Team/RobotAgent/Personal/baijiaqi/humanoid-benchmark-work/preflight-20261104T1648Z
G=$P/policies-20261109
F=$G/fastwam
E=$G/envs/fastwam/bin/python
exec >> $F/prep.log 2>&1
echo "$(date) start (v2: 12 tasks)"
until [ -f $G/envs/fastwam/.hb_ready ] || grep -q ENV_FAILED $F/setup.log; do sleep 30; done
[ -f $G/envs/fastwam/.hb_ready ] || { echo "ENV_FAILED"; exit 1; }
grep -q FW_DL_DONE $F/dl2.log || { echo "DOWNLOAD_FAILED"; exit 1; }
export FASTWAM_SRC=$F/FastWAM DIFFSYNTH_MODEL_BASE_PATH=$F/models DIFFSYNTH_SKIP_DOWNLOAD=true CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1
export HB_AUG_DIR=$G/hb_groot
unset RANK WORLD_SIZE LOCAL_RANK GROUP_RANK ROLE_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT $(compgen -e | grep -E "^PET_")
TC=$F/text-cache/all
# task profile source [first N episodes]
cat > $F/tasks.txt <<EOF
a10v3 g1 $P/a10-trolley-push-v3-6e410c5/data
a3v4 g1 $P/a3-fridge-open-v4-6393d4a/data
a7v4 g1 $P/a7-cabinet-close-v4-90cc2b5/data
b7 g1 $P/b7-black-clothes-hamper-v1-1bb002a/data 100
b5 g1 $P/b5-chair-push-v1-b85b7e9/data
b8 g1 $P/b8-table-wipe-v1-3b2e618/data
a17cr2 h2 $P/a17-unscrew-bottle-cropped-v2-3520b70/data
basketv2 g1 $P/basket-trimv2-c4df4d4/data
c6 g1 $P/c6-bathroom-door-fetch-v1-3afba69/data
b6v6 g1 $P/b6-pedal-bin-v6-ad2f5bf/data
a14v2 h2p $P/a14-unscrew-bulb-v2-8411a6d/data
b1 g1 $P/b1-riceball-microwave-v1-36da298/data
EOF
DIRS=""
while read t prof src lim; do
  D=$F/data/$t
  if [ ! -f $D/meta/hb_fastwam_interface.json ]; then
    $E $F/tools/hb_fastwam_convert.py $prof $src $D ${lim:+--episodes $lim} || { echo "CONVERT_FAILED $t"; continue; }
  fi
  DIRS="$DIRS,\"$D\""
done < $F/tasks.txt
DIRS="[${DIRS#,}]"
echo "$(date) CONVERTED $DIRS"
cd $F/FastWAM
for prof in g1 h2 h2p; do
  AI=$F/action-init-$prof.pt
  [ -f $AI ] && continue
  HB_FW_PROFILE=$prof HB_MODEL_YAML=$F/model-$prof.yaml $E $F/tools/hb_fastwam_train.py "data.train.dataset_dirs=$DIRS" "data.train.text_embedding_cache_dir=$TC" || { echo "YAML_FAILED $prof"; continue; }
  grep -E 'action_dim|proprio_dim' $F/model-$prof.yaml | tr '\n' ' '; echo
  $E scripts/preprocess_action_dit_backbone.py --model-config $F/model-$prof.yaml --output $AI --device cpu --dtype bfloat16 && echo "$(date) INIT_OK $prof $(stat -c %s $AI)" || echo "INIT_FAILED $prof"
done
mkdir -p $TC
$E scripts/precompute_text_embeds.py task=libero_uncond_2cam224_1e-4 model.redirect_common_files=false "data.train.dataset_dirs=$DIRS" "data.train.text_embedding_cache_dir=$TC" && echo "$(date) TEXT_OK $(ls $TC | wc -l) files" || echo TEXT_FAILED
ok=1
for pair in a10v3:g1 a17cr2:h2 a14v2:h2p; do
  t=${pair%%:*}; prof=${pair##*:}
  HB_FW_PROFILE=$prof HB_DATA_SAMPLE=1 HB_SAMPLE_WORKDIR=/tmp/hbfw-sample-$t $E $F/tools/hb_fastwam_train.py "data.train.dataset_dirs=[\"$F/data/$t\"]" "data.train.text_embedding_cache_dir=$TC" "model.action_dit_pretrained_path=$F/action-init-$prof.pt" > $F/sample-$t.log 2>&1
  grep -E 'HB_FW_AUG|HBFW_|Error' $F/sample-$t.log | cut -c1-800
  grep -q "HBFW_SAMPLE" $F/sample-$t.log || { ok=0; echo "SAMPLE_FAILED $t"; }
done
for t in $(cut -d' ' -f1 $F/tasks.txt); do [ -f $F/data/$t/meta/hb_fastwam_interface.json ] || { ok=0; echo "MISSING_DATA $t"; }; done
for prof in g1 h2 h2p; do [ -f $F/action-init-$prof.pt ] || { ok=0; echo "MISSING_INIT $prof"; }; done
[ $ok = 1 ] && touch $F/bench.ready && echo "$(date) BENCH_READY" || echo "$(date) NOT_READY"
