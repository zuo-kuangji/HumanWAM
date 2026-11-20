#!/bin/bash
# 2026-11-20 user: FastWAM on every task's latest data, ahead of Delta Flow. B7 green = HF b7-green-clothes-hamper @22e3b83
# (repo root = trimmed v2, all episodes per the 11-14 episode-count rule). CPU only (training runs on the GPUs).
P=/mnt/nas/cross/Team/RobotAgent/Personal/baijiaqi/humanoid-benchmark-work/preflight-20261104T1648Z
G=$P/policies-20261109; F=$G/fastwam; E=$G/envs/fastwam/bin/python
SRC=$P/b7-green-clothes-hamper-v2trim-22e3b83/data; T=b7green; D=$F/data/$T; TC=$F/text-cache/all
exec >> $F/prep-b7green-20261120.log 2>&1
echo "$(date) start"
until grep -q DL_DONE $P/b7-green-clothes-hamper-v2trim-22e3b83/dl.log; do grep -q -E 'Error|Traceback' $P/b7-green-clothes-hamper-v2trim-22e3b83/dl.log && { echo DL_FAILED; exit 1; }; sleep 20; done
export FASTWAM_SRC=$F/FastWAM DIFFSYNTH_MODEL_BASE_PATH=$F/models DIFFSYNTH_SKIP_DOWNLOAD=true CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 HB_AUG_DIR=$G/hb_groot
unset RANK WORLD_SIZE LOCAL_RANK GROUP_RANK ROLE_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT $(compgen -e | grep -E "^PET_")
python3 - $SRC <<'PY'
import json,sys,hashlib,pathlib
s=pathlib.Path(sys.argv[1]); info=json.load(open(s/'meta/info.json'))
pay=json.load(open(s/'meta/payload_sha256.json')) if (s/'meta/payload_sha256.json').exists() else {}
items=pay.get('files',pay) if isinstance(pay,dict) else {}
miss=[k for k in items if not (s/k).exists()]
print('EPISODES',info.get('total_episodes'),'FRAMES',info.get('total_frames'),'PAYLOAD_FILES',len(items),'MISSING',len(miss),miss[:5])
print('STATE_FEATURES',[k for k in info['features'] if k.startswith('observation')])
assert not miss, 'incomplete download'
PY
[ $? = 0 ] || { echo VERIFY_DL_FAILED; exit 1; }
[ -f $D/meta/hb_fastwam_interface.json ] || $E $F/tools/hb_fastwam_convert.py g1 $SRC $D || { echo CONVERT_FAILED; exit 1; }
cat $D/meta/hb_fastwam_interface.json; echo
python3 - $D <<'PY'
import json,sys; d=json.load(open(sys.argv[1]+'/meta/hb_fastwam_interface.json'))
assert d['state_dim']==44 and d['action_dim']==76 and d['profile']=='g1', d
info=json.load(open(sys.argv[1]+'/meta/info.json')); assert not any('vel' in k for k in info['features']), 'velocity feature'
print('IFACE_OK', d['episodes'], d['frames'])
PY
[ $? = 0 ] || { echo IFACE_FAILED; exit 1; }
cd $F/FastWAM
$E scripts/precompute_text_embeds.py task=libero_uncond_2cam224_1e-4 model.redirect_common_files=false "data.train.dataset_dirs=[\"$D\"]" "data.train.text_embedding_cache_dir=$TC" && echo "$(date) TEXT_OK $(ls $TC | wc -l)" || { echo TEXT_FAILED; exit 1; }
HB_FW_PROFILE=g1 HB_DATA_SAMPLE=1 HB_SAMPLE_WORKDIR=/tmp/hbfw-sample-$T $E $F/tools/hb_fastwam_train.py "data.train.dataset_dirs=[\"$D\"]" "data.train.text_embedding_cache_dir=$TC" "model.action_dit_pretrained_path=$F/action-init-g1.pt" > $F/sample-$T.log 2>&1
grep -E 'HB_FW_AUG|HBFW_|Error' $F/sample-$T.log | cut -c1-600
grep -q HBFW_SAMPLE $F/sample-$T.log && echo "$(date) B7GREEN_READY" || echo "$(date) SAMPLE_FAILED"
