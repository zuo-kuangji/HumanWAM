#!/bin/bash
# 2026-11-27 17:0x user (4-hour GPU rule applies to the A800, free nodes for the big models): ranks 0 and 1 run the big-model
# r80 halves now, OUTSIDE the relay, while ranks 2 and 3 keep the small pool (FM-DiT / FLASH consolidated there by
# hb_small/consolidate_20261127.sh). Same helper, recipes, seeds, layout (2 nodes x 8 x 32) and checkpoints as the queue items.
# Order GR00T > Psi-Zero. A14 (4 x 16) and FastWAM (4 nodes) stay in the queue. Stops before the next run as soon as ranks 2
# and 3 have finished the small pool (relay item 149 done there): from then on the relay takes the queue again (finished
# GR00T halves exit "already complete"; Psi-Zero items must be removed from the queue by hand - psi0_multi has no such check).
# Start on master-0 and worker-0 (DLC env RANK / MASTER_ADDR / WORLD_SIZE): setsid nohup bash <this> > /dev/null 2>&1 &
G=$(dirname "$(dirname "$(realpath "$0")")"); L=$G/newtrim-20261127/side01; mkdir -p $L
RK=${RANK:?}; [ "$RK" = 0 ] || [ "$RK" = 1 ] || exit 2
exec >> $L/node-$RK.log 2>&1; exec 9> /tmp/hb_side01.lock; flock -n 9 || exit 0
log() { echo "$(date -u) side01 rank $RK $*"; }
log "start MASTER_ADDR=${MASTER_ADDR:-unset} WORLD_SIZE=${WORLD_SIZE:-unset}"
for it in groot:a3v4p:32110 groot:a7v4p:32112 groot:b1v2p:32114 grooth2:a17cr4:32116 grooth2:a18cr4:32118 \
          psi0:a3v4p:32120 psi0:a7v4p:32124 psi0:b1v2p:32126 psi0:a17cr4:32128 psi0:a18cr4:32130; do
  K=${it%%:*}; r=${it#*:}; T=${r%%:*}; PORT=${r#*:}
  [ -f $L/$K-$T.rc-node-$RK ] && [ "$(cat $L/$K-$T.rc-node-$RK)" = 0 ] && continue
  [ -f $L/STOP ] && { log "STOP file"; break; }
  [ -f $G/relay/item-149-done-node-2 ] && [ -f $G/relay/item-149-done-node-3 ] && { log "ranks 2,3 left the small pool: relay takes over"; break; }
  for i in $(seq 2160); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break
    [ $((i % 60)) = 1 ] && log "waiting for free GPUs before $K $T: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l) processes"; sleep 10; done
  touch $L/start-$K-$T-node-$RK
  for i in $(seq 2160); do [ -f $L/start-$K-$T-node-0 ] && [ -f $L/start-$K-$T-node-1 ] && break; sleep 5; done
  log "START $K $T port $PORT"
  bash $G/hb_groot/gp_r80_mix.sh $K:$T none:none $PORT $PORT; rc=$?
  echo $rc > $L/$K-$T.rc-node-$RK; log "END $K $T rc=$rc"
  [ $rc = 0 ] || [ $rc = 3 ] || { log "failed, side lane stops"; break; }
done
log "end"
