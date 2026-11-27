#!/bin/bash
# Relay cmd helper (2026-11-27 user: every baseline retrains on the newest pause-trim data; big three at the end of the
# queue). Two independent 2-node halves of one 4-node relay item, each from its base model, no accumulation, 4000 steps,
# new checkpoint rule (protocol.yaml interface.save_rule): checkpoints 3200 + 4000 only (*_r80 launchers, HB_CKPT_SAVE).
#   gp_r80_mix.sh <kind01>:<task01> <kind23>:<task23> <port01> <port23>
#   kind groot   -> hb_groot/groot_pair2x32_r80.sh      (G1, 2 x 8 x 32 = 512, seed 42)
#   kind grooth2 -> hb_groot/groot_pair2x32_h2_r80.sh   (H2 head + chest, 2 x 8 x 32 = 512, seed 42)
#   kind psi0    -> hb_b512/psi0_multi.sh 2 nodes x 8 x 32 = 512 (default recipe, no proprio delay, seed 292285)
# Gates: newtrim-20261127/cache-<task>.ok and <groot|psi0>-<task>.ok (a .failed marker or 6 h without the marker = skip).
G=$(dirname "$(dirname "$(realpath "$0")")"); L=$G/newtrim-20261127
RK=${RANK:?DLC RANK missing}; T0=$(date +%s)
case "$RK" in 0|1) SPEC=$1 PORT=$3 LO=0 ;; 2|3) SPEC=$2 PORT=$4 LO=2 ;; *) echo "bad rank $RK"; exit 2 ;; esac
KIND=${SPEC%%:*} TASK=${SPEC#*:}
[ "$TASK" = none ] && { echo "rank $RK: nothing to do"; exit 0; }
M=$KIND; [ "$KIND" = grooth2 ] && M=groot
wait_ok() { for i in $(seq 2160); do [ -f "$1" ] && return 0; [ -f "$2" ] && return 1; [ $((i % 60)) = 1 ] && echo "$(date -u) waiting for $1"; sleep 10; done; return 1; }
wait_ok $L/cache-$TASK.ok $L/cache-$TASK.failed || { echo "SKIP $KIND $TASK: in-loop MSE cache not built"; exit 3; }
wait_ok $L/$M-$TASK.ok $L/$M-$TASK.failed || { echo "SKIP $KIND $TASK: data view not verified"; exit 3; }
unset HB_DELAY_PROB
case "$KIND" in
  groot)   [ $LO = 0 ] && exec bash "$G/hb_groot/groot_pair2x32_r80.sh" "$TASK" none "$PORT" "$PORT"
           exec bash "$G/hb_groot/groot_pair2x32_r80.sh" none "$TASK" "$PORT" "$PORT" ;;
  grooth2) [ $LO = 0 ] && exec bash "$G/hb_groot/groot_pair2x32_h2_r80.sh" "$TASK" none "$PORT" "$PORT"
           exec bash "$G/hb_groot/groot_pair2x32_h2_r80.sh" none "$TASK" "$PORT" "$PORT" ;;
  psi0)    export HB_CKPT_SAVE=3200
           MF=$G/psi0-runs/.r80-$TASK-master-addr
           if [ "$RK" = $LO ]; then IP=$(hostname -I | awk '{print $1}'); echo "$IP $(date +%s)" > "$MF"
             exec bash "$G/hb_b512/psi0_multi.sh" "$TASK" 2 0 "$IP" "$PORT" 32
           fi
           for i in $(seq 180); do [ -s "$MF" ] && [ "$(cut -d' ' -f2 "$MF")" -ge $((T0 - 300)) ] && break; sleep 5; done
           exec bash "$G/hb_b512/psi0_multi.sh" "$TASK" 2 1 "$(cut -d' ' -f1 "$MF")" "$PORT" 32 ;;
  *) echo "unknown kind $KIND"; exit 2 ;;
esac
