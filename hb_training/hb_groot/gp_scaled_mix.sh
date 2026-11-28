#!/bin/bash
# 2026-11-28 user: training steps scale with the mean episode length relative to A20 basket v2 (544.33 frames / episode),
# rounded UP to a multiple of 1000 (GR00T / Psi-Zero base 4000); big models save the last step + ~80 %. Otherwise the same
# recipes, seeds (GR00T 42, Psi-Zero 292285), layouts (2 nodes x 8 x 32 = 512, no accumulation) and gates as gp_r80_mix.sh.
#   gp_scaled_mix.sh <kind>:<task>:<steps>:<save> (ranks 0,1) <kind>:<task>:<steps>:<save> (ranks 2,3) <port01> <port23>
#   kind grooth2 -> hb_groot/groot_pair2x32_h2_scaled.sh ; psi0 -> hb_b512/psi0_multi.sh (HB_FIXED_STEPS / HB_CKPT_SAVE)
# Gates: <marker dir>/cache-<task>.ok and <groot|psi0>-<task>.ok; marker dir = newtask-20261128 for the tasks listed in its
# tasks.txt, else newtrim-20261127 (a .failed marker or 6 h without the marker = skip, exit 3).
G=$(dirname "$(dirname "$(realpath "$0")")")
RK=${RANK:?DLC RANK missing}; T0=$(date +%s)
case "$RK" in 0|1) SPEC=$1 PORT=$3 LO=0 ;; 2|3) SPEC=$2 PORT=$4 LO=2 ;; *) echo "bad rank $RK"; exit 2 ;; esac
IFS=: read -r KIND TASK STEPS SAVE <<< "$SPEC"
[ "$KIND" = none ] || [ -z "$TASK" ] && { echo "rank $RK: nothing to do"; exit 0; }
L=$G/newtask-20261128; grep -q "^$TASK " $L/tasks.txt 2>/dev/null || L=$G/newtrim-20261127
M=$KIND; [ "$KIND" = grooth2 ] && M=groot
wait_ok() { for i in $(seq 2160); do [ -f "$1" ] && return 0; [ -f "$2" ] && return 1; [ $((i % 60)) = 1 ] && echo "$(date -u) waiting for $1"; sleep 10; done; return 1; }
wait_ok $L/cache-$TASK.ok $L/cache-$TASK.failed || { echo "SKIP $KIND $TASK: in-loop MSE cache not built"; exit 3; }
wait_ok $L/$M-$TASK.ok $L/$M-$TASK.failed || { echo "SKIP $KIND $TASK: data view not verified"; exit 3; }
unset HB_DELAY_PROB
echo "$(date -u) gp_scaled_mix rank $RK: $KIND $TASK steps $STEPS save $SAVE port $PORT"
case "$KIND" in
  grooth2) export HB_STEPS=$STEPS HB_SAVE=$SAVE
           [ $LO = 0 ] && exec bash "$G/hb_groot/groot_pair2x32_h2_scaled.sh" "$TASK" none "$PORT" "$PORT"
           exec bash "$G/hb_groot/groot_pair2x32_h2_scaled.sh" none "$TASK" "$PORT" "$PORT" ;;
  psi0)    export HB_FIXED_STEPS=$STEPS HB_CKPT_SAVE=$SAVE
           MF=$G/psi0-runs/.sc-$TASK-master-addr
           if [ "$RK" = $LO ]; then IP=$(hostname -I | awk '{print $1}'); echo "$IP $(date +%s)" > "$MF"
             exec bash "$G/hb_b512/psi0_multi.sh" "$TASK" 2 0 "$IP" "$PORT" 32
           fi
           for i in $(seq 180); do [ -s "$MF" ] && [ "$(cut -d' ' -f2 "$MF")" -ge $((T0 - 300)) ] && break; sleep 5; done
           exec bash "$G/hb_b512/psi0_multi.sh" "$TASK" 2 1 "$(cut -d' ' -f1 "$MF")" "$PORT" 32 ;;
  *) echo "unknown kind $KIND"; exit 2 ;;
esac
