#!/bin/bash
# Relay cmd item (2026-11-25 user): GR00T on A20 Kosmo (HF a20-basket-transfer-kosmo-30hz-v1 @7324dbeb), the scanner is the
# ONLY variable vs groot/a20-basket-transfer/b512-trimv2-pure (basketv2p): same launcher groot_pair2x32.sh (2 nodes x 8 x 32 =
# 512, no accumulation, 4000 steps, lr 1e-4, warmup 5 %, wd 1e-5, ckpt 2000..4000 every 500, GR00T N1.7 base, seed 42,
# augmentation v2, in-training MSE every 250 on the Kosmo frames of the same train10 episodes) -> HF b512-kosmo-v1/.
# Ranks 2,3 meanwhile run two single-node GR00T runs of the basket ablation (hb_ablation/I0_fill_kosmo.spec).
G=$(dirname "$(dirname "$(realpath "$0")")"); L=$G/kosmo-20261125; T=a20kosmo
wait_ok() { for i in $(seq 2160); do [ -f "$1" ] && return 0; [ -f "$2" ] && return 1; [ $((i % 60)) = 1 ] && echo "$(date -u) waiting for $1"; sleep 10; done; return 1; }
RK=${RANK:?DLC RANK missing}
case "$RK" in
  0|1) wait_ok $L/cache-$T.ok $L/cache-$T.failed || { echo "SKIP $T: in-loop MSE cache missing"; exit 3; }
       wait_ok $L/groot-$T.ok $L/groot-$T.failed || { echo "SKIP groot $T: data view not verified"; exit 3; }
       exec bash "$G/hb_groot/groot_pair2x32.sh" $T none 29792 29794 ;;
  2|3) exec bash "$G/hb_ablation/abl_item.sh" "$G/hb_ablation/I0_fill_kosmo.spec" ;;
esac
