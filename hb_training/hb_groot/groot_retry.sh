#!/bin/bash
# hb_relay cmd item: re-run GR00T <task> production if the previous attempt did not complete (e.g. a node was still
# holding GPUs). Skips when production-complete.json exists. Node 0 moves the stale exit files aside and writes a
# marker; the other nodes wait for the marker, so no node reads a stale exit file.
H=$(dirname "$(realpath "$0")"); G=$(dirname "$H")
T=${1:?usage: groot_retry.sh <task> <tag>}; TAG=${2:?usage: groot_retry.sh <task> <tag>}
R=$G/groot-runs-bmax/$T
if [ -f "$R/production-complete.json" ]; then echo "GR00T $T already complete; retry $TAG skipped"; exit 0; fi
M=$R/logs/retry-$TAG.cleared
mkdir -p "$R/logs"
if [ "$RANK" = 0 ]; then
  mkdir -p "$R/logs/stale-$TAG"
  mv "$R"/logs/production-node-*.exit "$R/logs/stale-$TAG/" 2>/dev/null
  date -u > "$M.tmp" && mv "$M.tmp" "$M"
else
  for _ in $(seq 360); do [ -f "$M" ] && break; sleep 5; done
  [ -f "$M" ] || { echo "GR00T retry $TAG: no marker from node 0"; exit 1; }
fi
echo "GR00T $T retry $TAG on node $RANK"
exec bash "$H/groot_relay.sh" "$T"
