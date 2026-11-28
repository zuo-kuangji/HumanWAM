#!/bin/bash
# Relay cmd item (2026-11-28): every rank waits until side01b (Psi-Zero a17cr4 on ranks 0,1, outside the relay) has ended,
# so the next 4-node item does not start on top of it. Max 12 h.
G=$(dirname "$(dirname "$(realpath "$0")")"); L=$G/newtask-20261128
for i in $(seq 4320); do [ -f $L/side01b-a17cr4.rc-node-0 ] && [ -f $L/side01b-a17cr4.rc-node-1 ] && break
  [ $((i % 60)) = 1 ] && echo "$(date -u) gate: waiting for side01b psi0 a17cr4"; sleep 10; done
echo "$(date -u) gate passed"; exit 0
