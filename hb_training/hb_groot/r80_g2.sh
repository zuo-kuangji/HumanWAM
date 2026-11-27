#!/bin/bash
# Relay cmd item (2026-11-27 user: newest pause-trim data, big three at the end of the queue; checkpoint rule r80).
G=$(dirname "$(dirname "$(realpath "$0")")")
# GR00T / Psi-Zero halves: ranks 0,1 groot:b1v2p; ranks 2,3 grooth2:a17cr4
exec bash "$G/hb_groot/gp_r80_mix.sh" groot:b1v2p grooth2:a17cr4 32114 32116
