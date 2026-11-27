#!/bin/bash
# Relay cmd item (2026-11-27 user: newest pause-trim data, big three at the end of the queue; checkpoint rule r80).
G=$(dirname "$(dirname "$(realpath "$0")")")
# Psi-Zero halves: ranks 0,1 psi0:a17cr4; ranks 2,3 psi0:a18cr4
exec bash "$G/hb_groot/gp_r80_mix.sh" psi0:a17cr4 psi0:a18cr4 32128 32130
