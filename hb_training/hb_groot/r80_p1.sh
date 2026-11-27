#!/bin/bash
# Relay cmd item (2026-11-27 user: newest pause-trim data, big three at the end of the queue; checkpoint rule r80).
G=$(dirname "$(dirname "$(realpath "$0")")")
# Psi-Zero halves: ranks 0,1 psi0:a7v4p; ranks 2,3 psi0:b1v2p
exec bash "$G/hb_groot/gp_r80_mix.sh" psi0:a7v4p psi0:b1v2p 32124 32126
