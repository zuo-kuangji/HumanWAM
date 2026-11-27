#!/bin/bash
# Relay cmd item (2026-11-27 user: newest pause-trim data, big three at the end of the queue; checkpoint rule r80).
G=$(dirname "$(dirname "$(realpath "$0")")")
# GR00T / Psi-Zero halves: ranks 0,1 grooth2:a18cr4; ranks 2,3 psi0:a3v4p
exec bash "$G/hb_groot/gp_r80_mix.sh" grooth2:a18cr4 psi0:a3v4p 32118 32120
