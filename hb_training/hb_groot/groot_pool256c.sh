#!/bin/bash
# Relay cmd item (2026-11-12): GR00T C6 (full fine-tuning, 4000 steps), head of the C6 block.
G=$(dirname "$(dirname "$(realpath "$0")")")
exec bash "$G/hb_pool2.sh" groot256c
