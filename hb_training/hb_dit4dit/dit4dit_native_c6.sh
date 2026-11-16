#!/bin/bash
# 2026-11-13 user: long tasks (C-class cross-scene C6, and A14) train 2x the basket step budget of this model.
export HB_FIXED_STEPS=16000
# hb_relay cmd item: DiT4DiT c6, native state encoding (20 epochs, save at 10/20), chain hb1110n, global batch 64
exec bash "$(dirname "$(realpath "$0")")/dit4dit_native_relay.sh" c6
