#!/bin/bash
# Relay cmd item (2026-11-12): GR00T batch-256 re-train of trolley (single node), after DiT4DiT A3/A7/basket/A6.
G=$(dirname "$(dirname "$(realpath "$0")")")
exec bash "$G/hb_pool2.sh" groot256t
