#!/bin/bash
# Relay cmd item (2026-11-12): GR00T batch-256 re-trains (b6 first), one node per job.
G=$(dirname "$(dirname "$(realpath "$0")")")
exec bash "$G/hb_pool2.sh" groot256
