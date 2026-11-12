#!/bin/bash
# Relay cmd item (2026-11-12): second GR00T batch-256 pool (A14, B7, trolley, basket), one node per job.
G=$(dirname "$(dirname "$(realpath "$0")")")
exec bash "$G/hb_pool2.sh" groot256b
