#!/bin/bash
# Relay cmd item (2026-11-12): ACT batch-256 task pool, one node per job (least urgent, after the aligned re-trains).
G=$(dirname "$(dirname "$(realpath "$0")")")
exec bash "$G/hb_pool2.sh" act256
