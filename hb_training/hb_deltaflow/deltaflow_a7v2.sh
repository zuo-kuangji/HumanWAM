#!/bin/bash
# hb_relay cmd item: Delta-flow a7v2 (2026-11-14 user), 8000 steps, global 256.
exec bash "$(dirname "$(realpath "$0")")/deltaflow_task.sh" a7v2
