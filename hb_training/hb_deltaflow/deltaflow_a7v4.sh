#!/bin/bash
# hb_relay cmd item (2026-11-15 23:xx user: Delta Flow on the newest data): Delta Flow a7v4, 8000 steps,
# 256 = 32 x 8, no accumulation.
exec bash "$(dirname "$(realpath "$0")")/deltaflow_task.sh" a7v4
