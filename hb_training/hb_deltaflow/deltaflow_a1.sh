#!/bin/bash
# hb_relay `cmd` item for the Delta-flow A1 run (8000 steps; cmd items cannot pass arguments).
exec bash "$(dirname "$(realpath "$0")")/deltaflow_task.sh" a1
