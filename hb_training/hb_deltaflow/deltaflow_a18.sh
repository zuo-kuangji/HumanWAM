#!/bin/bash
# hb_relay `cmd` item for the Delta-flow a18 40-epoch run (cmd items cannot pass arguments).
exec bash "$(dirname "$(realpath "$0")")/deltaflow_task.sh" a18
