#!/bin/bash
# hb_relay cmd item: GR00T trolley retry r1 (skipped if the first attempt completed)
exec bash "$(dirname "$(realpath "$0")")/groot_retry.sh" trolley r1
