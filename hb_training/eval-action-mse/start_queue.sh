#!/bin/bash
# start the v2 eval queue on this node (kills an older queue instance first); output -> logs/start-<host>.txt
E=$(dirname "$(realpath "$0")"); H=$(hostname); cd "$E" || exit 1
{ date; echo "before:"; ps -eo pid,etime,args | grep 'eval_queue.py' | grep -v grep
  pkill -TERM -f 'python3 eval_queue.py' ; sleep 3
  setsid nohup python3 eval_queue.py >> "logs/queue-$H.log" 2>&1 < /dev/null &
  sleep 8; echo "after:"; ps -eo pid,etime,args | grep 'eval_queue.py' | grep -v grep; tail -3 "logs/queue-$H.log"; } > "logs/start-$H.txt" 2>&1
