#!/bin/bash
echo "[Info] Restart the SmartFence process..."
pkill -KILL -f SmartFence.py 
source /home/jim/fence_venv/bin/activate && nohup python /home/jim/SmartFence/darknet-integration/SmartFence.py > ~/SmartFence/darknet-integration/SmartFence_log.txt 2>&1
