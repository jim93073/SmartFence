#!/bin/bash
echo "Restart SmartFence process..."
pkill -KILL -f SmartFence.py 
source ~/fence_venv/bin/activate && nohup python ~/SmartFence/darknet-integration/SmartFence.py > ~/SmartFence/darknet-integration/SmartFence_log.txt 2>&1
