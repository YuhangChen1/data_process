#!/bin/bash
# Exclusive_Process GPUs: wait until no process holds the visible GPU, then exec the command.
#   tools/remote_launch.sh <host>:<ssh_port> 1 job bash tools/gpu_wait.sh python3 -m ...
UUID=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g="${CUDA_VISIBLE_DEVICES:-0}" '$1==g{print $2}')
while nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$UUID"; do sleep 20; done
exec "$@"
