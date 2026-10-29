#!/bin/bash
# Detached job on an intranet box:  tools/remote_launch.sh <host:port> <gpu> <name> <command...>
#   tools/remote_launch.sh <host>:<ssh_port> 1 geom python3 -m hesai.stage --scene KS-IN-OE-S-F1_01_202609121414
# machine-local settings (not in git): tools/local.env, e.g. ANNO_EVAL_SSH_KEY=~/.ssh/<key>
# log: /mnt/cfs/cyh/tmp/data_pipeline/logs/<name>.log
HP=$1; GPU=$2; NAME=$3; shift 3; HOST=${HP%:*}; PORT=${HP#*:}
[ -f "$(dirname "$0")/local.env" ] && . "$(dirname "$0")/local.env"; KEY=${ANNO_EVAL_SSH_KEY:-$HOME/.ssh/id_ed25519}
LOGD=/mnt/cfs/cyh/tmp/data_pipeline/logs; mkdir -p $LOGD
CMD=$(printf '%q ' "$@")
ssh -p $PORT -i $KEY -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20 root@$HOST \
  "CUDA_VISIBLE_DEVICES=$GPU OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 setsid nohup unshare -m bash /mnt/cfs/cyh/cvpr_ws/anno_eval/tools/remote_exec.sh $CMD > $LOGD/$NAME.log 2>&1 < /dev/null & echo started pid \$!"
