#!/bin/bash
# Run a command on an intranet GPU box with the dev-pod Python stack (CFS copies of /opt/venv + /opt/python),
# inside a private mount namespace so the box itself is untouched:
#   ssh -p <ssh_port> root@<host> 'unshare -m bash /mnt/cfs/cyh/cvpr_ws/anno_eval/tools/remote_exec.sh python3 -m ...'
# Boxes without /opt/venv get a namespace-private tmpfs on /opt (only this process tree sees it).
RT=/mnt/cfs/cyh/tmp/cluster_runtime
mount --make-rprivate / || exit 10
if [ ! -d /opt/venv ]; then mount -t tmpfs tmpfs /opt || exit 13; mkdir -p /opt/venv /opt/python; fi
mount --bind $RT/opt_venv /opt/venv || exit 11
mkdir -p /opt/python && mount --bind $RT/opt_python /opt/python || exit 12
T=/mnt/cfs/cyh/tmp/data_pipeline
export PATH=/opt/venv/bin:$RT/bin:$PATH LD_LIBRARY_PATH=/opt/python/3.11.9/lib:${LD_LIBRARY_PATH:-} PYTHONUNBUFFERED=1
export TMPDIR=$T/tmp HF_HOME=$T/hf XDG_CACHE_HOME=$T/cache TORCH_HOME=$T/torch TRITON_CACHE_DIR=$T/triton \
       TORCHINDUCTOR_CACHE_DIR=$T/inductor TORCH_EXTENSIONS_DIR=$T/torch_ext/$(hostname) CUDA_CACHE_PATH=$T/cuda \
       PIP_CACHE_DIR=$T/pip TORCH_CUDA_ARCH_LIST=8.0
mkdir -p $TMPDIR
cd /mnt/cfs/cyh/cvpr_ws/anno_eval
exec "$@"
