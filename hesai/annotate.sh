#!/bin/bash
# Run on a GPU box inside tools/remote_exec.sh (cwd = anno_eval):
#   tools/remote_launch.sh <host>:<ssh_port> 1 annotate bash hesai/annotate.sh <scene_key>
# Annotation flow: SAM3 labelling (GPT per-frame category inventory -> SAM3 masks -> 3D instances + multi-view vote)
# -> 'unknown' instances labelled once by gpt-6.1-sol -> fragments no frame shows are dropped (export stage).
# The GPU is held only while SAM3 segments (the boxes' GPUs are in Exclusive_Process mode).
# Optional text layer (per-frame descriptions + five-level QA): WITH_TEXT=1.
set -euo pipefail
SCENE=$1; PORTS=${SAM_PORTS:-18863}
LOGD=/mnt/cfs/cyh/tmp/data_pipeline/logs; mkdir -p $LOGD
export GLOBAL_VLM_MODEL=gpt-6.1-sol VLM_ATTEMPTS=5
export GLOBAL_SAM_URLS=$(for P in $PORTS; do printf 'http://127.0.0.1:%s,' $P; done | sed 's/,$//')
run(){ echo "== stage $1 $(date)"; python3 -m global_pipeline.run --scene "$SCENE" --stage "$1"; }
run frames
run discover
# Exclusive_Process GPUs: wait until no other process holds the visible GPU (e.g. a render job on the same card)
UUID=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g="${CUDA_VISIBLE_DEVICES:-0}" '$1==g{print $2}')
while nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$UUID"; do echo "GPU busy, waiting $(date)"; sleep 30; done
PIDS=()
cleanup(){ for p in "${PIDS[@]}"; do kill $p 2>/dev/null || true; done; }
trap cleanup EXIT
for PORT in $PORTS; do
  SAM3_PORT=$PORT OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python3 -u -m annotation_pipeline.sam_service > $LOGD/sam3_$PORT.log 2>&1 &
  PIDS+=($!)
done
for PORT in $PORTS; do
  for i in $(seq 1 300); do
    if python3 -c "import requests,sys;s=requests.Session();s.trust_env=False;sys.exit(0 if s.get('http://127.0.0.1:$PORT/health',timeout=2).json().get('ready') else 1)" 2>/dev/null; then echo "SAM3 $PORT ready"; break; fi
    sleep 2
    [ $i -eq 300 ] && { echo "SAM3 $PORT failed to start"; exit 3; }
  done
done
run segment
cleanup; PIDS=(); echo "SAM3 stopped, GPU released $(date)"
for S in geometry instances export; do run $S; done
if [ "${WITH_TEXT:-0}" = "1" ]; then
  run describe
  FRAME_QA_GPT_MODEL=gpt-6.1-sol python3 -m global_pipeline.frame_qa --scene "$SCENE" --workers 4
  FRAME_QA_GPT_MODEL=gpt-6.1-sol python3 -m global_pipeline.frame_qa_grounded --scene "$SCENE" --workers 4
fi
run verify
echo "ANNOTATION DONE $(date)"
