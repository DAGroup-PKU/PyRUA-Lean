#!/usr/bin/env bash
# usage: H3_RUN=<run dir> [H3_PROFILE=control-turbo] launch_control_worker.sh [memory_mode=full] [extra scheduler args...]
# Submits ONE resident one-GPU Ref2VA+ControlNet hybrid worker (h3_control_worker.py) through the cooperative
# scheduler (never preempts leases; the lease is released when the worker exits: idle-exit H3_IDLE_EXIT seconds,
# default 1800, H3_MAX_TICKETS tickets, or RUN/queue/STOP).  Scale out by submitting it once per GPU, or keep a pool
# with h3_pool.py.
# Local mode: H3_LOCAL_GPU=<index> runs the same worker in the foreground on that GPU of this host (no scheduler, no
#   ALLOWED_HOSTS), for a dedicated node the scheduler cannot reach; h3_pool.py uses it.  The caller picks the GPU.
# Deployment bindings (h3_paths.py, no built-in machine paths; fail closed when absent): H3_RUN, GPU_SCHEDULER
#   (scheduler mode), H3_PYTHON, H3_VIDEOX_FUN, H3_MODELS, and for turbo profiles (suffix -turbo) H3_TURBO_LORA.
# Allowed hosts: one whitespace-separated line in $H3_RUN/queue/ALLOWED_HOSTS (deployment/user GPU scope).
set -euo pipefail
MODE=${1:-full}; shift $(( $# < 1 ? $# : 1 ))
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
paths() { python3 "$HERE/h3_paths.py" get "$1"; }
RUN=$(paths H3_RUN)
PY=$(paths H3_PYTHON)
LOCAL=${H3_LOCAL_GPU:-}
[ -n "$LOCAL" ] || SCHED=$(paths GPU_SCHEDULER)
VXF=$(paths H3_VIDEOX_FUN)
MODELS=$(paths H3_MODELS)
TURBO=""
PROFILE=${H3_PROFILE:-control-turbo}
case "$PROFILE" in *-turbo) TURBO="--turbo-lora $(paths H3_TURBO_LORA)";; esac
IDLE=${H3_IDLE_EXIT:-1800}   # expanded here: the scheduler does not forward this shell's env
MAXT=${H3_MAX_TICKETS:+--max-tickets $H3_MAX_TICKETS}
mkdir -p $RUN/queue/pending/$PROFILE $RUN/queue/logs
WORKER="unset PYTHONPATH JAX_PLATFORMS; \
  export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VIDEOX_ATTENTION_TYPE=SDPA \
  H3_RUN=$RUN H3_VIDEOX_FUN=$VXF H3_MODELS=$MODELS; \
  cd $RUN; exec $PY $HERE/h3_control_worker.py --queue $RUN/queue --profile $PROFILE \
  --memory-mode $MODE --idle-exit $IDLE $MAXT $TURBO"
if [ -n "$LOCAL" ]; then
  case "$LOCAL" in *[!0-9]*) echo "H3_LOCAL_GPU must be one GPU index, got '$LOCAL'" >&2; exit 2;; esac
  [ $# -eq 0 ] || { echo "local mode takes no scheduler arguments: $*" >&2; exit 2; }
  exec env CUDA_VISIBLE_DEVICES=$LOCAL bash -lc "$WORKER"
fi
# full residency peaks at 165.5 GiB reserved (169.5k MiB) per GPU on a 192-frame chunk, so a GPU with only 165000 MiB
# free runs out of memory; require 172000 MiB by default.
MINFREE=${H3_MIN_FREE_MB:-172000}
# every H3 lease is pinned to the hosts the deployment/user allows for this run (e.g. a node cap)
[ -s $RUN/queue/ALLOWED_HOSTS ] || { echo "missing $RUN/queue/ALLOWED_HOSTS (one line of allowed scheduler hosts)" >&2; exit 2; }
HOSTS=$(cat $RUN/queue/ALLOWED_HOSTS)
HOSTARGS=""; for h in $HOSTS; do HOSTARGS="$HOSTARGS --host $h"; done
for a in "$@"; do case "$a" in --host) echo "refusing extra --host: H3 is pinned to: $HOSTS" >&2; exit 2;; esac; done
exec "$SCHED" submit $HOSTARGS --gpus 1 --min-free-mb $MINFREE --max-util 10 \
  --name h3-$PROFILE "$@" -- bash -lc "$WORKER"
