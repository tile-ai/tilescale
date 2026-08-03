#!/bin/bash
# Launch an inter-node collective example across two nodes, one GPU each.
#
# Usage: run_internode.sh <rank0_node> <rank0_gpu> <rank1_node> <rank1_gpu> <script> [extra args...]
#   e.g. run_internode.sh node076 0 node074 4 example_internode_allgather.py --numel 4194304
#
# Run this from node071, which is the only node reachable directly from a laptop;
# it ssh's to both target nodes. Rank 0 also hosts the rendezvous, so its internal
# 10.27.28.x address is what both ranks must use -- the externally routable name
# is not on the RoCE fabric.

set -eo pipefail

if [[ $# -lt 5 ]]; then
  sed -n '2,12p' "$0"
  exit 1
fi

R0_NODE=$1; R0_GPU=$2; R1_NODE=$3; R1_GPU=$4; SCRIPT=$5
shift 5
EXTRA=("$@")

WORKSPACE=/tilert/wt/tilescale_workspace/tilescale
EXDIR=${WORKSPACE}/examples/distributed/internode
MASTER_PORT=${MASTER_PORT:-29576}

# Only node071 has the original conda env; the other nodes use a copy, so the
# NCCL path differs per node and has to be probed rather than assumed.
pick_python() {
  if [[ $1 == node071 ]]; then
    echo /root/tilert/mlx/miniconda3/envs/wt-tl/bin/python
  else
    echo /tilert/wt/tilescale_workspace/tools/wt-tl-env/bin/python
  fi
}
pick_nccl_dir() {
  local py; py=$(pick_python "$1")
  echo "${py%/bin/python}/lib/python3.12/site-packages/nvidia/nccl"
}

# The -internal names are not in DNS or /etc/hosts on every node, so the RoCE
# addresses are listed here and getent is only a fallback for nodes not listed.
internal_ip() {
  case $1 in
    node071) echo 10.27.28.73 ;;
    node073) echo 10.27.28.105 ;;
    node074) echo 10.27.28.113 ;;
    node076) echo 10.27.28.115 ;;
    *) getent hosts "$1-internal" | awk '{print $1}' ;;
  esac
}

MASTER_ADDR=$(internal_ip "${R0_NODE}")
if [[ -z ${MASTER_ADDR} ]]; then
  echo "cannot resolve ${R0_NODE}-internal" >&2
  exit 1
fi

# NOTE: this script must run ON node071 (see header). From node071 a plain
# `ssh node074 "<cmd>"` works. It fails only when invoked from a laptop whose
# ssh config sets RemoteCommand for node073/node074 ("Cannot execute
# command-line and remote command", exit 255) -- that is a client-config
# artifact, not something to work around here.
# Forward any NCCL_*/TILESCALE_*/TILELANG_* set in this shell to both ranks, so
# knobs like NCCL_GIN_NCONTEXTS, NCCL_GIN_TYPE or NCCL_IB_QPS_PER_CONNECTION can
# be swept without editing this script. The vars set explicitly in launch() are
# skipped here and appear after this block, so they always win.
forwarded_env() {
  local out="" v
  for v in $(env | sed -n 's/^\(\(NCCL\|TILESCALE\|TILELANG\|TL\)_[A-Za-z0-9_]*\)=.*/\1/p' | sort -u); do
    case ${v} in
      NCCL_IB_DISABLE|TILESCALE_USE_VMM|TILESCALE_USE_GIN|TILESCALE_NCCL_LIB|TILELANG_CACHE_DIR|NCCL_DEBUG)
        continue ;;
    esac
    out+="${v}=${!v} "
  done
  printf '%s' "${out}"
}

launch() {  # node gpu rank
  local node=$1 gpu=$2 rank=$3
  local py nccl_dir fwd
  py=$(pick_python "${node}")
  nccl_dir=$(pick_nccl_dir "${node}")
  fwd=$(forwarded_env)
  # NCCL_IB_DISABLE=0 is the point of the exercise: init_dist would otherwise
  # default it to 1 and confine the "inter-node" transfer to shared memory.
  ssh "${node}" "cd ${EXDIR} && \
    ${fwd} \
    CUDA_VISIBLE_DEVICES=${gpu} \
    MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} \
    WORLD_SIZE=2 LOCAL_WORLD_SIZE=1 NNODES=2 \
    RANK=${rank} NODE_RANK=${rank} LOCAL_RANK=0 \
    NCCL_IB_DISABLE=0 TILESCALE_USE_VMM=1 TILESCALE_USE_GIN=1 \
    PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 \
    ${TL_STAGE_TRACE:+TL_STAGE_TRACE=${TL_STAGE_TRACE}} \
    ${NCCL_DEBUG:+NCCL_DEBUG=${NCCL_DEBUG}} \
    TORCH_NCCL_BLOCKING_WAIT=${TORCH_NCCL_BLOCKING_WAIT:-1} \
    TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
    TL_PG_TIMEOUT_SEC=${TL_PG_TIMEOUT_SEC:-180} \
    ${TILELANG_CACHE_DIR:+TILELANG_CACHE_DIR=${TILELANG_CACHE_DIR}} \
    TILESCALE_NCCL_LIB=${nccl_dir}/lib/libnccl.so.2 \
    LD_LIBRARY_PATH=${nccl_dir}/lib:\${LD_LIBRARY_PATH} \
    PYTHONPATH=${WORKSPACE}:${EXDIR} \
    timeout ${RANK_TIMEOUT:-900} ${py} ${SCRIPT} ${EXTRA[*]}"
}

launch "${R1_NODE}" "${R1_GPU}" 1 > /tmp/internode_rank1.log 2>&1 &
R1_PID=$!
sleep 2
launch "${R0_NODE}" "${R0_GPU}" 0 > /tmp/internode_rank0.log 2>&1 &
R0_PID=$!

wait ${R0_PID}; R0_EXIT=$?
wait ${R1_PID}; R1_EXIT=$?

echo "########## RANK0 (${R0_NODE}:${R0_GPU}) ##########"
cat /tmp/internode_rank0.log
echo "########## RANK1 (${R1_NODE}:${R1_GPU}) ##########"
cat /tmp/internode_rank1.log
echo "EXIT rank0=${R0_EXIT} rank1=${R1_EXIT}"
exit $(( R0_EXIT | R1_EXIT ))
