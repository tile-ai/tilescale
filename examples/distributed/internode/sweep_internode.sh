#!/bin/bash
# Sweep the inter-node GIN collectives over chunk count and GIN context policy.
#
# Usage: sweep_internode.sh <r0_node> <r0_gpu> <r1_node> <r1_gpu>
#   e.g. sweep_internode.sh <node> 0 <node> 5
#
# Env knobs:
#   NUMEL       elements in the gathered/full buffer (default 8388608)
#   CHUNKS      space-separated chunk counts to try (default "1 2 4 8 16 32")
#   CTXS        space-separated -DTL_GIN_CONTEXTS values (default "1 0")
#   SCRIPTS     which examples to run (default all three)
#   CACHE       cache dir for the whole sweep (default a fresh /tmp path)
#
# Every (chunks, ctx) pair changes either the prim_func or compile_flags, so each
# gets its own kernel cache entry. The single fresh CACHE on top of that is what
# guards against the real hazard: the cache key does not cover the device
# templates, so a stale entry from before a header edit would be reused silently.
#
# Runs are sequential on purpose -- two runs at once would contend for the same
# GPUs and NICs and make every number meaningless.

set -uo pipefail

if [[ $# -lt 4 ]]; then
  sed -n '2,20p' "$0"
  exit 1
fi

R0_NODE=$1; R0_GPU=$2; R1_NODE=$3; R1_GPU=$4
EXDIR=<workspace>

NUMEL=${NUMEL:-8388608}
CHUNKS=${CHUNKS:-"1 2 4 8 16 32"}
CTXS=${CTXS:-"1 0"}
SCRIPTS=${SCRIPTS:-"example_internode_allgather.py example_internode_allreduce.py example_internode_reduce_scatter.py"}
CACHE=${CACHE:-/tmp/tlsweep_$$}
PORT=${PORT:-29800}

# Leftover ranks from a killed run hold the GPU in Exclusive_Process mode and
# make the next launch fail with cudaErrorDevicesUnavailable. Kill by explicit
# PID: `pkill -f <pattern>` over ssh also matches the wrapper shell running it,
# which kills the connection instead of the rank.
cleanup() {
  for spec in "${R0_NODE}:${R0_GPU}" "${R1_NODE}:${R1_GPU}"; do
    local node=${spec%%:*}
    # The bracket keeps the pattern from matching the remote shell that carries
    # this very command line -- otherwise pgrep finds it and the kill takes out
    # the ssh session instead of the rank.
    ssh -o BatchMode=yes -o StrictHostKeyChecking=no "${node}" '
      for p in $(pgrep -f "[b]in/python example_internode"); do kill -9 "$p" 2>/dev/null; done' \
      >/dev/null 2>&1 || true
  done
  sleep 3
}

printf '%-22s %7s %5s %5s %11s %11s %9s %s\n' \
  COLLECTIVE CHUNKS CTX PUT_KB TILESCALE TORCH SPEEDUP STATUS

for script in ${SCRIPTS}; do
  name=${script#example_internode_}; name=${name%.py}
  for ctx in ${CTXS}; do
    for chunks in ${CHUNKS}; do
      PORT=$((PORT + 2))
      cleanup
      out=$(cd "${EXDIR}" && MASTER_PORT=${PORT} RANK_TIMEOUT=${RANK_TIMEOUT:-180} \
        TILELANG_CACHE_DIR=${CACHE} \
        timeout "${WRAP_TIMEOUT:-240}" bash run_internode.sh \
        "${R0_NODE}" "${R0_GPU}" "${R1_NODE}" "${R1_GPU}" "${script}" \
        --numel "${NUMEL}" --chunks "${chunks}" --gin-contexts "${ctx}" \
        --threads "${THREADS:-1024}" \
        --warmup "${WARMUP:-20}" --rep "${REP:-50}" 2>&1)

      line=$(echo "${out}" | grep -E 'tilescale .*GB/s' | tail -1)
      putkb=$(echo "${out}" | grep -oE 'put=[0-9]+KiB' | head -1 | tr -d 'put=KiB')
      if echo "${out}" | grep -q '^PASS'; then status=PASS
      elif echo "${out}" | grep -q 'MISMATCH\|^FAIL'; then status=FAIL
      else status=HANG_OR_ERR; fi

      if [[ -n ${line} ]]; then
        tl=$(echo "${line}"  | sed -E 's/.*tilescale +([0-9.]+) ms +([0-9.]+) GB.*/\1ms|\2GB\/s/')
        th=$(echo "${line}"  | sed -E 's/.*torch +([0-9.]+) ms +([0-9.]+) GB.*/\1ms|\2GB\/s/')
        sp=$(echo "${line}"  | grep -oE 'speedup +[0-9.]+x' | awk '{print $2}')
      else
        tl="-"; th="-"; sp="-"
      fi
      printf '%-22s %7s %5s %5s %11s %11s %9s %s\n' \
        "${name}" "${chunks}" "${ctx}" "${putkb:--}" "${tl}" "${th}" "${sp:--}" "${status}"
    done
  done
done

cleanup
echo "cache: ${CACHE}"
