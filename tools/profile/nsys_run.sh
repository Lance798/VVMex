#!/usr/bin/env bash
# Timeline-profile one VVMex run.   usage: tools/profile/nsys_run.sh <config.json> [extra nsys flags]
#
# Runs ./build/vvm directly rather than through submit.py: submit.py derives
# VVM_BINARY from the CMake preset and overwrites whatever you export
# (submit.py:203), so there is no hook to hang a profiler on, and a single-rank
# profiling run does not need the MPI launcher anyway. For a multi-rank profile,
# make that line respect an existing VVM_BINARY and point it at a wrapper.
set -euo pipefail
cd "${VVM_ROOT:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}"

CFG="${1:?usage: nsys_run.sh <config.json> [extra nsys flags]}"; shift || true
OUT="${VVM_NSYS_OUT:-/tmp/vvm_prof}"; mkdir -p "$OUT"
NAME="$(basename "${CFG%.json}")"

# --sample/--cpuctxsw off: this kernel runs at perf_event_paranoid=4, so CPU
#   sampling cannot arm; leaving it on only produces warnings. GPU tracing is
#   unaffected (it does not go through perf_event).
# --cuda-graph-trace=node: optimization.cuda_graph_halo_exchange captures halo
#   exchanges into CUDA graphs. Without this the whole graph is one opaque range
#   on the timeline instead of the kernels inside it.
exec nsys profile \
    --output "$OUT/$NAME" --force-overwrite true \
    --trace=cuda,nvtx,mpi \
    --sample=none --cpuctxsw=none \
    --cuda-graph-trace=node \
    "$@" \
    ./build/vvm "$CFG"
