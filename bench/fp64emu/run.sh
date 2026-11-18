#!/usr/bin/env bash
# Run the benchmark in RUNDIR (made by make_run.sh). Single GPU, single process.
# Usage: run.sh RUNDIR [PROFILER ARGS...]
#   e.g. run.sh r24_fp                      plain run
#        run.sh r24_fp nsys profile ...     run under a profiler (see prof_*.sh)
# Environment passed to the binary (defaults in parentheses):
#   BRIDGE_PERF_NREP (200)  BRIDGE_NSOLVE (12)  BRIDGE_SKIP_CHECK (unset)
#   BRIDGE_ONLY_5DIN (set here to 1)
#   CUDA_VISIBLE_DEVICES selects the GPU.  BIN overrides the binary path.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
bin=${BIN:-$here/../../build/cuda-ampere-mpi/bridge_test_alt_spectrum}
run=$1; shift
[ -x "$bin" ] || { echo "binary not found: $bin"; exit 1; }
cd "$run"
export BRIDGE_ONLY_5DIN=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
echo "binary: $bin"
echo "BRIDGE_PERF_NREP=${BRIDGE_PERF_NREP:-200} BRIDGE_NSOLVE=${BRIDGE_NSOLVE:-12} BRIDGE_SKIP_CHECK=${BRIDGE_SKIP_CHECK:-unset} GPU=${CUDA_VISIBLE_DEVICES:-all}"
nvidia-smi --query-gpu=name,clocks.max.sm,clocks.max.mem --format=csv,noheader || true
mpirun -np 1 "$@" "$bin"
