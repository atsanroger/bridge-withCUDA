#!/usr/bin/env bash
# Timeline profile: per-kernel time share of the whole run.
# Usage: prof_nsys.sh RUNDIR      (output: RUNDIR/nsys.nsys-rep, RUNDIR/kern_cuda_gpu_kern_sum.csv)
# Suggested: BRIDGE_NSOLVE=1 (one full CGNR solve); BRIDGE_PERF_NREP default 200.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
run=$(readlink -f "$1")
"$here/run.sh" "$run" nsys profile -f true -t cuda,nvtx -o "$run/nsys" 2>&1 | tee "$run/out_nsys.log"
nsys stats -r cuda_gpu_kern_sum -f csv -o "$run/kern" "$run/nsys.nsys-rep" > /dev/null
python3 -I "$here/kern_table.py" "$run/kern_cuda_gpu_kern_sum.csv"
