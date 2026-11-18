#!/usr/bin/env bash
# Hardware counters per kernel: time, DRAM bytes, FP64/FP32 instruction counts,
# FP64 pipe utilization, registers, occupancy.  Needs native Linux (not WSL2)
# and counter access (NVreg_RestrictProfilingToAdminUsers=0 or root).
# Usage: prof_ncu.sh RUNDIR [KERNEL_REGEX] [LAUNCH_SKIP] [LAUNCH_COUNT]
#   defaults: regex below, skip 0, count 400.
# Suggested: BRIDGE_PERF_NREP=2 BRIDGE_NSOLVE=1 and a small MAXITER in make_run.sh.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
run=$(readlink -f "$1")
re=${2:-'hopb|5dir|LUinv|LUdaginv|axpy|aypx|copy|dot|norm|scal|normalize'}
skip=${3:-0} count=${4:-400}
metrics=gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,\
lts__t_bytes.sum,\
smsp__sass_thread_inst_executed_op_dadd_pred_on.sum,\
smsp__sass_thread_inst_executed_op_dmul_pred_on.sum,\
smsp__sass_thread_inst_executed_op_dfma_pred_on.sum,\
smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,\
smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,\
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,\
sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active,\
sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
launch__registers_per_thread,launch__occupancy_limit_registers,\
sm__warps_active.avg.pct_of_peak_sustained_active
"$here/run.sh" "$run" ncu --target-processes all -f -o "$run/ncu" \
  --kernel-name-base demangled -k "regex:$re" --launch-skip "$skip" --launch-count "$count" \
  --metrics "$metrics" 2>&1 | tee "$run/out_ncu.log"
ncu -i "$run/ncu.ncu-rep" --csv --page raw > "$run/ncu_raw.csv"
python3 -I "$here/ncu_table.py" "$run/ncu_raw.csv"
