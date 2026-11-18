#!/usr/bin/env bash
# Create a run directory for the eo-DWF CGNR benchmark.
# Usage: make_run.sh RUNDIR MODE LATTICE LS CONF
#   MODE    Solver.multiword_mode:
#             FP         native double
#             lowendgpu  double storage, float-float arithmetic in hopb only
#             DW         float-float everywhere
#             TW         float-triple everywhere
#   LATTICE e.g. 24x24x24x24 or 24x24x24x48
#   CONF    Bridge++ config: *.txt = Text, otherwise Binary (big-endian double, no header)
# Optional environment: MQ (0.10) M0 (1.0) TOL (1.0e-24, on |r|^2/|b|^2)
#                       MAXITER (100) NRESTART (40): total iterations = MAXITER*NRESTART
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
[ $# -eq 5 ] || { sed -n 2,12p "$0"; exit 1; }
run=$1 mode=$2 lat=$3 ls=$4 conf=$(readlink -f "$5")
[ -f "$conf" ] || { echo "config not found: $5"; exit 1; }
IFS=x read -r lx ly lz lt <<< "$lat"
case $mode in
  FP|lowendgpu|DW|TW) ;;
  *) echo "MODE must be FP, lowendgpu, DW or TW"; exit 1 ;;
esac
nreal=$((lx*ly*lz*lt*4*18))
if [[ $conf == *.txt ]]; then           # Bridge++ Text: one real per line
  ctype=Text; size=$(wc -l < "$conf"); expect=$nreal
else                                    # Bridge++ Binary: big-endian double
  ctype=Binary; size=$(stat -c %s "$conf"); expect=$((nreal*8))
fi
[ "$size" -eq "$expect" ] || { echo "config $ctype size $size != $expect for $lat"; exit 1; }
mkdir -p "$run"
sed -e "s/@LX@/$lx/; s/@LY@/$ly/; s/@LZ@/$lz/; s/@LT@/$lt/" "$here/main.yaml.in" > "$run/main.yaml"
sed -e "s|@MW@|$mode|; s|@CONF@|$conf|; s|@CTYPE@|$ctype|" \
    -e "s|@MQ@|${MQ:-0.10}|g; s|@M0@|${M0:-1.0}|g; s|@LS@|$ls|g" \
    -e "s|@TOL@|${TOL:-1.0e-24}|; s|@MAXITER@|${MAXITER:-100}|; s|@NRESTART@|${NRESTART:-40}|" \
    "$here/spectrum_5din.yaml.in" > "$run/test_alt_Spectrum_Domainwall_5din_Hadron2ptFunction.yaml"
echo "run dir  : $run"
echo "lattice  : $lx x $ly x $lz x $lt, Ls = $ls"
echo "multiword: $mode"
echo "mq, M0   : ${MQ:-0.10}, ${M0:-1.0}   (b, c) = (1.5, 0.5)"
echo "solver   : CGNR, tol |r|^2/|b|^2 = ${TOL:-1.0e-24}, max iter = ${MAXITER:-100} x ${NRESTART:-40}"
echo "config   : $conf ($ctype)"
