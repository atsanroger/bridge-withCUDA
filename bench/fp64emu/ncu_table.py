"""Summarize `ncu --csv --page raw` per kernel name.

FP64 flop = dadd + dmul + 2*dfma; FP32 flop likewise. I64 = FP64 flop / DRAM bytes.
Values are averaged over the profiled launches of each kernel.
"""
import csv
import sys
from collections import defaultdict

rows = list(csv.reader(open(sys.argv[1])))
h, units, data = rows[0], rows[1], rows[2:]
col = {k: i for i, k in enumerate(h)}

def f(r, k):
    v = r[col[k]].replace(",", "") if k in col else ""
    try:
        return float(v)
    except ValueError:
        return 0.0

def scale(k):
    u = units[col[k]] if k in col else ""
    return {"nsecond": 1e-9, "usecond": 1e-6, "msecond": 1e-3, "second": 1.0,
            "ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0, "Tbyte": 1e12,
            "byte": 1.0, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9}.get(u, 1.0)

agg = defaultdict(lambda: defaultdict(float))
for r in data:
    name = r[col["Kernel Name"]].split("(")[0].replace("BridgeACC::", "")[:60]
    a = agg[name]
    a["n"] += 1
    a["t"] += f(r, "gpu__time_duration.sum") * scale("gpu__time_duration.sum")
    a["B"] += (f(r, "dram__bytes_read.sum") * scale("dram__bytes_read.sum")
               + f(r, "dram__bytes_write.sum") * scale("dram__bytes_write.sum"))
    p = "smsp__sass_thread_inst_executed_op_"
    a["f64"] += f(r, p + "dadd_pred_on.sum") + f(r, p + "dmul_pred_on.sum") + 2 * f(r, p + "dfma_pred_on.sum")
    a["f32"] += f(r, p + "fadd_pred_on.sum") + f(r, p + "fmul_pred_on.sum") + 2 * f(r, p + "ffma_pred_on.sum")
    a["p64"] += f(r, "sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active")
    a["dram"] += f(r, "dram__throughput.avg.pct_of_peak_sustained_elapsed")
    a["occ"] += f(r, "sm__warps_active.avg.pct_of_peak_sustained_active")
    a["reg"] = f(r, "launch__registers_per_thread")

print(f"{'kernel':60} {'n':>4} {'us':>8} {'MB':>8} {'GF64':>7} {'GF32':>7} {'I64':>6} {'GB/s':>6} {'fp64%':>6} {'dram%':>6} {'occ%':>5} {'reg':>4}")
for name, a in sorted(agg.items(), key=lambda kv: -kv[1]["t"]):
    n = a["n"]
    t, B = a["t"] / n, a["B"] / n
    f64, f32 = a["f64"] / n, a["f32"] / n
    print(f"{name:60} {n:4.0f} {t*1e6:8.1f} {B/1e6:8.2f} {f64/t/1e9 if t else 0:7.1f} {f32/t/1e9 if t else 0:7.1f} "
          f"{f64/B if B else 0:6.2f} {B/t/1e9 if t else 0:6.0f} {a['p64']/n:6.1f} {a['dram']/n:6.1f} {a['occ']/n:5.1f} {a['reg']:4.0f}")
