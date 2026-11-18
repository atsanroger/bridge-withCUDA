"""Print the nsys cuda_gpu_kern_sum CSV as a short table (top 25 kernels by time)."""
import csv
import sys

rows = list(csv.reader(open(sys.argv[1])))
h = rows[0]
c = {k: h.index(k) for k in ("Time (%)", "Total Time (ns)", "Instances", "Avg (ns)", "Name")}
print(f"{'time%':>6} {'total ms':>9} {'calls':>6} {'avg us':>8}  kernel")
for r in rows[1:26]:
    print(f"{float(r[c['Time (%)']]):6.1f} {float(r[c['Total Time (ns)']])/1e6:9.2f} "
          f"{r[c['Instances']]:>6} {float(r[c['Avg (ns)']])/1e3:8.1f}  {r[c['Name']][:90]}")
