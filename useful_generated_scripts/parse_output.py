from collections import defaultdict
import os
import math
import re
import sys

path = "outputs_openmp/"
print('path: ', path)
print()
files = [f for f in os.listdir(path) if f.endswith(".out") and "out_" in f]
r = defaultdict(list)

for filename in files:
    with open(os.path.join(path, filename), "r") as file:
        content = file.readlines()
        parts = filename.replace(".out", "").split("_")
        n_value = int(parts[2])
        processes = int(parts[3])
        threads = int(parts[4])
        max_eps = None
        time_elapsed = None
        for line in content:
            if line.startswith("Final error:"):
                max_eps = float(line.split()[-1])
            elif line.startswith("Elapsed time:"):
                time_elapsed = float(line.split()[-1])
        if time_elapsed is not None and max_eps is not None:
            r[(n_value, processes, threads)].append((max_eps, time_elapsed))
print(f"{'N':>6} {'proc':>5} {'thr':>5} | {'mean_eps':>12} {'mean_time':>12} {'var_time':>12}")
print("-" * 60)

for key, values in sorted(r.items()):
    n_value, processes, threads = key
    times = [v[1] for v in values]
    epses = [v[0] for v in values]
    mean_time = sum(times) / len(times)
    mean_eps = sum(epses) / len(epses)
    var_time = sum((t - mean_time) ** 2 for t in times) / len(times)
    print(f"{n_value:6d} {processes:5d} {threads:5d} | {mean_eps:12.6e} {mean_time:12.6f} {var_time:12.6f}")
