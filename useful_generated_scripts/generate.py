import os

Ns = [128, 256]
threads_list = [16, 32]
num_tries = 5
base_filename = "job"
L = 'PI'

os.makedirs("jobs", exist_ok=True)

i = 0
submit_lines = []

for N in Ns:
    for threads in threads_list:
        if threads == 16:
            n_procs = 3
        elif threads == 32:
            n_procs = 5
        else:
            n_procs = 1
        for try_num in range(num_tries):
            filename = f"jobs/job_{i}.lsf"
            content = f"""#BSUB -n {n_procs}
#BSUB -W 00:15
#BSUB -o out_{N}_1_{threads}_try_{try_num}.txt
#BSUB -e err_{N}_1_{threads}_try_{try_num}.txt
#BSUB -R "span[hosts=1]"
OMP_NUM_THREADS={threads} ./openmp_solve {N} {L}
"""
            with open(filename, "w") as f:
                f.write(content)
            i += 1
            submit_lines.append(f"bsub < {filename}")
            submit_lines.append(f"sleep 5")

with open("submit_all.sh", "w") as f:
    for line in submit_lines:
        f.write(line + "\n")


