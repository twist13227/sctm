import os
import sys
from datetime import datetime

if len(sys.argv) != 5:
    print(f"Usage: python3 {sys.argv[0]} <mode> <parallel_name> <serial_name> <X_RUNS>")
    sys.exit(1)

mode = sys.argv[1]
parallel_name = sys.argv[2]
serial_name = sys.argv[3]
X_RUNS = int(sys.argv[4])

Ns = [128, 256]
Ls = ['1', 'PI']

if mode == 'openmp':
    processes = [1]
    threads = [1, 2, 4, 8, 16, 32]
    threads = list(reversed(threads))
elif mode == 'mpi':
    processes = [1, 2, 4, 8, 16, 32]
    processes = list(reversed(processes))
    threads = [1]
elif mode == 'mpi_plus_openmp':
    processes = [4, 8]
    processes = list(reversed(processes))
    threads = [1, 2, 4, 8]
    threads = list(reversed(threads))
else:
    raise ValueError(f"Invalid mode: {mode}")

with open("compile.sh", "w") as f:
    sys.stdout = f
    print("module load SpectrumMPI")
    if mode == 'openmp':
        print(f"g++ -fopenmp -std=c++11 {parallel_name}.cpp -O2 -o {parallel_name}")
    elif mode == 'mpi':
        print(f"mpic++ -std=c++11 {parallel_name}.cpp -O2 -o {parallel_name}")
    elif mode == 'mpi_plus_openmp':
        print(f"OMPI_CXX=g++ mpicxx -I/opt/ibm/spectrum_mpi/include -I/opt/ibm/spectrum_mpi/include -L/opt/ibm/spectrum_mpi/lib -lmpiprofilesupport -lmpi_ibm -O3 -Wall -Wextra -std=c++11 -fopenmp {parallel_name}.cpp -o {parallel_name}")
        # print(f"mpic++ -fopenmp -std=c++11 {parallel_name}.cpp -O2 -o {parallel_name}")
    else:
        raise ValueError(f"Invalid mode: {mode}")
    print(f"g++ -std=c++11 {serial_name}.cpp -O2 -o {serial_name}")
    print()

with open(mode + "_submit.sh", "w") as f:
    sys.stdout = f
    if mode == 'openmp':
        os.makedirs("jobs_openmp", exist_ok=True)
    for processes_number in processes:
        for threads_number in threads:
            for n in Ns:
                for L in Ls:
                    for x in range(X_RUNS):
                        if mode == 'mpi_plus_openmp' and not ((n == 128 and processes_number == 4) or (n == 256 and processes_number == 8)):
                            continue
                        if mode == 'openmp' and threads_number > 8:
                            n_procs = (threads_number // 8) + 1
                            filename = f"jobs_openmp/try_{mode}_{processes_number}_{threads_number}_{n}_{L}_{x}.lsf"
                            content = f"""#BSUB -n {n_procs}
#BSUB -W 00:15
#BSUB -o try_{mode}_{processes_number}_{threads_number}_{n}_{L}_{x}.out
#BSUB -e try_{mode}_{processes_number}_{threads_number}_{n}_{L}_{x}.err
#BSUB -R "span[hosts=1]"
OMP_NUM_THREADS={threads_number} ./{parallel_name} {n} {L}
"""
                            with open(filename, "w") as f:
                                f.write(content)
                            print(f"bsub < {filename}")
                            print("sleep 5")
                        else:
                            is_parallel = threads_number == 1 and processes_number == 1
                            executable = serial_name if is_parallel else parallel_name
                            stdout_file = f"try_{mode}_{processes_number}_{threads_number}_{n}_{L}_{x}.out"
                            stderr_file = f"try_{mode}_{processes_number}_{threads_number}_{n}_{L}_{x}.err"
                            print(
                                f"mpisubmit.pl -p {processes_number} -t {threads_number} "
                                f"--stdout {stdout_file} --stderr {stderr_file} "
                                "-w 00:30 "
                                f"{executable} -- {n} {L}"
                            )
                            print("sleep 5")
sys.stdout = sys.__stdout__
print("GENERATED.")
