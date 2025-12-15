import os
import sys
from datetime import datetime

##################################
RUN_OPENMP_INSTEAD_OF_MPI = False
RUN_SEQUENTIAL = True

RUN_1_THREADS_IN_MPI = True
RUN_4_THREADS_IN_MPI = True
L = '1'

# ADD/REMOVE 2 FROM MPI processes if running with/without openmp
##################################

X_RUNS = 7
SERIAL_NAME = "serial_solve"
OPENMP_NAME = "omp_solve"
MPI_NAME = "mpi_solve"

if RUN_OPENMP_INSTEAD_OF_MPI:
    parallel_name = OPENMP_NAME
    processes = [1]
    threads = [1, 2, 4, 8]
    threads = list(reversed(threads))
    if not RUN_SEQUENTIAL:
        threads.remove(1)
else:
    # parallel_name = MPI_NAME
    parallel_name = 'mpi_plus_openmp_solve'
    # processes = [1, 2, 4, 8, 16, 32]
    processes = [4, 8]
    processes = list(reversed(processes))
    # threads = [1]
    threads = [1, 2, 4, 8]
    threads = list(reversed(threads))
    if not RUN_1_THREADS_IN_MPI:
        threads.remove(1)
    if not RUN_4_THREADS_IN_MPI:
        threads.remove(4)
    if not RUN_SEQUENTIAL:
        processes.remove(1)


current_time_str = datetime.now().strftime("%H_%M_%S")

with open("compile.sh", "w") as f:
    sys.stdout = f
    print("module load SpectrumMPI")
    print(f"g++ -fopenmp -std=c++11 {OPENMP_NAME}.cpp -O2 -o {OPENMP_NAME}")
    print(f"mpic++ -fopenmp -std=c++11 {MPI_NAME}.cpp -O2 -o {MPI_NAME}")
    print(f"g++ -std=c++11 {SERIAL_NAME}.cpp -O2 -o {SERIAL_NAME}")
    print()

with open("submit.sh", "w") as f:
    sys.stdout = f
    for processes_number in processes:
        for threads_number in threads:
            for n in (128, 256):
                for x in range(X_RUNS):
                    if not ((n == 128 and processes_number == 4) or (n == 256 and processes_number == 8)):
                        continue
                    is_parallel = threads_number == 1 and processes_number == 1
                    executable = SERIAL_NAME if is_parallel else parallel_name
                    stdout_file = f"out_{x}_{n}_{processes_number}_{threads_number}.out"
                    stderr_file = f"err_{x}_{n}_{processes_number}_{threads_number}.err"
                    print(
                        f"mpisubmit.pl -p {processes_number} -t {threads_number} "
                        f"--stdout {stdout_file} --stderr {stderr_file} "
                        "-w 00:30 "
                        f"{executable} -- {n} {L}"
                    )
                    print("sleep 5")
sys.stdout = sys.__stdout__
print("GENERATED.")
