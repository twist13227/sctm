#!/bin/bash
OUT_DIR="mpi_cuda_tests_final"
mkdir -p "$OUT_DIR"

L=1
WALLTIME="00:30"
run_experiments() {
    local N=$1
    mpisubmit.pl -p 1 -t 1 \
        --stdout "$OUT_DIR/serial_${N}_${L}.out" \
        --stderr "$OUT_DIR/serial_${N}_${L}.err" \
        -w $WALLTIME \
        serial_solve -- $N $L

    mpisubmit.pl -p 20 -t 1 \
        --stdout "$OUT_DIR/mpi_20_${N}_${L}.out" \
        --stderr "$OUT_DIR/mpi_20_${N}_${L}.err" \
	-w $WALLTIME \
        mpi_solve -- $N $L

    mpisubmit.pl -p 20 -t 8 \
        --stdout "$OUT_DIR/mpi_omp_20x8_${N}_${L}.out" \
        --stderr "$OUT_DIR/mpi_omp_20x8_${N}_${L}.err" \
	-w $WALLTIME \
        mpi_omp_solve -- $N $L

    bsub -n 1 \
        -o "$OUT_DIR/cuda_1gpu_${N}_${L}.out" \
        -e "$OUT_DIR/cuda_1gpu_${N}_${L}.err" \
        -R "span[hosts=1]" \
        -gpu "num=1:mode=shared" \
        -W $WALLTIME \
        "mpiexec -n 1 ./mpi_cuda_solve $N $L"

    bsub -n 2 \
        -o "$OUT_DIR/cuda_2gpu_${N}_${L}.out" \
        -e "$OUT_DIR/cuda_2gpu_${N}_${L}.err" \
        -R "span[hosts=1]" \
        -gpu "num=2:mode=shared" \
        -W $WALLTIME \
        "mpiexec -n 2 ./mpi_cuda_solve $N $L"
}

run_experiments 256
run_experiments 512
run_experiments 768
