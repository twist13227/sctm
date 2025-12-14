# Воспроизведение результатов MPI+CUDA

## Сборка

```bash
module load SpectrumMPI
nvcc -O3 -arch=sm_35 -ccbin mpic++ -o mpi_cuda_solve mpi_cuda_solve.cu -x cu -Xcompiler -Wall -std=c++11
```

## Необходимые бинари (есть в репозитории в соответствующих папках)

- `serial_solve`
- `mpi_solve`
- `mpi_omp_solve`
- `mpi_cuda_solve`

## Запуск экспериментов

```bash
chmod +x run_all_experiments.sh
./run_all_experiments.sh
```

## Результаты

Результаты сохраняются в папку`mpi_cuda_tests_final/`:
- `serial_N_1.out`
- `mpi_20_N_1.out`
- `mpi_omp_20x8_N_1.out`
- `cuda_1gpu_N_1.out`
- `cuda_2gpu_N_1.out`
для N = 256, 512, 768.
Пример полученных результатов приведён в `mpi_cuda_jobs_output.zip`.
