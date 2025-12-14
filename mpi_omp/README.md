# Воспроизведение результатов MPI+OpenMP

## Сборка

```bash
module load SpectrumMPI
OMPI_CXX=g++ mpicxx -I/opt/ibm/spectrum_mpi/include -L/opt/ibm/spectrum_mpi/lib -lmpiprofilesupport -lmpi_ibm -O3 -Wall -Wextra -std=c++11 -fopenmp mpi_omp_solve.cpp -o mpi_omp_solve
```

## Запуск экспериментов

```bash
chmod +x mpi_omp_submit.sh
./mpi_omp_submit.sh
```

## Результаты

Результаты сохраняются в виде файлов:

```
try_mpi_omp_<mpi_procs>_<omp_threads>_<N>_<L>_<run_num>.out
```

где:
- `<mpi_procs>` — число MPI-процессов (4 или 8),
- `<omp_threads>` — число OpenMP-потоков на каждый MPI-процесс (1, 2, 4, 8),
- `<N>` — размер задачи (128 или 256),
- `<L>` — значение параметра L (1 или Pi),
- `<run_num>` — номер запуска.
Все результаты сохраняются в той же директории, откуда запускается `mpi_omp_submit.sh`.
Пример полученных результатов приведён в `mpi_omp_jobs_output.zip`.
