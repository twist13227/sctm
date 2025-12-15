import re
import sys

def main():
    if len(sys.argv) != 2:
        print("Usage: python full_analyze.py <output_file.out>")
        return

    filename = sys.argv[1]
    with open(filename, 'r') as f:
        content = f.read()

    N = int(re.search(r'N = (\d+)', content).group(1))
    L = float(re.search(r'L = ([\d.]+)', content).group(1))
    processes = int(re.search(r'processes = (\d+)', content).group(1))
    ni, nj, nk = map(int, re.search(r'Local indices: (\d+) x (\d+) x (\d+)', content).groups())
    total_time = float(re.search(r'Total time: ([\d.]+) s', content).group(1))
    init_ex = float(re.search(r'Initial exchange: ([\d.]+) ms', content).group(1))
    init_step = float(re.search(r'Initial step computation: ([\d.]+) ms', content).group(1))

    steps = []
    step_blocks = re.findall(r'Step (\d+)[\s\S]*?total time: ([\d.eE+-]+) ms', content)
    for step_num, total in step_blocks:
        step = {'step': int(step_num), 'total_step': float(total)}
        steps.append(step)

    patterns = {
        'pack_post': r'Step \d+ - pack & post: ([\d.eE+-]+) ms',
        'core_kernel': r'Step \d+ - core kernel launch: ([\d.eE+-]+) ms',
        'wait_mpi': r'Step \d+ - wait MPI: ([\d.eE+-]+) ms',
        'recv_unpack': r'Step \d+ - recv & unpack: ([\d.eE+-]+) ms',
        'halo_kernel': r'Step \d+ - halo kernel: ([\d.eE+-]+) ms',
    }

    for name, pattern in patterns.items():
        matches = re.findall(pattern, content)
        for i, val in enumerate(matches):
            if i < len(steps):
                steps[i][name] = float(val)

    error_times = re.findall(r'Error calc: ([\d.eE+-]+) ms', content)
    for i, val in enumerate(error_times):
        if i < len(steps):
            steps[i]['error_calc'] = float(val)

    def stats(key):
        vals = [s[key] for s in steps if key in s]
        return min(vals), sum(vals)/len(vals), max(vals) if vals else (0,0,0)

    ni, nj, nk = ni, nj, nk
    num_points = ni * nj * nk
    total_flops = 12 * num_points

    _, avg_core, _ = stats('core_kernel')
    gflops = total_flops / (avg_core / 1000.0 * 1e9) if avg_core > 0 else 0

    TPP = 4700.0
    BW = 700.0
    AI = 0.1875
    tbp = min(TPP, BW * AI)
    efficiency = (gflops / tbp * 100) if tbp > 0 else 0

    # Output
    print("="*80)
    print("ПОЛНЫЙ АНАЛИЗ ПРОИЗВОДИТЕЛЬНОСТИ: {}".format(filename))
    print("="*80)
    print("\n--- Основная информация ---")
    print("Размер задачи:       N = {} (узлов = {})".format(N, N+1))
    print("Локальный домен:     {} x {} x {} = {:,} точек".format(ni, nj, nk, num_points))
    print("Число процессов:     {}".format(processes))
    print("Общее время:         {:.3f} сек".format(total_time))
    print("\n--- Инициализация ---")
    print("Первичный обмен:     {:.3f} мс".format(init_ex))
    print("Начальный шаг:       {:.3f} мс".format(init_step))
    print("\n--- Усреднённые времена на шаг (мс) ---")
    for name, label in [
        ('pack_post', 'pack & post'),
        ('core_kernel', 'core kernel'),
        ('wait_mpi', 'wait MPI'),
        ('recv_unpack', 'recv & unpack'),
        ('halo_kernel', 'halo kernel'),
        ('error_calc', 'error calc'),
        ('total_step', 'ИТОГО шаг')
    ]:
        minv, avgv, maxv = stats(name)
        if avgv > 0:
            print("{:15}: min={:6.3f}, avg={:6.3f}, max={:6.3f}".format(label, minv, avgv, maxv))
    print("\n--- Производительность вычислений ---")
    print("Общее число FLOP:    {:.3e}".format(total_flops))
    print("Время core_kernel:   {:.3f} мс (усреднённо)".format(avg_core))
    print("Производительность:  {:.2f} GFLOP/s".format(gflops))
    print("TBP (Tesla P100):    {:.2f} GFLOP/s".format(tbp))
    print("Эффективность:       {:.2f}%".format(efficiency))
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
