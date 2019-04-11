import tabulate
import numpy as np
import cupy as cp

from timer import stopwatch

# define ReductionKernel
rd_kinetic_kernel = cp.ReductionKernel(
    'T m, T v', # input
    'T y', # output
    'v * v', # pre-process
    'a + b', # reduction
    'y = 0.5 * m * a', # post-process
    '0', # initial
    'rd_kinetic') # name

@stopwatch
def kinetic_kernel(m, v):
    return rd_kinetic_kernel(m, v)

@stopwatch
def kinetic(m, v):
    return 0.5 * m * np.sum(v*v)

# test ReductionKernel
N = [1,10,1000000,10000000,100000000]
times_cpu = []
times_gpu = []
times_kernel = []
for n in N:
    v = np.sin(np.linspace(-np.pi, np.pi, n)).astype(np.float32) / n
    # reduction numpy
    t, y = kinetic(1, v)
    times_cpu.append(t)
    
    # reduction cupy
    v_cp = cp.asarray(v).astype(cp.float32)
    t, y_cp = kinetic(1, v_cp)
    times_gpu.append(t)
    
    # reduction kernel
    t, y_kernel = kinetic_kernel(1, v_cp)
    times_kernel.append(t)

# output table
times_cpu = np.asarray(times_cpu)
ratio_gpu = ['{:.2f} x'.format(r) for r in times_cpu / np.asarray(times_gpu)]
ratio_kernel = ['{:.2f} x'.format(r) for r in times_cpu / np.asarray(times_kernel)]

table = tabulate.tabulate(
    zip(N, times_cpu, ratio_gpu, ratio_kernel),
    headers=['N', 'NumPyでの実行時間 (sec)', '高速化倍率 (GPU)', '高速化倍率 (Kernel)'])

print(table)
