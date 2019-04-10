import time
import tabulate
import numpy as np
import cupy as cp

# define ElementwiseKernel
def ew_kinetic(x, v, dt):
    return y = x + v*dt

ew_kinetic_kernel = cp.ElementwiseKernel(
    'T x, T v, T dt', # input
    'T y', # output
    'x + v*dt', # elementwise
    'ew_kinetic') # name

# define ReductionKernel
rd_kinetic_kernel = cp.ReductionKernel(
    'T x', # input
    'T y', # outpur
    'x * x', # pre-process
    'a + b', # reduction
    'y = sqrt(a)', # post-process
    'rd_kinetic') # name

# test ElementwiseKernel
N = [100,1000,10000]
times_cpu = []
times_gpu = []
for n in N:
    x = np.cos(np.linspace(-np.pi, np.pi, n))
    v = np.cos(np.linspace(-np.pi, np.pi, n))

    # elementwise numpy
    t0 = time.time()
    y = ew_kinetic(x, v, 1)
    times_cpu.append(time.time() - t0)

    # elementwise cupy
    x_cp = cp.asarray(x)
    v_cp = cp.asarray(v)

    t0 = time.time()
    y_cp = ew_kinetic_kernel(x, v, 1)
    times_gpu.append(time.time() - t0)

times_cpu = np.asarray(times_cpu)
times_gpu = np.asarray(times_gpu)
ratio = ['{:.2f} x'.format(r) for r in times_cpu / times_gpu]

table = tabulate.tabulate(
    zip(N, times_cpu, times_gpu, ratio),
    headers=['N', 'NumPyでの実行時間 (sec)', 'CuPy での実行時間 (sec)', '高速化倍率'])

print(table)
