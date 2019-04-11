import tabulate
import numpy as np
import cupy as cp

from timer import stopwatch

# define ElementwiseKernel
ew_kinetic_kernel = cp.ElementwiseKernel(
    'T x, T v, T dt', # input
    'T y', # output
    'y = x + v*dt', # elementwise
    'ew_kinetic') # name

@stopwatch
def kinetic_kernel(x, v, dt):
    return ew_kinetic_kernel(x, v, dt)

@stopwatch
def kinetic(x, v, dt):
    return x + v*dt

# test ElementwiseKernel
N = [1,10,1000000,10000000,100000000]
times_cpu = []
times_gpu = []
times_kernel = []
for n in N:
    x = np.cos(np.linspace(-np.pi, np.pi, n)).astype(np.float32)
    v = np.sin(np.linspace(-np.pi, np.pi, n)).astype(np.float32)
    # elementwise numpy
    t, y = kinetic(x, v, 1)
    times_cpu.append(t)
    
    # elementwise cupy
    x_cp = cp.asarray(x).astype(cp.float32)
    v_cp = cp.asarray(v).astype(cp.float32)
    t, y_cp = kinetic(x_cp, v_cp, 1)
    times_gpu.append(t)
    
    # elementwise kernel
    t, y_kernel = kinetic_kernel(x_cp, v_cp, 1)
    times_kernel.append(t)

# output table
times_cpu = np.asarray(times_cpu)
ratio_gpu = ['{:.2f} x'.format(r) for r in times_cpu / np.asarray(times_gpu)]
ratio_kernel = ['{:.2f} x'.format(r) for r in times_cpu / np.asarray(times_kernel)]

table = tabulate.tabulate(
    zip(N, times_cpu, ratio_gpu, ratio_kernel),
    headers=['N', 'NumPyでの実行時間 (sec)', '高速化倍率 (GPU)', '高速化倍率 (Kernel)'])

print(table)
