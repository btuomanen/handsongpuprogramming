from __future__ import division
from time import time
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from ctypes import *
from cuda_driver import *

def mandelbrot(breadth, low, high, max_iters, upper_bound):

    # Initialize GPU 0
    cuInit(0)

    cnt = c_int(0)
    cuDeviceGetCount(byref(cnt))

    cuDevice = c_int(0)
    cuDeviceGet(byref(cuDevice), 0)
    
    cuContext = c_void_p()
    cuCtxCreate(byref(cuContext), 0, cuDevice)

    cuModule = c_void_p()
    cuModuleLoad(byref(cuModule), c_char_p('./mandelbrot.ptx'))

    #cuCtxSynchronize()

    lattice = np.linspace(low, high, breadth, dtype=np.float32)
    lattice_c = lattice.ctypes.data_as(POINTER(c_float))
    lattice_gpu = c_void_p(0)

    cuMemAlloc(addressof(lattice_gpu), c_size_t(lattice.size*sizeof(c_float)))

    #cuCtxSynchronize()

    graph_gpu = c_void_p(0)
    cuMemAlloc(addressof(graph_gpu), c_size_t(lattice.size**2 * sizeof(c_float)))

    # synchronize context
    #cuCtxSynchronize()

    graph = np.zeros(shape=(lattice.size, lattice.size), dtype=np.float32)

    cuMemcpyHtoD(lattice_gpu, lattice_c, c_size_t(lattice.size*sizeof(c_float)))

    #cuCtxSynchronize()

    mandel_ker = c_void_p(0)

    cuModuleGetFunction(addressof(mandel_ker), cuModule, c_char_p('mandelbrot_ker'))

    max_iters = c_int(max_iters)
    upper_bound_squared = c_float(upper_bound**2)
    lattice_size = c_int(lattice.size)

    mandel_args0 = [lattice_gpu, graph_gpu, max_iters, upper_bound_squared, lattice_size ]
    mandel_args = [c_void_p(addressof(x)) for x in mandel_args0]
    mandel_params = (c_void_p * len(mandel_args))(*mandel_args)

    gridsize = int(np.ceil(lattice.size**2 / 32))
    
    cuLaunchKernel(mandel_ker, gridsize, 1, 1, 32, 1, 1, 10000, None, mandel_params, None)

    # synchronize context
    cuCtxSynchronize()


    cuMemcpyDtoH( cast(graph.ctypes.data, c_void_p), graph_gpu,  c_size_t(lattice.size**2*sizeof(c_float)))

    
    cuMemFree(lattice_gpu)
    cuMemFree(graph_gpu)
    
    cuCtxDestroy(cuContext)

    
    return graph


if __name__ == '__main__':

    t1 = time()
    mandel = mandelbrot(512,-2,2,256, 2)
    t2 = time()

    mandel_time = t2 - t1

    print 'It took %s seconds to calculate the Mandelbrot graph.' % mandel_time
    
    fig = plt.figure(1)
    plt.imshow(mandel, extent=(-2, 2, -2, 2))
    plt.show()
