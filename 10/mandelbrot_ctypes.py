from time import time
import matplotlib
#this will prevent the figure from popping up
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from ctypes import *

mandelbrot_dll = windll.LoadLibrary('./mandelbrot.dll')
lm = mandelbrot_dll.launch_mandelbrot
# launch_mandelbrot(float * lattice, float * mandelbrot_graph, int max_iters,
# float upper_bound, int lattice_size)
lm.argtypes = [POINTER(c_float), POINTER(c_float), c_int, c_float, c_int]


def gpu_mandelbrot(width, height, real_low, real_high, imag_low, imag_high, max_iters, upper_bound):

    # we set up our complex lattice as such
    lattice = np.matrix(np.linspace(real_low, real_high, width), dtype=np.float32)
    #lattice_im = np.matrix(np.linspace( imag_high, imag_low, height), dtype=np.float32) 
    #lattice_im = lattice_im.transpose()
    
    out = np.empty(shape=(lattice.size,lattice.size), dtype=np.float32)
    
    lm(lattice.ctypes.data_as(POINTER(c_float)), out.ctypes.data_as(POINTER(c_float)), c_int(max_iters), c_float(upper_bound), c_int(lattice.size) )  
    
    # copy complex lattice to the GPU
    #mandelbrot_lattice_gpu = gpuarray.to_gpu(mandelbrot_lattice)

    # allocate an empty array on the GPU
    #mandelbrot_graph_gpu = gpuarray.empty(shape=mandelbrot_lattice.shape, dtype=np.float32)

    #mandel_ker( mandelbrot_lattice_gpu, mandelbrot_graph_gpu, np.int32(max_iters), np.float32(upper_bound))
              
    #mandelbrot_graph = mandelbrot_graph_gpu.get()
    
    
    
    return out


if __name__ == '__main__':

    t1 = time()
    mandel = gpu_mandelbrot(512,512,-2,2,-2,2,100, 2)
    t2 = time()

    mandel_time = t2 - t1

    t1 = time()
    fig = plt.figure(1)
    plt.imshow(mandel, extent=(-2, 2, -2, 2))
    plt.savefig('mandelbrot_ctypes.png', dpi=fig.dpi)
    t2 = time()

    dump_time = t2 - t1

    print 'It took {} seconds to calculate the Mandelbrot graph.'.format(mandel_time)
    print 'It took {} seconds to dump the image.'.format(dump_time)
