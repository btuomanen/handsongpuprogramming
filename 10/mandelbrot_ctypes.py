from time import time
import matplotlib
#this will prevent the figure from popping up
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from ctypes import *

mandelbrot_dll = CDLL('./mandelbrot.dll')
mandel = mandelbrot_dll.launch_mandelbrot
mandel.argtypes = [POINTER(c_float), POINTER(c_float), c_int, c_float, c_int]


def gpu_mandelbrot(breadth, low, high, max_iters, upper_bound):

    lattice = np.linspace(low, high, breadth, dtype=np.float32)
    out = np.empty(shape=(lattice.size,lattice.size), dtype=np.float32)
    mandel(lattice.ctypes.data_as(POINTER(c_float)), out.ctypes.data_as(POINTER(c_float)), c_int(max_iters), c_float(upper_bound), c_int(lattice.size) )  
    
    return out


if __name__ == '__main__':

    t1 = time()
    mandel = gpu_mandelbrot(512,-2,2,100, 2)
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
