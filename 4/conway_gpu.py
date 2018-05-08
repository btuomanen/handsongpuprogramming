# Conway's game of life in Python / CUDA C
# written by Brian Tuomanen for "Hands on GPU Programming with Python and CUDA"

import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation

ker = SourceModule("""
#define _X  ( threadIdx.x + blockIdx.x * blockDim.x )
#define _Y  ( threadIdx.y + blockIdx.y * blockDim.y )

#define _WIDTH  ( blockDim.x * gridDim.x )
#define _HEIGHT ( blockDim.y * gridDim.y  )

#define _XM(x)  ( (x + _WIDTH) % _WIDTH )
#define _YM(y)  ( (y + _HEIGHT) % _HEIGHT )

#define _INDEX(x,y)  ( _XM(x)  + _YM(y) * _WIDTH )

__global__ void conway_ker(int * out, int * in  )
{
   // x, y are the appropriate values for the cell covered by this thread
   int x = (int) _X, y = (int) _Y;
   
   // count the number of neighbors around the current cell
   int n = in[ _INDEX(x -1, y+1) ] + in[ _INDEX(x-1, y) ] + in[ _INDEX(x-1, y-1) ] \
                   + in[ _INDEX(x, y+1)] + in[_INDEX(x, y - 1)] \
                   + in[ _INDEX(x+1, y+1) ] + in[ _INDEX(x+1, y) ] + in[ _INDEX(x+1, y-1) ];
                   
    
    // if the current cell is alive, then determine if it lives or dies for the next generation.
    if ( in[_INDEX(x,y)] == 1)
       switch(n)
       {
          // if the cell is alive: it remains alive only if it has 2 or 3 neighbors.
          case 2:
          case 3: out[_INDEX(x,y)] = 1;
                  break;
          default: out[_INDEX(x,y)] = 0;                   
       }
    else if( in[_INDEX(x,y)] == 0 )
         switch(n)
         {
            // a dead cell comes to life only if it has 3 neighbors that are alive.
            case 3: out[_INDEX(x,y)] = 1;
                    break;
            default: out[_INDEX(x,y)] = 0;         
         }
         
}
""")


conway_ker = ker.get_function("conway_ker")
     

def update_gpu(frameNum, img, newLattice_gpu, lattice_gpu, N):
    
    conway_ker(  newLattice_gpu, lattice_gpu, grid=(N/32,N/32,1), block=(32,32,1)   )
    
    img.set_data(newLattice_gpu.get() )
    
    
    lattice_gpu[:] = newLattice_gpu[:]
    
    return img
    

if __name__ == '__main__':
    # set lattice size
    N = 256
    
    lattice = np.int32( np.random.choice([1,0], N*N, p=[0.25, 0.75]).reshape(N, N) )
    lattice_gpu = gpuarray.to_gpu(lattice)
    
    newLattice_gpu = gpuarray.empty_like(lattice_gpu)
    
    updateInterval = 0.5
    
    

    fig, ax = plt.subplots()
    img = ax.imshow(lattice_gpu.get(), interpolation='nearest')
    ani = animation.FuncAnimation(fig, update_gpu, fargs=(img, newLattice_gpu, lattice_gpu, N, ) , interval=0, frames=100, save_count=100)    
     
    plt.show()
     

