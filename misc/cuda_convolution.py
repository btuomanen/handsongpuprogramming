from __future__ import division
import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np
from scipy.ndimage.filters import convolve
from time import time

#  This file contains a simple convolution function implemented as a CUDA kernel,
#  which has an interface to Python through the "cuda_convolve" function.

#  "cuda_convolve" acts exactly as SciPy's convolve with mode='constant':
#  output = cuda_convolve(X, window)

#  (X and window can be tensors of rank 1, 2, or 3, represented as NumPy float32 arrays.)

#  The "__main__" here has many test cases comparing the output to SciPy's convolve,
#  as well as timing comparisons for different sized random tensors.
#  (Note that this was only tested on a GTX 1050 and GTX 1070; you might want to modify the tests
#   so that the generated tensors are much larger if you are using a more powerful GPU.)

#  Note that this program is reliant on having the PyCUDA module installed; of course, this won't work without an
#  appropriate GPU setup either!

#  Also note that this implementation makes a direct computation rather than using FFTs.
#  (I can implement an FFT version if anyone wants it!)

#  Please feel free to get in touch if you have any questions.

#  -- Brian Tuomanen, 9/18/2018 (btuomanen@outlook.com)


#  -------

#  The following is the CUDA C code for the main kernel we will be using.

#  Parallelization is achieved by assigning a CUDA thread to each point in the
#  output;  we design the grid / block launch parameters so that a thread's global
#  (x,y,z) tuple corresponds exactly the output index.

ConvCode = '''
//  When we plug a PyCUDA gpuarray object into a CUDA kernel, the kernel ends up "seeing" only
//  a single pointer array.  If we want to use this as a 3D tensor, the easiest thing to do is to
//  set up some macros that will translate an (x,y,z) tuple to a single-index.
//  (By the nature that PyCUDA stores arrays, it made more sense to use (k,i,j) indexing here.)

//  The following macro gives the values of the window array (x) in reverse order, for convolution.

#define w_r(k,i,j)  ( w[(w_k - k - 1) * ( w_i * w_j) + (w_i - i - 1) * w_j + (w_j - j - 1) ] )

//  The following macro gives the (k,i,j) value of the "x" array.
//  Notice how this will automatically yield a 0 if we go over or under the max/min indices--
//  this allows for automatic 0 padding.

#define x_(k,i,j)  ( ( (k) < x_k && (i) < x_i && (j) < x_j && (k) >= 0 && (i) >= 0 && (j) >= 0 ) ? x[ (k) * ( x_i * x_j) + (i) * x_j + (j) ] : 0.0f )

//  The following macro gives the (k,i,j) of the "X" array with some offsets.
//  (This is necessary to exactly replicate the behavior of "convolution" from the scipy.ndimage.filters submodule.)

#define xo_(k,i,j) x_( (k +offset_k) , (i+offset_i), (j+offset_j) )


//  This is just to give a (k,i,j) index for x, rather than the corresponding x value.

#define kij_x(k,i,j)  ( (k) * ( x_i * x_j) + (i) * x_j + (j) )



//  The following function calculates the convolution value at a particular (k,i,j) point.
//  It's really just the "multiply and accumulate" step in convolution.

//  (We set this to be "inline" to save a little time from branching from the kernel.)

__device__ float inline filter_at(float * x, int x_i, int x_j, int x_k, \
                                  float * w, int w_i, int w_j, int w_k, \
                                  int k0, int i0, int j0 , \
                                  int offset_k, int offset_i, int offset_j)
{

     // when we multiply and accumulate a large number of values,
     // there is especially great numerical loss when we only use 32 bit
     // floats.
     
     // This can be mitigated by accumulating our 32 bit floats into a 64 bit double,
     // and then typecasting that back to a 32 bit float upon return.
     
     //  This has little impact on performance: using a double here requires
     //  135 ms to complete Wolf's example, and will satisfy NumPy's allclose
     //  for larger arrays when compared to the SciPy output. (This is according to "timeit")
     
     //  Using a float takes 134 ms for Wolf's example, but doesn't always 
     //  satisfy allclose for larger arrays.  Greater accuracy is always worth a millisecond!
     
     double v = 0;
     
     for (int k=0; k < w_k; k++)
         for (int i=0; i < w_i; i++)
             for (int j=0; j < w_j; j++)
                 v += (double) (  ( (double) xo_(k + k0, i + i0, j + j0) )  * ( (double) w_r(k,i,j)) );
                 
    return ((float) v);

}


// This is our main CUDA kernel for convolution.

// "x" is the tensor "image" array we which to convolve against.  Its tensor dimensions are given
// by x_k, x_i, and x_j.

// "w" is the convolution kernel/window that we will convolve against "x".  Its tensor dimensions
// are given by w_k, w_i, w_j.

// "offset_k/i/j" are offset parameters for convolution.  We mainly do this to center our convolution
//  so that it will correspond to SciPy's "convolve" function.

// "output" is a pre-allocated array that we will store the outputs to.  It should have the same size
// and same dimensions as x.

__global__ void convolve_kernel( float * x, int x_k, int x_i, int x_j,  \
  float * w, int w_k, int w_i, int w_j, int offset_k, int offset_i, int offset_j, float *output)
{


     // As stated, each (k,i,j) value for the output will be represented by a single thread;
     // We can get the global (k,i,j) values as seen below;  this will allow for scaling of the
     // block and grid launch parameters later.


     int k = blockIdx.x * blockDim.x + threadIdx.x;
     int i = blockIdx.y * blockDim.y + threadIdx.y;
     int j = blockIdx.z * blockDim.z + threadIdx.z;
     
     
     // In most cases, there will be some superfluous threads that are launched.
     // This is just the nature of CUDA and is usually something we can't avoid.
     // We will want to check that our current thread actually corresponds to an output point
     // before we calculate anything;  we can do this by seeing if any value in (k,i,j)
     // exceeds the boundaries of the output array (which will have the same boundaries
     // as x here)

     // (If this doesn't correspond to an output point, then just return!)

     if ( k < x_k && i < x_i && j < x_j)
         output[kij_x(k,i,j)] = filter_at(x, x_i, x_j, x_k, w, w_i, w_j, w_k, k, i, j, \
                                            offset_k, offset_i, offset_j);

     return;

}
'''

# This will compile and link the above CUDA code for us automatically.
conv_mod = SourceModule(source=ConvCode)

# This gives us a function that will automatically launch our CUDA kernel from Python.
convolve_ker = conv_mod.get_function("convolve_kernel")


#  "cuda_convolve" provides an interface to our CUDA kernel, so that it has 
#  a similar interface to the SciPy "convolve" (with mode set to 'constant')

#  Note that x and w can be either NumPy arrays or PyCUDA gpuarray objects.
#  If either is already a gpuarray, then we can just use it.  Otherwise
#  we assume it is a NumPy array, and copy its values to the GPU with gpuarray.to_gpu

#  We can also plug in a pre-allocated output array, output_gpu, and specify a CUDA stream
#  as well.

#  One other thing to note:  x and w can be tensors of rank 1, 2, or 3.
#  Even though our kernel is designed for rank-3 tensors, it easily scales to 2 or 1 with
#  some tweaks.

def cuda_convolve(x, w, output_gpu=None, stream=None):
    
    
    if (type(x) != gpuarray.GPUArray):
        x_gpu = gpuarray.to_gpu(np.float32(x) )
    else:
        x_gpu = x
        
    if (type(w) != gpuarray.GPUArray):
        w_gpu = gpuarray.to_gpu( np.float32(w) )
    else:
        w_gpu = w
        
    if (len(x.shape) != len (w.shape)):
        raise Exception("Error: Tensors x and w must have same rank.")
    
    tensor_rank = len(x.shape)
    
    if (tensor_rank > 3):
        raise Exception("Error: tensor rank of x and w must not exceed 3!")

    if (tensor_rank == 0):
        raise Exception("Error: scalars (tensor rank 0) are not supported!")
    

    #  Allocate memory if a pre-allocated array is not passed in.
    if (output_gpu is None):
        output_gpu = gpuarray.empty_like(x_gpu)
        

    #  We set these 1-padded lists up so that our kernel
    #  may handle rank-2 and 1 tensors.

    #  (We just tell the CUDA kernel that the first dimension is 1 in the case
    #  of rank 2, or that the first two dimensions are rank 1
    #  in the case of rank 1.)

    x_shape = [1,1,1]
    w_shape = [1,1,1]

    #  Notice how we set the CUDA block size parameter in the follwing code;  if a particular
    #  dimension is not being used (as in the case of rank 2 and 1 tensors)
    #  we set it to 1.

    #  Also, notice that the sizes are 64 or 32.  (4*4*4 == 64, 1*8*8==64, 1*1*32==32).
    #  A CUDA block will always reside in at least CUDA warp (which is 32 threads), so one that is
    #  (1,1,33) will take up 64 threads anyway. 

    #  (That's why it's good practice to make the total number of threads in a CUDA block a multiple of 32.)

    
    if(tensor_rank == 3):
        
        x_shape[0] = x.shape[0]
        x_shape[1] = x.shape[1]
        x_shape[2] = x.shape[2]
        
        w_shape[0] = w.shape[0]
        w_shape[1] = w.shape[1]
        w_shape[2] = w.shape[2]
        
        block=(4,4,4)
        
    elif(tensor_rank == 2):
        
        x_shape[1] = x.shape[0]
        x_shape[2] = x.shape[1]
        
        w_shape[1] = w.shape[0]
        w_shape[2] = w.shape[1]
        
        block=(1,8,8)
        
    elif(tensor_rank == 1):
        
        x_shape[2] = x.shape[0]
        w_shape[2] = w.shape[0]
        
        block=(1,1,32)
        
    
    # The grid size will scale with the dimensions of x
    # Remember:  a CUDA kernel is launched over a grid with a particular block size.

    # (The grid values are given by the blockDim.x/y/z values in CUDA, which will exactly
    #  correspond to these values.)

    
    grid_k = int(np.ceil(float(x_shape[0]) / float(block[0])))
    grid_i = int(np.ceil(float(x_shape[1]) / float(block[1])))
    grid_j = int(np.ceil(float(x_shape[2]) / float(block[2])))
    
    grid = (grid_k, grid_i, grid_j)
    
    x_k = np.int32(x_shape[0])
    x_i = np.int32(x_shape[1])
    x_j = np.int32(x_shape[2])
    
    w_k = np.int32(w_shape[0])
    w_i = np.int32(w_shape[1])
    w_j = np.int32(w_shape[2])
    

    # these are the offsets that SciPy's convolve uses, which are related to the size of the window parameter.

    offset_k = np.int32(-w_k // 2 + 1)
    offset_i = np.int32(-w_i // 2 + 1)
    offset_j = np.int32(-w_j // 2 + 1)


    # We launch the kernel from PyCUDA here.  Note that we have to very carefully typeset all values input into the
    # kernel itself with the exact corresponding NumPy types.

    # we specify the block and grid launch parameters towards the end (Usually we use <<< >>> for this in CUDA C
    # proper.)

    # We also specify the CUDA stream (if any) at the end.
    
    convolve_ker(x_gpu, x_k, x_i, x_j, w_gpu, w_k, w_i, w_j, offset_k, offset_i, offset_j, output_gpu, block=block, grid=grid, stream=stream)
    
    output = output_gpu.get()
    
    return output
    


def test_1d_tensors():
    
    lengths = [512, 999, 1000, 1001, 1024, 5000, 6328, 10000]
    
    conv_lengths = [10, 16, 20, 32, 64, 100, 199, 256]
    
    
    
    fi = np.random.choice([np.random.rand,np.random.randn])
    fk = np.random.choice([np.random.rand,np.random.randn])
    
    x_l = np.random.choice(lengths)
    c_l = np.random.choice(conv_lengths)
    
    X = np.array( fi( x_l ), dtype=np.float32)
    w = np.array( fk (c_l), dtype=np.float32 ) 
    
    t0 = time()
    res = convolve(X, w, mode='constant')
    t = time() - t0
    
    t1 = time()
    res_cu = cuda_convolve(X,w)
    t_cuda = time() - t1
    

    print('\n Generating random rank-1 tensor with length %s, with random window of length %s \n' \
          % (x_l, c_l))
    
    success = np.allclose(res, res_cu)

    print('Does cuda_convolve match convolve?: %s \n' % success ) 
    print('Duration (host) : %s\nDuration (GPU): %s' % (t, t_cuda) )

    return (success, t, t_cuda)


    
def test_2d_tensors():
    
    lengths = [100, 103, 128, 256, 500, 512, 1000, 1024]
    
    conv_lengths = [1, 2, 3, 5, 8, 10, 16, 20, 32, 44, 51, 55, 64]
    
    x_i = np.random.choice(lengths)
    x_j = np.random.choice(lengths)
    
    w_i = np.random.choice(conv_lengths)
    w_j = np.random.choice(conv_lengths)
    
    fi = np.random.choice([np.random.rand,np.random.randn])
    fk = np.random.choice([np.random.rand,np.random.randn])
    
    X = np.array( fi( x_i, x_j  ), dtype=np.float32)
    w = np.array( fk (w_i,w_j) , dtype=np.float32) 
    
    t0 = time()
    res = convolve(X, w, mode='constant')
    t = time() - t0

    t1 = time()
    res_cu = cuda_convolve(X,w)
    t_cuda = time() - t1
    
    print('\n Generating random rank-2 tensor with dimensions %s, with random window with dimensions %s \n' \
          % ((x_i, x_j), (w_i, w_j)))
    
    success = np.allclose(res, res_cu)

    print('Does cuda_convolve match convolve?: %s \n' % success ) 
    print('Duration (host) : %s\nDuration (GPU): %s' % (t, t_cuda) )

    return (success, t, t_cuda)




def test_3d_tensors():
    
    lengths_k = [8, 10, 16, 32, 33, 44, 50, 53, 64, 128]
    lengths_i = [32, 44, 50, 53, 64, 82, 100, 128, 256]
    lengths_j = [256, 500, 512, 640]
    
    conv_lengths = [1, 4, 8, 16]    
    
    x_k = np.random.choice(lengths_k)
    x_i = np.random.choice(lengths_i)
    x_j = np.random.choice(lengths_j)
    
    w_k = np.random.choice(conv_lengths)
    w_i = np.random.choice(conv_lengths)
    w_j = np.random.choice(conv_lengths)
    
    fi = np.random.choice([np.random.rand,np.random.randn])
    fk = np.random.choice([np.random.rand,np.random.randn])
    
    X = np.array( fi( x_k, x_i, x_j  ), dtype=np.float32)
    w = np.array( fk (w_i,w_j,w_k) , dtype=np.float32) 
    
    t0 = time()
    res = convolve(X, w, mode='constant')
    t = time() - t0
    
    t1 = time()
    res_cu = cuda_convolve(X,w)
    t_cuda = time() - t1
    

    print('\nGenerating random rank-3 tensor with dimensions %s, with random window with dimensions %s . \n' \
          % ((x_k, x_i, x_j), (w_k, w_i, w_j)))

    success = np.allclose(res, res_cu)

    print('Does cuda_convolve match convolve?: %s \n' % success ) 
    print('Duration (host) : %s\nDuration (GPU): %s' % (t, t_cuda) )

    return (success, t, t_cuda)




if __name__ == '__main__':
    
    #  (We will run cuda_convolve once on some small arrays and disregard
    #  the output.  This will ensure that the CUDA kernel has already been 
    #  compiled and linked by PyCUDA before we being our tests.)

    cuda_convolve(np.ones((10,10),dtype=np.float32), np.ones((2,2),dtype=np.float32) )

    print('----')
    print('Test case: Wolf\'s example ')
    print('----\n')

    X = np.array( np.random.rand(10, 1000, 1000), dtype=np.float32) 
    window = np.ones((1,9,9), dtype=np.float32) 

    window /= np.sum(window.ravel())
    
    t0 = time()
    res = convolve(X, window, mode='constant')
    t = time() - t0

    t1 = time()
    res_cu = cuda_convolve(X, window)
    t_cuda = time() - t1

    print('Does cuda_convolve match convolve?: %s' % np.allclose(res, res_cu))
    print('Duration (host) : %s\nDuration (GPU): %s' % (t, t_cuda) )

    print('\n----')
    print('Performing additional test cases:')
    print('----\n')

    passed = []

    print('\n----')
    print('Testing convolution over random rank-1 tensors (5 runs):')
    print('----')

    for _ in range(5):
        
        passed.append(test_1d_tensors()[0])

    print('\n----')
    print('Testing convolution over random rank-2 tensors (5 runs):')
    print('----')

    for _ in range(5):
        passed.append(test_2d_tensors()[0])

    print('\n----')
    print('Testing convolution over random rank-3 tensors (5 runs):')
    print('----')

    for _ in range(5):
        passed.append(test_3d_tensors()[0])


    print('\n----')
    print('Results: passed %s out of %s test cases.' % (sum(passed), len(passed)))

    print('----\n')


