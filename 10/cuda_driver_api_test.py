from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from ctypes import *
import time
# https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#driver-api
low = -2
high = 2
breadth = 512

# NOW FULLY WORKING!!!!!!9r389389893r289fwe89jrwg89jgjgefbjoifb


# CUDA_SUCCESS = 0
# The API call returned with no errors. In the case of query calls, this also means that the operation being queried is complete (see cuEventQuery() and cuStreamQuery()). 


# culaunchkernel gives this..
# CUDA_ERROR_INVALID_VALUE = 1
# This indicates that one or more of the parameters passed to the API call is not within an acceptable range of values. 





# this stuff happens with printf

# CUDA_ERROR_INVALID_IMAGE = 200
# This indicates that the device kernel image is invalid. This can also indicate an invalid CUDA module. 

# CUDA_ERROR_INVALID_HANDLE = 400
# This indicates that a resource handle passed to the API call was not valid. Resource handles are opaque types like CUstream and CUevent. 

cuda = CDLL('nvcuda.dll')

#def cuInit(Flags):

cuInit = cuda.cuInit
cuInit.restype = int
cuInit.argtypes = [c_uint]


custatus = cuInit(0)
print 'custatus %s ' % custatus


cuDeviceGetCount = cuda.cuDeviceGetCount
cuDeviceGetCount.argtypes = [POINTER(c_int)]
cnt = c_int(0)
cuDeviceGetCount(byref(cnt))
print 'cudevicecnt %s ' % cnt.value

cuDevice = c_int(0)

cuDeviceGet = cuda.cuDeviceGet
cuDeviceGet.argtypes = [POINTER(c_int), c_int]
cuDeviceGet(byref(cuDevice), 0)

print 'cudevice %s ' % cuDevice.value


# create context
cuContext = c_void_p()

cuCtxCreate = cuda.cuCtxCreate
cuCtxCreate.argtypes = [c_void_p, c_int, c_int]
cuCtxCreate(byref(cuContext), 0, cuDevice)





cuModule = c_void_p()

cuModuleLoad = cuda.cuModuleLoad
cuModuleLoad.argtypes = [POINTER(c_void_p), c_char_p]
cuModuleLoad.restype  = int
ml_out = cuModuleLoad(byref(cuModule), c_char_p('./mandelbrot.ptx'))
print 'moduleload %s' % ml_out

# wrapper for context synchronization

cuCtxSynchronize = cuda.cuCtxSynchronize

cuCtxSynchronize.argtypes = []
cuCtxSynchronize.restype = int

# synchronize context
ctx_out = cuCtxSynchronize()
print 'ctx sync: %s ' % ctx_out

#
lattice = np.linspace(low, high, breadth, dtype=np.float32)
lattice_c = lattice.ctypes.data_as(POINTER(c_float))

# cuMemcpyHtoD ( CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount ) 

lattice_gpu = c_void_p(0)

cuMemAlloc = cuda.cuMemAlloc
cuMemAlloc.argtypes = [c_void_p, c_size_t]
cuMemAlloc.restype = int

ma_out = cuMemAlloc(addressof(lattice_gpu), c_size_t(lattice.size*sizeof(c_float)))

print 'mem alloc: %s , ' % ma_out

# synchronize context
ctx_out = cuCtxSynchronize()
print 'ctx sync: %s ' % ctx_out

graph_gpu = c_void_p(0)
ma_out = cuMemAlloc(addressof(graph_gpu), c_size_t(lattice.size**2 * sizeof(c_float)))
print 'memalloc (graph) %s' % ma_out

# synchronize context
ctx_out = cuCtxSynchronize()
print 'ctx sync: %s ' % ctx_out

graph = np.zeros(shape=(lattice.size, lattice.size), dtype=np.float32)

cuMemcpyHtoD = cuda.cuMemcpyHtoD 
cuMemcpyHtoD.argtypes = [c_void_p, c_void_p, c_size_t]
h2d_out  = cuMemcpyHtoD(lattice_gpu, lattice_c, c_size_t(lattice.size*sizeof(c_float)))
print 'h2d %s ' % h2d_out

# synchronize context
ctx_out = cuCtxSynchronize()
print 'ctx sync: %s ' % ctx_out

mandel_ker = c_void_p(0)

cuModuleGetFunction = cuda.cuModuleGetFunction
cuModuleGetFunction.argtypes = [c_void_p, c_void_p, c_char_p ]
cuModuleGetFunction.restype = int

getfun_out = cuModuleGetFunction(addressof(mandel_ker), cuModule, c_char_p('mandelbrot_ker'))
print 'cuModuleGetFUnction: %s' % getfun_out

'''
CUresult cuLaunchKernel 
(
CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, 
unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, 
unsigned int sharedMemBytes, CUstream hStream, void ** kernelParams, void ** extra)
'''

cuLaunchKernel = cuda.cuLaunchKernel
# 
cuLaunchKernel.argtypes = [c_void_p, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_void_p, c_void_p, c_void_p]
#cuLaunchKernel.argtypes = [c_void_p, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_void_p, c_void_p, c_void_p]

cuLaunchKernel.restype = int
# mandelbrot_ker(float * lattice, float * mandelbrot_graph, int max_iters, float upper_bound_squared, int lattice_size)
max_iters = c_int(256)
upper_bound_squared = c_float(2*2)
lattice_size = c_int(lattice.size)

pvp = POINTER(c_void_p)

mandel_args0 = [lattice_gpu, graph_gpu, max_iters, upper_bound_squared, lattice_size ]
mandel_args = [c_void_p(addressof(x)) for x in mandel_args0]
mandel_params = (c_void_p * len(mandel_args))(*mandel_args)

gridsize = int(np.ceil(lattice.size**2 / 32))

lk_out = cuLaunchKernel(mandel_ker, gridsize, 1, 1, 32, 1, 1, 0, 0, mandel_params, None) #cast(mandel_params, POINTER(c_void_p)), None)
print 'luanchker out %s ' % lk_out
# synchronize context
ctx_out = cuCtxSynchronize()
print 'ctx sync: %s ' % ctx_out

cuMemcpyDtoH = cuda.cuMemcpyDtoH 
cuMemcpyDtoH.argtypes = [c_void_p, c_void_p, c_size_t]
cuMemcpyDtoH.restype = int

d2h_out = cuMemcpyDtoH( cast(graph.ctypes.data, c_void_p), graph_gpu,  c_size_t(lattice.size**2*sizeof(c_float)))

print ' d2h %s' % d2h_out

# synchronize context
ctx_out = cuCtxSynchronize()
print 'ctx sync: %s ' % ctx_out
 
fig = plt.figure(1)
plt.imshow(graph, extent=(-2, 2, -2, 2))
plt.show()

