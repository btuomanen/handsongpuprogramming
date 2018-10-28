from ctypes import *
import sys

if 'linux' in sys.platform:
	cuda = CDLL('libcuda.so')
elif 'win' in sys.platform:
	cuda = CDLL('nvcuda.dll')

CUDA_ERRORS = {0 : 'CUDA_SUCCESS', 1 : 'CUDA_ERROR_INVALID_VALUE', 200 : 'CUDA_ERROR_INVALID_IMAGE', 400 : 'CUDA_ERROR_INVALID_HANDLE' }

cuInit = cuda.cuInit
cuInit.argtypes = [c_uint]
cuInit.restype = int

cuDeviceGetCount = cuda.cuDeviceGetCount
cuDeviceGetCount.argtypes = [POINTER(c_int)]
cuDeviceGetCount.restype = int

cuDeviceGet = cuda.cuDeviceGet
cuDeviceGet.argtypes = [POINTER(c_int), c_int]
cuDeviceGet.restype = int

cuCtxCreate = cuda.cuCtxCreate
cuCtxCreate.argtypes = [c_void_p, c_int, c_int]
cuCtxCreate.restype = int

cuModuleLoad = cuda.cuModuleLoad
cuModuleLoad.argtypes = [POINTER(c_void_p), c_char_p]
cuModuleLoad.restype  = int

cuCtxSynchronize = cuda.cuCtxSynchronize
cuCtxSynchronize.argtypes = []
cuCtxSynchronize.restype = int

cuMemAlloc = cuda.cuMemAlloc
cuMemAlloc.argtypes = [c_void_p, c_size_t]
cuMemAlloc.restype = int

cuMemcpyHtoD = cuda.cuMemcpyHtoD 
cuMemcpyHtoD.argtypes = [c_void_p, c_void_p, c_size_t]
cuMemAlloc.restype = int

cuMemcpyDtoH = cuda.cuMemcpyDtoH 
cuMemcpyDtoH.argtypes = [c_void_p, c_void_p, c_size_t]
cuMemcpyDtoH.restype = int

cuLaunchKernel = cuda.cuLaunchKernel
cuLaunchKernel.argtypes = [c_void_p, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_void_p, c_void_p, c_void_p]
cuLaunchKernel.restype = int



