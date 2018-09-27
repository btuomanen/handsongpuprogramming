from __future__ import division
import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np

LayerEvalSource = '''
#define _RELU(x) ( ((x) > 0.0f) ? (x) : 0.0f )
#define _SIGMOID(x)  ( 1.0f / (1.0f + exp(-(x)) ))

__global__ void layer_eval(int num_outputs, int num_inputs, int relu, int sigmoid, float * w, float * b, \
                           float * x, float *y, int batch_size, int w_t, int b_t, float delta)
{
     int i = blockDim.x*blockIdx.x + threadIdx.x;
     
     float old_w, old_b;
     

     if( w_t >= 0 )
     {
          old_w = w[w_t];
          w[w_t] += delta;
     }
     
     
     if( b_t >= 0 )
     {
          old_b = b[b_t];
          b[b_t] += delta;
     }
     
     
     if(batch_size <= 0)
         batch_size = 1;
     
        
     if (i < num_outputs)
     {
         for(int k=0; k < batch_size; k++)
         {    
              double temp = 0;
              
              for (int j = 0; j < num_inputs; j++)
              {
                  temp += ((double) w[ (num_inputs) * i + j ] ) * ( (double) x[k * num_inputs + j]);  // 
              }
                  
              temp += (double) b[i];
              
              if (relu > 0)
                  temp = _RELU(temp);
                  
              if (sigmoid > 0)
                  temp = _SIGMOID(temp);
              
              
              y[k * num_outputs + i] = (float) temp;                 
         }
    }
         
    if( w_t >= 0 )
    {
          w[w_t] = old_w;
    }
     
     if( b_t >= 0 )
     {
          b[b_t] = old_b;
     }
         
    return;
}
'''

mod = SourceModule(LayerEvalSource)

eval_ker = mod.get_function('layer_eval')


class DenseLayer:
    
    def __init__(self, num_inputs=None, num_outputs=None, weights=None, b=None, stream=None, relu=False, sigmoid=False, dropout=None, delta=0.001):
        
        self.stream = stream
        
        if weights is None:
            weights = np.random.rand(num_outputs, num_inputs)
            self.num_inputs = np.int32(num_inputs)
            self.num_outputs = np.int32(num_outputs)            
        
        if type(weights) != pycuda.gpuarray.GPUArray:
            self.weights = gpuarray.to_gpu_async(np.array(weights, dtype=np.float32) , stream = self.stream)
        else:
            self.weights = weights
        
        
        if num_inputs is None or num_outputs is None:
            
            self.num_inputs = np.int32(self.weights.shape[1])
            self.num_outputs = np.int32(self.weights.shape[0])


        if b is None:
            b = gpuarray.zeros((self.num_outputs,),dtype=np.float32)
            
        if type(b) != pycuda.gpuarray.GPUArray:
            self.b = gpuarray.to_gpu_async(np.array(b, dtype=np.float32) , stream = self.stream)
        else:
            self.b = b   
        
        self.relu = np.int32(relu)
        self.sigmoid = np.int32(sigmoid)
        
        self.block = (32,1,1)
        
        self.grid = (int(np.ceil(self.num_outputs / 32)), 1,1)
        
        self.delta = np.float32(delta)
        

    def eval_(self, x, y=None):
    
        if type(x) != pycuda.gpuarray.GPUArray:
            x = gpuarray.to_gpu_async(np.array(x,dtype=np.float32) , stream=self.stream)
            
        if len(x.shape) == 2:
            batch_size = np.int32(x.shape[0])
        else:
            batch_size = np.int32(1)
        
        
        if y is None:
            if batch_size == 1:
                y = gpuarray.empty((self.num_outputs,), dtype=np.float32)
            else:
                y = gpuarray.empty((self.num_outputs, batch_size), dtype=np.float32)
            #y = gpuarray.empty(x.shape, dtype=np.float32)
            
            
        #__global__ void layer_eval(int num_outputs, int num_inputs, int relu, int sigmoid, float * w, float * b, \
        #    float * x, float *y, int batch_size, int w_t, int b_t, float delta)
        eval_ker(self.num_outputs, self.num_inputs, self.relu, self.sigmoid, \
                 self.weights, self.b, x, y, batch_size, np.int32(-1), np.int32(-1), \
                 self.delta , block=self.block, grid=self.grid , stream=self.stream)
        
        return y
        
        
    def eval_batch():
        pass
        
    def eval_batch_weight():
        pass
    

weights = [[1,2,3],[4,5,6]]
nl = DenseLayer(weights=weights)
