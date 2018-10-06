from __future__ import division
import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule
from pycuda.elementwise import ElementwiseKernel
import numpy as np
from Queue import Queue

def cross_entropy(predictions=None, ground_truth=None):
    
    if predictions is None or ground_truth is None:
        raise Exception("error!  Both predictions and ground truth must be float32 arrays")
    
    p = np.array(predictions).copy()
    
    y = np.array(ground_truth).copy()
    
    if p.shape != y.shape:
        raise Exception("Error!  Both predictions and ground_truth must have same shape.")
    
    if len(p.shape) != 2:
        raise Exception("Error!  Both predictions and ground_truth must be 2D arrays.")
    
    
    
    total_entropy = 0
    
    for i in range(p.shape[0]):
        total_entropy += -np.sum(y[i,:] * np.nan_to_num( np.log(p[i,:]) ) + (1 - y[i,:]) * np.nan_to_num(np.log(1 - p[i,:])) )
        
    return total_entropy / p.size


DenseEvalCode = '''
#define _RELU(x) ( ((x) > 0.0f) ? (x) : 0.0f )
#define _SIGMOID(x)  ( 1.0f / (1.0f + expf(-(x)) ))


__global__ void dense_eval(int num_outputs, int num_inputs, int relu, int sigmoid, float * w, float * b, \
                           float * x, float *y, int batch_size, int w_t, int b_t, float delta)
{
     int i = blockDim.x*blockIdx.x + threadIdx.x;
     
     float old_w, old_b;
     

     /*if( w_t >= 0 )
     {
          old_w = w[w_t];
          w[w_t] += delta;
     }*/
     
     
     /*if( b_t >= 0 )
     {
          old_b = b[b_t];
          b[b_t] += delta;
     }*/
     
     
     if(batch_size <= 0)
         batch_size = 1;
     
        
     if (i < num_outputs)
     {
         for(int k=0; k < batch_size; k++)
         {    
              double temp = 0.0f;
              
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
         
            
            // modify this here
    if( w_t >= 0 && i == (w_t % num_outputs))
    {
          int j = w_t % num_inputs;
          
          for(int k=0; k < batch_size; k++)
              y[k*num_outputs + i] += delta*x[k*num_inputs+j];
              
          
    }
     
    if( b_t >= 0 && i == b_t )
    {
          int j = b_t % num_inputs;
          
          for(int k=0; k < batch_size; k++)
              y[k*num_outputs + i] += b_t; // delta*x[k*num_inputs+j];
    }
         
    return;
}
'''

eval_mod = SourceModule(DenseEvalCode)

eval_ker = eval_mod.get_function('dense_eval')


class DenseLayer:
    
    def __init__(self, num_inputs=None, num_outputs=None, weights=None, b=None, stream=None, \
    relu=False, sigmoid=False, dropout=None, delta=None):
        
        self.stream = stream
        
        if delta is None:
            self.delta = np.float32(0.001)
        else:
            self.delta = np.float32(delta)
        
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
            
        else:
            self.num_inputs = np.int32(num_inputs)
            self.num_outputs = np.int32(num_outputs)


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
        
        
        

    def eval_(self, x, y=None, batch_size=None, stream=None, delta=None, w_t = None, b_t = None):
    
        if type(x) != pycuda.gpuarray.GPUArray:
            x = gpuarray.to_gpu_async(np.array(x,dtype=np.float32) , stream=self.stream)
            
        if batch_size is None:
            if len(x.shape) == 2:
                batch_size = np.int32(x.shape[0])
            else:
                batch_size = np.int32(1)
                
        if delta is None:
            delta = self.delta
            
        if w_t is None:
            w_t = np.int32(-1)
            
        if b_t is None:
            b_t = np.int32(-1)
        
        
        if y is None:
            if batch_size == 1:
                y = gpuarray.empty((self.num_outputs,), dtype=np.float32)
            else:
                y = gpuarray.empty((batch_size, self.num_outputs), dtype=np.float32)


        eval_ker(self.num_outputs, self.num_inputs, self.relu, self.sigmoid, \
                 self.weights, self.b, x, y, np.int32(batch_size), w_t, b_t, \
                 delta , block=self.block, grid=self.grid , stream=stream)
        
        return y
        
        
    def eval_batch():
        pass
        
    def eval_batch_weight():
        pass
        
        
DropoutCode = '''
#include <curand_kernel.h>
#define ULL  unsigned long long

// dropout layer is for training
// use CuRAND here

extern "C" {

    __global__ void dropout_layer(int num, float *x, float *y, int batch_size, float prob)
    {
        
        
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (i < num)
        {
            curandState cr_state;
            
            curand_init( (ULL)  clock(), (ULL) i, (ULL) 0, &cr_state);
    
    
            for(int k =0; k < batch_size; k++)
            {
                if ( curand_uniform(&cr_state) <= prob)
                    y[k*num + i] = 0.0f;
                else
                    y[k*num + i] = x[k*num + i];
            }
        }
    }

}
'''  

dropout_mod = SourceModule(no_extern_c=True, source=DropoutCode)

dropout_ker = dropout_mod.get_function('dropout_layer')
        
class DropoutLayer:
    def __init__(self):
        pass
        
        


#exp_ker = ElementwiseKernel("float * x, float * y", "y[i] = expf(x[i]);",  "exp_ker")

# threads: at least "num"
SoftmaxExpCode='''
__global__ void softmax_exp( int num, float *x, float *y, int batch_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < num)
    {
        for (int k=0; k < batch_size; k++)
        {
            y[num*k + i] = expf(x[num*k+i]);
        
        }
    }
}
'''

exp_mod = SourceModule(SoftmaxExpCode)
exp_ker = exp_mod.get_function('softmax_exp')

# threads: at least batch size
SoftmaxMeanCode='''
__global__ void softmax_mean( int num, float *x, float *y, int batch_size)
{

    // parallelize over
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    
    if (i < batch_size)
    {
        float temp = 0.0f;
        
        for(int k=0; k < num; k++)
            temp += x[i*num + k];
            
        
        for(int k=0; k < num; k++)
            y[i*num+k] = x[i*num+k] / temp;
    
    }
    
    return;
}'''

mean_mod = SourceModule(SoftmaxMeanCode)
mean_ker = mean_mod.get_function('softmax_mean')

        
class SoftmaxLayer:
    def __init__(self, num=None, stream=None):
        self.num = np.int32(num)
        self.stream = None
        
        
    def eval_(self, x, y=None, batch_size=None, stream=None):

        if type(x) != pycuda.gpuarray.GPUArray:
            temp = np.array(x,dtype=np.float32)
            x = gpuarray.to_gpu_async( temp , stream=stream)
            
        if batch_size==None:
            if len(x.shape) == 2:
                batch_size = np.int32(x.shape[0])
            else:
                batch_size = np.int32(1)
        else:
            batch_size = np.int32(batch_size)
        
        
        if y is None:
            if batch_size == 1:
                y = gpuarray.empty((self.num,), dtype=np.float32)
            else:
                y = gpuarray.empty((batch_size, self.num), dtype=np.float32)

                
        exp_ker(self.num, x, y, batch_size, block=(32,1,1), grid=(int( np.ceil( self.num / 32) ), 1, 1), stream=stream)
        
        mean_ker(self.num, y, y, batch_size, block=(32,1,1), grid=(int( np.ceil( batch_size / 32)), 1,1), stream=stream)
    
        return y
    

    

class SequentialNetwork:

    def __init__(self, layers=None, delta=None, stream = None, max_batch_size=32, max_streams=10, epochs = 10):
        
        self.network = []
        self.network_summary = []
        self.network_mem = []
        
        if stream is not None:
            self.stream = stream
        else:
            self.stream = drv.Stream()
            
        self.delta = delta
        self.max_batch_size=max_batch_size
        
        self.max_streams = max_streams
        
        self.epochs = epochs
        
        if layers is not None:
            for layer in layers:
                add_layer(self, layer)
    
    def add_layer(self, layer):
    
        if layer['type'] == 'dense':
            if len(self.network) == 0:
                num_inputs = layer['num_inputs']
            else:
                num_inputs = self.network_summary[-1][2]
            
            num_outputs = layer['num_outputs']
            sigmoid = layer['sigmoid']
            relu = layer['relu']
            
            weights = layer['weights']
            
            b = layer['bias']
            
            self.network.append(DenseLayer(num_inputs=num_inputs, num_outputs=num_outputs, sigmoid=sigmoid, relu=relu, weights=weights, b=b))
            self.network_summary.append( ('dense', num_inputs, num_outputs))
            
            if self.max_batch_size > 1:
                if len(self.network_mem) == 0:
                    self.network_mem.append(gpuarray.empty( (self.max_batch_size, self.network_summary[-1][1] ), dtype=np.float32 ) )
                self.network_mem.append(gpuarray.empty((self.max_batch_size, self.network_summary[-1][2] ), dtype=np.float32  ) ) 
            else:
                if len(self.network_mem) == 0:
                    self.network_mem.append( gpuarray.empty( (self.network_summary[-1][1], ), dtype=np.float32 ) )
                self.network_mem.append( gpuarray.empty((self.network_summary[-1][2], ), dtype=np.float32  ) ) 
    
        elif layer['type'] == 'softmax':
            
            if len(self.network) == 0:
                raise Exception("Error!  Softmax layer can't be first!")
            
            if self.network_summary[-1][0] != 'dense':
                raise Exception("Error!  Need a dense layer before a softmax layer!")
            
            
            num = self.network_summary[-1][1]
            
            self.network.append(SoftmaxLayer(num=num))
            
            self.network_summary.append(('softmax', num, num))
            
            if self.max_batch_size > 1:
                self.network_mem.append(gpuarray.empty((self.max_batch_size, self.network_summary[-1][2] ), dtype=np.float32  ) ) 
            else:
                self.network_mem.append( gpuarray.empty((self.network_summary[-1][2], ), dtype=np.float32  ) ) 


            
    
    # assuming batch_size = 1
    def predict(self, x, stream=None):
        
        if stream is None:
            stream = self.stream
        
        #if(x.shape != self.network_mem)
        if type(x) != np.ndarray:
            temp = np.array(x, dtype = np.float32)
            x = temp
        
        if(x.size == self.network_mem[0].size):
            self.network_mem[0].set_async(x, stream=stream)
        else:
            
            if x.size > self.network_mem[0].size:
                raise Exception("Error: batch size too large for input.")
            
            x0 = np.zeros((self.network_mem[0].size,), dtype=np.float32)
            x0[0:x.size] = x.ravel()
            self.network_mem[0].set_async(x0.reshape(self.network_mem[0].shape), stream=stream)
        
        if(len(x.shape) == 2):
            batch_size = x.shape[0]
        else:
            batch_size = 1
        
        for i in xrange(len(self.network)):
            
            self.network[i].eval_(x=self.network_mem[i], y = self.network_mem[i+1], batch_size=batch_size, stream = stream)
            
        y = self.network_mem[-1].get_async(stream=stream)
        
        if len(y.shape) == 2:
            y = y[0:batch_size, :]
        
        return y
    
    
    
    # w_t :  weight to check
    #  (bank, weight)
    # b_t : bias to check
    # (bank, bias)
    
    def partial_predict(self, layer_index=None, w_t=None, b_t=None, partial_mem=None, stream=None, batch_size=None):
        
        self.network[layer_index].eval_(x=self.network_mem[layer_index], y = partial_mem[layer_index+1], batch_size=batch_size, stream = stream, w_t=w_t, b_t=b_t)
        
        for i in xrange(layer_index+1, len(self.network)):
            self.network[i].eval_(x=partial_mem[i], y =partial_mem[i+1], batch_size=batch_size, stream = stream)
            
        #y = partial_mem[-1]
        
        #return y
            
            
        
    # batch stochastic gradient descent
    
    def bsgd(self, training=None, training_rate=0.01, labels=None, delta=None, max_streams = None, epochs = 1):
        
        
        training = np.float32(training)
        labels = np.float32(labels)
        
        if( training.shape[0] != labels.shape[0] ):
            raise Exception("Number of training data points should be same as labels!")
        

        if max_streams is None:
            max_streams = self.max_streams
            
        if epochs is None:
            epochs = self.epochs
            
        if delta is None:
            delta = self.delta
        
        streams = []
        bgd_mem = []
        
        
        # create the streams needed for training
        
        for _ in xrange(max_streams):
            streams.append(drv.Stream())
            bgd_mem.append([])
            
        
        # allocate memory for each stream
        
        for i in xrange(len(bgd_mem)):
            for mem_bank in self.network_mem:
                bgd_mem[i].append( gpuarray.empty_like(mem_bank) )
        
        
        # begin training!
        
        num_points = training.shape[0]
        batch_size = self.max_batch_size
        
        for k in xrange(epochs):    
            
            print '-----------------------------------------------------------'
            print 'Starting training epoch: %s' % k
            print 'Batch size: %s , Total number of training samples: %s' % (batch_size, num_points)
            print '-----------------------------------------------------------'
            
            all_grad = []
            
            for _ in xrange( int(np.ceil(training.shape[0] / batch_size)) ):
            
                batch_index = np.random.choice(num_points, batch_size, replace=False)
                
                batch_training = training[batch_index, :]
                batch_labels = labels[batch_index, :]
                
                batch_predictions = self.predict(batch_training)
        
                cur_entropy = cross_entropy(predictions=batch_predictions, ground_truth=batch_labels)
                
                # need to iterate over each weight / bias , check entropy
                # def partial_predict(self, layer_index=None, w_t=None, b_t=None, partial_mem=None, stream=None, batch_size=None):
                
                
                
                
                for i in xrange(len(self.network)):
                    
                    if self.network_summary[0] != 'dense':
                        continue
                    
                    all_weights = Queue()
                    
                    grad_w = np.zeros((self.network[i].weights.size,), dtype=np.float32)
                    grad_b = np.zeros((self.network[i].b.size,), dtype=np.float32)
                    
                    for w in xrange( self.network[i].weights.size ):
                        all_weights.put( ('w', w ) )
                        
                    for b in xrange( self.network[i].b.size ):
                        all_weights.put(('b', b))
                        
                    while not all_weights.empty():
                        
                        stream_weights = Queue()
                        
                        for j in xrange(max_streams):
                            
                            if all_weights.empty():
                                break
                            
                            wb = all_weights.get()
                            
                            if wb[0] == 'w':
                                w_t = wb[1]
                                b_t = None
                            elif wb[0] == 'b':
                                b_t = wb[1]
                                w_t = None
                            
                            stream_weights.put( wb )
                            
                            self.partial_predict(layer_index=i, w_t=w_t, b_t=b_t, partial_mem=bgd_mem[j], stream=streams[j], batch_size=self.batch_size)
                            
                        for j in xrange(max_streams):
                            
                            if stream_weights.empty():
                                break
                            
                            wb = stream_weights.get()
                            
                            w_predictions = bgd_mem[j][-1].get_async(stream=streams[j])
                            
                            w_entropy = cross_entropy(predictions=w_predictions, ground_truth=batch_labels)
                            
                            
                            
                            if wb[0] == 'w':
                                w_t = wb[1]
                                # subtract entropy, divide by delta
                                grad_w[w_t] = (cur_entropy - w_entropy) / delta
                                
                            elif wb[0] == 'b':
                                b_t = wb[1]
                                grad_b[b_t] = (cur_entropy - w_entropy) / delta
                        
                    all_grad.append(grad_w)
                    all_grad.append(grad_b)
                        
                        
                            
                            

        
        
        
if __name__ == '__main_343424_':
    cross_entropy([[1,0]],[[1,0]])
        
                
                
        
    
if __name__ == '__main__':
    sn = SequentialNetwork( max_batch_size=10 )
    sn.add_layer({'type' : 'dense', 'num_inputs' : 2, 'num_outputs' : 3, 'relu': False, 'sigmoid': False, 'weights': [[1,2],[3,4],[5,6]], 'bias' : None })
    sn.add_layer({'type' : 'dense', 'num_inputs' : 3, 'num_outputs' : 2, 'relu': False, 'sigmoid': False, 'weights': [[1,2,3],[3,4, 5] ], 'bias' : None })
    x = np.float32([[1,1],[1,0]])
    y = sn.predict(x)
    
    print y
    
    sn.add_layer({'type' : 'dense', 'num_inputs' : 2, 'num_outputs' : 2, 'relu': False, 'sigmoid': False, 'weights': [[-1,0],[0,-1] ], 'bias' : None })
    x = np.float32([[1,1],[1,0]])
    y = sn.predict(x)

    print y
    
    sn.add_layer({'type' : 'softmax'})
    
    x = np.float32([[1,1],[1,0]])
    y = sn.predict(x)
    
    print y

