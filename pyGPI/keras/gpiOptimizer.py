import numpy as np
import tensorflow.keras as keras
import tensorflow as tf

import pyGPI
import pyGPI.ComprEx as ComprEx
from pyGPI.Gpi import gaspi_printf
import pyGPI.Gpi as Gpi


################################################################################################
## comprexAllreduce
## simple gather-broadcast based reduction with comprex as communicator
################################################################################################
def comprexAllreduce(data, comprex_map, comm_map, chief_rank):
    data_size = data.size
    data_shape = data.shape
    data = data.flatten()
    output = data.copy()

    myRank = pyGPI.gaspi_context.getRank()
    numRanks = pyGPI.gaspi_context.getSize()

    # Phase 1: Collect all updates
    # reduces information, then send it back
    if myRank == chief_rank:
        # chief rank: reduce
        for rank in range(numRanks):
            if rank == chief_rank:
                continue
            buffer = comprex_map[rank].readRemote()
            if(buffer.size != output.size):
                raise RuntimeError("Received Vector does not match size of accumulation vector (%d vs. %d)!"%(buffer.size, output.size))
            # accumulate
            output += buffer
    else:
        # worker: send
        comprex_map[chief_rank].writeRemote(data)

    # Phase 2: Send reduced update back
    if myRank == chief_rank:
        # chief: send back
        for rank in range(numRanks):
            if rank == chief_rank:
                continue
            comm_map[rank].writeRemote(output)
    else:
        # worker: receive new update
        output = comm_map[chief_rank].readRemote()

    output = np.reshape(output, data_shape)
    return output


def gpi_comprexAllreduce(data, tag=0, chief_rank=0):
    size= data.shape.num_elements()
    myRank = pyGPI.gaspi_context.getRank()
    numRanks = pyGPI.gaspi_context.getSize()
    # TODO: the topK value is hard coded here for now
    topK = 1e-2

    comprexTag = int(tag*2*numRanks)
    commTag = int(tag*2*numRanks+1)  

    comm_map = {}
    comprex_map = {}

    # connection for forward communication is all to one
    if myRank==chief_rank:
        for i in range(numRanks):
            if i==chief_rank:
                continue
            gaspi_printf("Connecting Rx Comprex with rank %d\n"%(i))
            comprex=ComprEx.Comprex(pyGPI.gaspi_runtime, pyGPI.gaspi_context, pyGPI.gaspi_segment, size)
            threshold = ComprEx.ThresholdTopK(topK) # does not matter
            compressor = ComprEx.CompressorRLE()
            comprex.setThreshold(threshold)
            comprex.setCompressor(compressor)
            comprex.connectRx(i, comprexTag+i, size_factor=2)
            comprex_map[i] = comprex
    else:
        gaspi_printf("Connecting Tx Comprex with rank %d\n"%(chief_rank))
        comprex=ComprEx.Comprex(pyGPI.gaspi_runtime, pyGPI.gaspi_context, pyGPI.gaspi_segment, size)
        threshold = ComprEx.ThresholdTopK(topK)
        compressor = ComprEx.CompressorRLE()
        comprex.setThreshold(threshold)
        comprex.setCompressor(compressor)
        comprex.connectTx(chief_rank, comprexTag+myRank, size_factor=2)
        comprex_map[chief_rank] = comprex
        
        
    # connection for backward communication is one to all
    if myRank==chief_rank:
        for i in range(numRanks):
            if i==chief_rank:
                continue
            comm=ComprEx.Comprex(pyGPI.gaspi_runtime, pyGPI.gaspi_context, pyGPI.gaspi_segment, size)
            threshold = ComprEx.ThresholdTopK(topK)
            compressor = ComprEx.CompressorRLE()
            comm.setThreshold(threshold)
            comm.setCompressor(compressor)
            comm.connectTx(i, commTag+i, size_factor=2)
            comm_map[i] = comm
    else:
        comm=ComprEx.Comprex(pyGPI.gaspi_runtime, pyGPI.gaspi_context, pyGPI.gaspi_segment, size)
        threshold = ComprEx.ThresholdTopK(topK) # does not matter
        compressor = ComprEx.CompressorRLE()
        comm.setThreshold(threshold)
        comm.setCompressor(compressor)
        comm.connectRx(chief_rank, commTag+myRank, size_factor=2)
        comm_map[chief_rank] = comm

    def func(data):
        return comprexAllreduce(data, comprex_map, comm_map, chief_rank)
    return tf.py_func(func=func, inp=[data], Tout=data.dtype, name="gpi_comprexAllreduce")


# adapted from horovod/_keras/__init__.py
def create_distributed_optimizer(optimizer, name=None):
    class _DistributedOptimizer(keras.optimizers.Optimizer):
        def __init__(self, name, config):
            if name is None:
                name = "Distributed%s" % self.__class__.__base__.__name__
            self._name = name
            self._get_gradients_used = False
            super(self.__class__, self).__init__(**config)

        def get_gradients(self, loss, params):
            """
            Compute gradients of all trainable variables.
            See Optimizer.get_gradients() for more info.
            In DistributedOptimizer, get_gradients() is overriden to also
            allreduce the gradients before returning them.
            """
            self._get_gradients_used = True
            gradients = super(self.__class__, self).get_gradients(loss, params)
            if pyGPI.gaspi_context.getSize() > 1:
                averaged_gradients = []
                with tf.name_scope(self._name + "_Allreduce"):
                    for iter,grad in enumerate(gradients):
                        if grad is not None:
                            if isinstance(grad, tf.IndexedSlices):
                                grad = tf.convert_to_tensor(grad)
                            grad_size = grad.shape.num_elements() or 0
                            #gaspi_printf("Gradient size of layer %d=%d"%(iter, grad_size))
                            gaspi_printf("Layer %d: %s"%(iter, str(grad)))
                            if grad_size != 0:
                                # avg_grad = gpi_ringAllreduce(grad, tag=iter)
                                avg_grad = gpi_comprexAllreduce(grad, tag=iter, chief_rank=0)
                            else:
                                avg_grad = grad
                            averaged_gradients.append(avg_grad)
                        else:
                            averaged_gradients.append(None)
                    return averaged_gradients
            else:
                return gradients

        def apply_gradients(self, *args, **kwargs):
            if not self._get_gradients_used:
                  raise Exception('`apply_gradients()` was called without a call to '
                                  '`get_gradients()`. If you\'re using TensorFlow 2.0, '
                                  'please specify `experimental_run_tf_function=False` in '
                                  '`compile()`.')
            return super(self.__class__, self).apply_gradients(*args, **kwargs)

    # We dynamically create a new class that inherits from the optimizer that was passed in.
    # The goal is to override get_gradients() method with an allreduce implementation.
    # This class will have the same name as the optimizer it's wrapping, so that the saved
    # model could be easily restored without Horovod.
    cls = type(optimizer.__class__.__name__, (optimizer.__class__,),dict(_DistributedOptimizer.__dict__))
    obj = cls(name, optimizer.get_config())
    return obj