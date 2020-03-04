import numpy as np
import tensorflow.keras as keras
import tensorflow as tf

import pyGPI
import pyGPI.GaspiEx as GaspiEx
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

    myRank = pyGPI.gaspi_context.getRank()
    numRanks = pyGPI.gaspi_context.getSize()
    #gaspi_printf("Rang %d of %d enters RingAllreduce"%(myRank, numRanks))

    # Allocate the output buffer.
    # Copy your data to the output buffer to avoid modifying the input buffer.
    output = data

    # Chief Rank
    # reduces information, then sends it back
    #gaspi_printf("Entering Gather")
    if myRank == chief_rank:
        # Allocate a temporary buffer to store incoming data.
        # buffer = np.ndarray([data_size])
        # reduce
        for rank in range(numRanks):
            if rank == chief_rank:
                continue
            #gaspi_printf("Read Remote size:%d, rank:%d, tag:%d"%(data_size, rank, tag))
            buffer = comprex_map[rank].readRemote()
            #gaspi_printf("Read data:%s"%(str(buffer)))
            if(buffer.size != output.size):
                raise RuntimeError("Received Vector does not match size of accumulation vector (%d vs. %d)!"%(buffer.size, output.size))
            output += buffer
    else:
        # send own
        comprex_map[chief_rank].writeRemote(data)

    #gaspi_printf("Gather finished")
    #pyGPI.gaspi_context.barrier()

    if myRank == chief_rank:
        #buffer = np.ndarray([data_size])
        # broadcast
        for rank in range(numRanks):
            if rank == chief_rank:
                continue
            #comm_map[rank].writeRemote(output, 0, 0)
            comm_map[rank].writeRemote(output)
    # Worker Rank
    else:
        # receive new
        #output = comm_map[chief_rank].readRemote(data_size, 0, 0)
        output = comm_map[chief_rank].readRemote()

    # wait until update is received
    #pyGPI.gaspi_context.barrier()
    #gaspi_printf("Broadcast finished")

    output = np.reshape(output, data_shape)
    return output


################################################################################################
## ringAllreduce
## ring reduction with compression-less communicator
################################################################################################
# adapted from https://github.com/baidu-research/baidu-allreduce/blob/master/collectives.cu
#@profile
def ringAllreduce(data, comm, feedback_comm, tag=0):
    #check if data_length and comprex size are equal
    data_shape = data.shape
    data = data.flatten()
    data_size = data.size

    myRank = pyGPI.gaspi_context.getRank()
    numRanks = pyGPI.gaspi_context.getSize()
    # gaspi_printf("Rang %d of %d enters RingAllreduce"%(myRank, numRanks))
    #print("Data on Rank %d: %s"%(myRank, str(data)))
    # Partition the elements of the array into N approximately equal-sized chunks, where N is the MPI size.
    segment_size = data_size // numRanks
    segment_sizes = [segment_size]*numRanks

    segment_residual = data_size % numRanks
    for i in range(segment_residual):
        segment_sizes[i] += 1

    # Compute where each chunk ends.
    segment_ends = [segment_sizes[0]]
    for i in range(numRanks-1):
        segment_ends.append(segment_sizes[i+1]+segment_ends[i])

    # The last segment should end at the very end of the buffer.
    if(segment_ends[-1] != data_size):
        raise RuntimeError("segment_ends[-1] should be data_size!")

    # Allocate the output buffer.
    # Copy your data to the output buffer to avoid modifying the input buffer.
    output = np.array(data, dtype=data.dtype)
    # Allocate a temporary buffer to store incoming data.
    # We know that segment_sizes[0] is going to be the largest buffer size,
    # because if there are any overflow elements at least one will be added to
    # the first segment.
    buffer = np.ndarray([segment_sizes[0]], dtype=data.dtype)

    # Receive from your left neighbor with wrap-around.
    recv_from = (myRank - 1 + numRanks) % numRanks

    # Send to your right neighbor with wrap-around.
    send_to = (myRank + 1) % numRanks

    # Feedback synchronization
    feedback_message = np.array([tag], dtype=data.dtype)
    # if myRank%2==0:
    #     feedback_comm.writeRemote(feedback_message, recv_from, tag+1)
    #     token = feedback_comm.readRemote(1, send_to, tag+1) #blocking!
    # else:
    #     token = feedback_comm.readRemote(1, send_to, tag+1) #blocking!
    #     feedback_comm.writeRemote(feedback_message, recv_from, tag+1)

    # Now start ring. At every step, for every rank, we iterate through
    # segments with wraparound and send and recv from our neighbors and reduce
    # locally. At the i'th iteration, sends segment (rank - i) and receives
    # segment (rank - i - 1).

    # synchronize all ranks before starting the scatter
    # pyGPI.gaspi_context.barrier()

    # gaspi_printf("%d Reducing Ring"%tag)
    for i in range(numRanks-1):
        recv_chunk = (myRank - i - 1 + numRanks) % numRanks
        send_chunk = (myRank - i + numRanks) % numRanks
        segment_send_start = segment_ends[send_chunk] - segment_sizes[send_chunk]
        segment_send = output[segment_send_start:segment_ends[send_chunk]] #.tolist()
        segment_size = segment_sizes[recv_chunk]

        
        comm.writeRemote(segment_send, send_to, tag)
        comm.readRemote_vec(buffer, segment_size, recv_from, tag)
        # if myRank%2==0:
        #     comm.writeRemote(segment_send, send_to, tag)
        #     buffer = comm.readRemote(segment_size, recv_from, tag)
        # else:
        #     buffer = comm.readRemote(segment_size, recv_from, tag)
        #     comm.writeRemote(segment_send, send_to, tag)

        segment_recv_start = segment_ends[recv_chunk] - segment_sizes[recv_chunk]

        # wait until update is received
        # pyGPI.gaspi_context.barrier()

        # reduce
        output[ segment_recv_start:segment_ends[recv_chunk] ] += buffer[0:segment_sizes[recv_chunk]]

        # synchronization
        feedback_comm.writeRemote(feedback_message, recv_from, tag+1)
        token = feedback_comm.readRemote(1, send_to, tag+1) #blocking!


    # synchronize all ranks before starting the gather
    # pyGPI.gaspi_context.barrier()
    # feedback_comm.writeRemote(feedback_message, recv_from, tag+1)
    # token = feedback_comm.readRemote(1, send_to, tag+1) #blocking!
    # Now start pipelined ring allgather. At every step, for every rank, we
    # iterate through segments with wraparound and send and recv from our
    # neighbors. At the i'th iteration, rank r, sends segment (rank + 1 - i)
    # and receives segment (rank - i).
    # gaspi_printf("%d Broadcast Ring"%tag)
    for i in range(numRanks-1):
        send_chunk = (myRank - i + 1 + numRanks) % numRanks
        recv_chunk = (myRank - i + numRanks) % numRanks
        # Segment to send - at every iteration we send segment (r+1-i)
        segment_send_start = segment_ends[send_chunk] - segment_sizes[send_chunk]
        segment_send = output[segment_send_start:segment_ends[send_chunk]] #.tolist()
        segment_size = segment_sizes[recv_chunk]

        # Segment to recv - at every iteration we receive segment (r-i)
        segment_recv_start = segment_ends[recv_chunk] - segment_sizes[recv_chunk]

        comm.writeRemote(segment_send, send_to, tag)
        comm.readRemote_vec(buffer, segment_size, recv_from, tag)

        # if myRank%2==0:
        #     comm.writeRemote(segment_send, send_to, tag)
        #     buffer = comm.readRemote(segment_size, recv_from, tag)
        # else:
        #     buffer = comm.readRemote(segment_size, recv_from, tag)
        #     comm.writeRemote(segment_send, send_to, tag)
        # gaspi_printf("Send token...")
        
        # gaspi_printf("Send segment...")
        # comm.writeRemote(segment_send, send_to, tag)
        # gaspi_printf("Receive segment...")
        # buffer = comm.readRemote(segment_size, recv_from, tag)

        # wait until update is received
        #pyGPI.gaspi_context.barrier()

        # write
        output[ segment_recv_start:segment_ends[recv_chunk] ] = buffer[0:segment_sizes[recv_chunk]]
        # gaspi_printf("Wait to receive token...")
        feedback_comm.writeRemote(feedback_message, recv_from, tag+1)
        token = feedback_comm.readRemote(1, send_to, tag+1) #blocking!
        # gaspi_printf("Token %d received (expected %d)"%(token,tag+1))

    # pyGPI.gaspi_context.barrier()
    return np.reshape(output,data_shape)


def gpi_ringAllreduce(data, tag=0):
    myRank = pyGPI.gaspi_context.getRank()
    numRanks = pyGPI.gaspi_context.getSize()
    # Receive from your left neighbor with wrap-around.
    recv_from = (myRank - 1 + numRanks) % numRanks
    # Send to your right neighbor with wrap-around.
    send_to = (myRank + 1) % numRanks

    size = data.shape.num_elements()

    feedbackTag = int(tag*2)
    commTag = int(tag*2+1)

    # reverse recv and send rank for feedback lines
    feedback_comm = GaspiEx.GaspiEx(pyGPI.gaspi_runtime, pyGPI.gaspi_context, pyGPI.gaspi_segment)
    feedback_comm.connectTo(send_to, recv_from, 1, feedbackTag)

    # data communication line
    comm = GaspiEx.GaspiEx(pyGPI.gaspi_runtime, pyGPI.gaspi_context, pyGPI.gaspi_segment)
    comm.connectTo(recv_from, send_to, size, commTag)
    
    def func(data):
        return ringAllreduce(data, comm, feedback_comm, tag)
    return tf.py_func(func=func, inp=[data], Tout=data.dtype, name="gpi_ringAllreduce")


def gpi_comprexAllreduce(data, tag=0, chief_rank=0):
    size= data.shape.num_elements()
    myRank = pyGPI.gaspi_context.getRank()
    numRanks = pyGPI.gaspi_context.getSize()
    compression_ratio = 1e-2

    comprexTag = int(tag*2*numRanks)
    commTag = int(tag*2*numRanks+1)  

    comm_map = {}
    comprex_map = {}

    # connection for comprex is all to one
    if myRank==chief_rank:
        for i in range(numRanks):
            if i==chief_rank:
                continue
            gaspi_printf("Connecting Rx Comprex with rank %d\n"%(i))
            comprex=ComprEx.Comprex(pyGPI.gaspi_runtime, pyGPI.gaspi_context, pyGPI.gaspi_segment, size)
            threshold = ComprEx.ThresholdTopK(compression_ratio) # does not matter
            compressor = ComprEx.CompressorRLE()
            comprex.setThreshold(threshold)
            comprex.setCompressor(compressor)
            comprex.connectRx(i, comprexTag+i, size_factor=2)
            comprex_map[i] = comprex
    else:
        gaspi_printf("Connecting Tx Comprex with rank %d\n"%(chief_rank))
        comprex=ComprEx.Comprex(pyGPI.gaspi_runtime, pyGPI.gaspi_context, pyGPI.gaspi_segment, size)
        threshold = ComprEx.ThresholdTopK(compression_ratio)
        compressor = ComprEx.CompressorRLE()
        comprex.setThreshold(threshold)
        comprex.setCompressor(compressor)
        comprex.connectTx(chief_rank, comprexTag+myRank, size_factor=2)
        comprex_map[chief_rank] = comprex
        
        
    # connection for comm is one to all
    if myRank==chief_rank:
        for i in range(numRanks):
            if i==chief_rank:
                continue
            comm=ComprEx.Comprex(pyGPI.gaspi_runtime, pyGPI.gaspi_context, pyGPI.gaspi_segment, size)
            threshold = ComprEx.ThresholdTopK(compression_ratio)
            compressor = ComprEx.CompressorRLE()
            comm.setThreshold(threshold)
            comm.setCompressor(compressor)
            comm.connectTx(i, commTag+i, size_factor=2)
            comm_map[i] = comm
    else:
        comm=ComprEx.Comprex(pyGPI.gaspi_runtime, pyGPI.gaspi_context, pyGPI.gaspi_segment, size)
        threshold = ComprEx.ThresholdTopK(compression_ratio) # does not matter
        compressor = ComprEx.CompressorRLE()
        comm.setThreshold(threshold)
        comm.setCompressor(compressor)
        comm.connectRx(chief_rank, commTag+myRank, size_factor=2)
        comm_map[chief_rank] = comm

    def func(data):
        #comprex.resetRests()
        return comprexAllreduce(data, comprex_map, comm_map, chief_rank)
    return tf.py_func(func=func, inp=[data], Tout=data.dtype, name="gpi_comprexAllreduce")


# from horovod/_keras/__init__.py
def create_distributed_optimizer(optimizer, name=None):
    class _DistributedOptimizer(keras.optimizers.Optimizer):
        def __init__(self, name, config):
            gaspi_printf("Creating distributed optimizer...")
            if name is None:
                name = "Distributed%s" % self.__class__.__base__.__name__
            self._name = name
            self._get_gradients_used = False
            super(self.__class__, self).__init__(**config)
            gaspi_printf("created distributed optimizer")

        def get_gradients(self, loss, params):
            """
            Compute gradients of all trainable variables.
            See Optimizer.get_gradients() for more info.
            In DistributedOptimizer, get_gradients() is overriden to also
            allreduce the gradients before returning them.
            """
            gaspi_printf("Building Gradients")
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