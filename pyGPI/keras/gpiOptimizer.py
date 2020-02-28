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
def comprexAllreduce(data, comprex, comm, tag=0, chief_rank=0):
    data_size = data.size
    data_shape = data.shape
    data = data.flatten()

    myRank = pyGPI.gaspi_context.getRank()
    numRanks = pyGPI.gaspi_context.getSize()
    #gaspi_printf("Rang %d of %d enters RingAllreduce"%(myRank, numRanks))

    # Allocate the output buffer.
    # Copy your data to the output buffer to avoid modifying the input buffer.
    output = np.array(data)

    # Chief Rank
    # reduces information, then sends it back
    #gaspi_printf("Entering Gather")
    if myRank == chief_rank:
        # Allocate a temporary buffer to store incoming data.
        buffer = np.ndarray([data_size])
        # reduce
        for rank in range(numRanks):
            if rank == chief_rank:
                continue
            #gaspi_printf("Read Remote size:%d, rank:%d, tag:%d"%(data_size, rank, tag))
            buffer = comprex.readRemote(rank, tag)
            #gaspi_printf("Read data:%s"%(str(buffer)))
            if(buffer.size != output.size):
                raise RuntimeError("Received Vector does not match size of accumulation vector (%d vs. %d)!"%(buffer.size, output.size))
            output += buffer
    else:
        # send own
        comprex.writeRemote(data, chief_rank, tag)

    #gaspi_printf("Gather finished")
    pyGPI.gaspi_context.barrier()

    if myRank == chief_rank:
        buffer = np.ndarray([data_size])
        # broadcast
        for rank in range(numRanks):
            if rank == chief_rank:
                continue
            comm.writeRemote(output, rank, tag)
    # Worker Rank
    else:
        # receive new
        output = np.array(comm.readRemote(data_size, chief_rank, tag))

    # wait until update is received
    pyGPI.gaspi_context.barrier()

    '''
    # Chief Rank
    # reduces information, then sends it back
    if myRank == chief_rank:
        # Allocate a temporary buffer to store incoming data.
        buffer = np.ndarray([data_size])
        # reduce
        for rank in range(numRanks):
            if rank == chief_rank:
                continue
            buffer = np.array(comprex.readRemote(rank, tag))
            if(buffer.size != output.size):
                raise RuntimeError("Received Vector does not match size of accumulation vector (%d vs. %d)!"%(buffer.size, output.size))
            output += buffer

        # broadcast
        for rank in range(numRanks):
            if rank == chief_rank:
                continue
            comm.writeRemote(output, rank, tag)
    # Worker Rank
    else:
        # send own
        comprex.writeRemote(data, chief_rank, tag)
        # receive new
        output = np.array(comm.readRemote(data_size, chief_rank, tag))

    # wait until update is received
    #gaspi_context.barrier()
    '''
    return np.reshape(output,data_shape)


################################################################################################
## ringAllreduce
## ring reduction with compression-less communicator
################################################################################################
# adapted from https://github.com/baidu-research/baidu-allreduce/blob/master/collectives.cu
@profile
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

    #myGaspi_context = Gpi.Gaspi_Context()

    # feedback_comm = GaspiEx.GaspiEx(pyGPI.gaspi_runtime.get(), myGaspi_context.get(), pyGPI.gaspi_segment.get())
    # comm = GaspiEx.GaspiEx(pyGPI.gaspi_runtime.get(), myGaspi_context.get(), pyGPI.gaspi_segment.get())
    # if myRank%2==0:
    #     feedback_comm.connectTx(recv_from, 1, feedbackTag)
    #     comm.connectTx(send_to, size, commTag)
    # else:
    #     feedback_comm.connectRx(send_to, 1, feedbackTag)
    #     comm.connectRx(recv_from, size, commTag)
    
    # if myRank%2==1:
    #     feedback_comm.connectTx(recv_from, 1, feedbackTag)
    #     comm.connectTx(send_to, size, commTag)
    # else:
    #     feedback_comm.connectRx(send_to, 1, feedbackTag)
    #     comm.connectRx(recv_from, size, commTag)
    # gaspi_printf("Connected layer %d"%tag)


    # reverse recv and send rank for feedback lines
    feedback_comm = GaspiEx.GaspiEx(pyGPI.gaspi_runtime.get(), pyGPI.gaspi_context.get(), pyGPI.gaspi_segment.get())
    feedback_comm.connectTo(send_to, recv_from, 1, feedbackTag)

    # data communication line
    comm = GaspiEx.GaspiEx(pyGPI.gaspi_runtime.get(), pyGPI.gaspi_context.get(), pyGPI.gaspi_segment.get())
    comm.connectTo(recv_from, send_to, size, commTag)
    
    def func(data):
        return ringAllreduce(data, comm, feedback_comm, tag)
    return tf.py_func(func=func, inp=[data], Tout=data.dtype, name="gpi_ringAllreduce")


def gpi_comprexAllreduce(data, tag=0, chief_rank=0):
    grad_size= np.prod(data.shape)
    comprex=ComprEx.Comprex(pyGPI.gaspi_runtime.get(), pyGPI.gaspi_context.get(), pyGPI.gaspi_segment.get(), grad_size)
    threshold = ComprEx.ThresholdTopK(0.01)
    compressor = ComprEx.CompressorRLE()
    comprex.setThreshold(threshold.get())
    comprex.setCompressor(compressor.get())
    comm = GaspiEx.GaspiEx(pyGPI.gaspi_runtime.get(), pyGPI.gaspi_context.get(), pyGPI.gaspi_segment.get())
    def func(data):
        #comprex.resetRests()
        return comprexAllreduce(data, comprex, comm, tag=tag, chief_rank=chief_rank)
    return tf.py_func(func=func, inp=[data], Tout=data.dtype, name="gpi_comprexAllreduce")


# from horovod/_keras/__init__.py
def create_distributed_optimizer(optimizer, name=None):
    class _DistributedOptimizer(keras.optimizers.Optimizer):
        def __init__(self, name, config):
            if name is None:
                name = "Distributed%s" % self.__class__.__base__.__name__
            self._name = name
            #self.comm = GaspiEx.GaspiEx(pyGPI.gaspi_runtime.get(), pyGPI.gaspi_context.get(), pyGPI.gaspi_segment.get())
            self._get_gradients_used = False
            super(self.__class__, self).__init__(**config)
            #self.communicator = []


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
                            #self.communicator.append( ComprEx.Comprex(pyGPI.gaspi_runtime.get(), pyGPI.gaspi_context.get(), pyGPI.gaspi_segment.get(), grad_size) )
                            if grad_size != 0:
                                avg_grad = gpi_ringAllreduce(grad, tag=iter)#self.communicator[iter])
                            #avg_grad = gpi_comprexAllreduce(grad, tag=iter, chief_rank=0)
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
    cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
               dict(_DistributedOptimizer.__dict__))
    return cls(name, optimizer.get_config())