import sys
sys.path.append('..')
import ComprEx
import GaspiEx
import Gpi as gpi
from Gpi import gaspi_printf
import ctypes
import numpy as np

import cProfile, pstats
from timeit import default_timer as timer

# initialize GASPI
global gaspi_runtime
gaspi_runtime = gpi.Gaspi_Runtime()
global gaspi_context
gaspi_context = gpi.Gaspi_Context()


def print_vector(label, array):
    string=label+": ["
    for i in range(len(array)):
        string = string + "%.2f, "%(array[i])
    string = string[:-2] + "]\n"
    gaspi_printf(string)


def comprexAllreduce(data, comprex, comm, chief_rank=0, tag=0):
    data_size = data.size

    myRank = gaspi_context.getRank()
    numRanks = gaspi_context.getSize()
    #gaspi_printf("Rang %d of %d enters RingAllreduce"%(myRank, numRanks))

    # Allocate the output buffer.
    # Copy your data to the output buffer to avoid modifying the input buffer.
    output = np.array(data)

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
    else:
        # send own
        comprex.writeRemote(data, chief_rank, tag)

    gaspi_context.barrier()

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
    gaspi_context.barrier()

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
    return output

# adapted from https://github.com/baidu-research/baidu-allreduce/blob/master/collectives.cu
def ringAllreduce(data, comm, tag=0):

    #check if data_length and comprex size are equal
    '''
    if(data.size != cmprex.getSize()):
        raise RuntimeError("Data size is not equal Comprex communicator size!")
    data_size = cmprex.getSize()
    '''
    data_size = data.size

    myRank = gaspi_context.getRank()
    numRanks = gaspi_context.getSize()
    #gaspi_printf("Rang %d of %d enters RingAllreduce"%(myRank, numRanks))

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
    output = np.array(data, dtype=np.float32)
    # Allocate a temporary buffer to store incoming data.
    # We know that segment_sizes[0] is going to be the largest buffer size,
    # because if there are any overflow elements at least one will be added to
    # the first segment.
    buffer = np.ndarray([segment_sizes[0]])

    # Receive from your left neighbor with wrap-around.
    recv_from = (myRank - 1 + numRanks) % numRanks

    # Send to your right neighbor with wrap-around.
    send_to = (myRank + 1) % numRanks

    # Now start ring. At every step, for every rank, we iterate through
    # segments with wraparound and send and recv from our neighbors and reduce
    # locally. At the i'th iteration, sends segment (rank - i) and receives
    # segment (rank - i - 1).
    gaspi_printf("Reducing Ring")
    for i in range(numRanks-1):
        recv_chunk = (myRank - i - 1 + numRanks) % numRanks
        send_chunk = (myRank - i + numRanks) % numRanks
        segment_send_start = segment_ends[send_chunk] - segment_sizes[send_chunk]
        segment_send = output[segment_send_start:segment_ends[send_chunk]] #.tolist()
        segment_size = segment_sizes[recv_chunk]
        if myRank%2==0:
            comm.writeRemote(segment_send, send_to, tag)
            buffer = np.array(comm.readRemote(segment_size, recv_from, tag))
        else:
            buffer = np.array(comm.readRemote(segment_size, recv_from, tag))
            comm.writeRemote(segment_send, send_to, tag)

        segment_recv_start = segment_ends[recv_chunk] - segment_sizes[recv_chunk]

        # wait until update is received
        gaspi_context.barrier()

        # reduce
        output[ segment_recv_start:segment_ends[recv_chunk] ] += buffer[0:segment_sizes[recv_chunk]]

    # Now start pipelined ring allgather. At every step, for every rank, we
    # iterate through segments with wraparound and send and recv from our
    # neighbors. At the i'th iteration, rank r, sends segment (rank + 1 - i)
    # and receives segment (rank - i).
    gaspi_printf("Broadcast Ring")
    for i in range(numRanks-1):
        send_chunk = (myRank - i + 1 + numRanks) % numRanks
        recv_chunk = (myRank - i + numRanks) % numRanks
        # Segment to send - at every iteration we send segment (r+1-i)
        segment_send_start = segment_ends[send_chunk] - segment_sizes[send_chunk]
        segment_send = output[segment_send_start:segment_ends[send_chunk]] #.tolist()
        segment_size = segment_sizes[recv_chunk]

        # Segment to recv - at every iteration we receive segment (r-i)
        segment_recv_start = segment_ends[recv_chunk] - segment_sizes[recv_chunk]

        if myRank%2==0:
            comm.writeRemote(segment_send, send_to, tag)
            buffer = np.array(comm.readRemote(segment_size, recv_from, tag))
        else:
            buffer = np.array(comm.readRemote(segment_size, recv_from, tag))
            comm.writeRemote(segment_send, send_to, tag)

        # wait until update is received
        gaspi_context.barrier()

        # write
        output[ segment_recv_start:segment_ends[recv_chunk] ] = buffer[0:segment_sizes[recv_chunk]]
    return output

# number of elements in array
size = 1000000

# number of send/receive cycles
num_runs = 1

# define GASPI roles
srcRank = 0
destRank = 1

# communication tag
tag = 1


print("Transmit values from rank 0 to rank 1. Rank 1 accumulates values.")
print("Use Gaspi Logger to intermediate outputs. Start it with \'gaspi_logger&\' in your console.")

# get local rank and size
myRank = gaspi_context.getRank()
numRanks = gaspi_context.getSize()

# set profiler
# only profile on first rank!
if myRank == srcRank:
    profiler = cProfile.Profile()
    profiler.enable()
    start_time = timer()

# getting memory for GASPI
gaspi_segment = gpi.Gaspi_Segment(2**27) # 64 MB

# create threshold
threshold = ComprEx.ThresholdTopK(1.0)
gaspi_printf("Using ThresholdTopK")

# create compressor
compressor = ComprEx.CompressorRLE()
gaspi_printf("Using CompressorRLE")

# create Comprex
comprex = ComprEx.Comprex(gaspi_runtime.get(), gaspi_context.get(), gaspi_segment.get(), size)
comprex.setThreshold(threshold.get())
comprex.setCompressor(compressor.get())

comm = GaspiEx.GaspiEx(gaspi_runtime.get(), gaspi_context.get(), gaspi_segment.get())
gaspi_printf("Building communicator")
gaspi_printf("Maximum number of elements for gaspi_allreduce: %d"%gpi.gaspi_allreduce_elem_max())

# create input data
init_values = [0]*size
for i in range(size):
    init_values[i] = i * (1-2*(i%2))
init_values = np.array(init_values, dtype=np.float32)
if(myRank == destRank):
    # create buffer for accumulation
    accu = [0]*size
    gold_result = [0]*size
    for i in range(size):
        gold_result[i] = init_values[i]*numRanks

# main transmission loop
for run in range(num_runs):
    values = init_values
    # Source side
    # ------------------------------------------------------------
    if(myRank == srcRank):
        pass
        #gaspi_printf("Run #%d"%run)
        #gaspi_printf("=========================================")
    #print_vector("Vector", values)
    #values = comprexAllreduce(values, comprex, comm)
    values = ringAllreduce(values, comm)
    #values = gpi.gaspi_allreduce_floatsum(values, gaspi_context.get())
    #gaspi_context.barrier()

# check if transmitted values add up to correct values:
if(myRank == destRank):
    errors=0
    for i in range(size):
        if(values[i] != gold_result[i]):
            errors += 1

    #print("Values is:")
    #print(values)
    #print("expected:")
    #print(gold_result)
    if(errors==0):
        print("+++PASSED+++")
    else:
        print("---FAILED---")
        print("with %d errors."%errors)

gaspi_printf("Done.")


if myRank == srcRank:
    end_time = timer()
    profiler.disable()

    print("Timer time: %f s"%(end_time-start_time))
    profiler.create_stats()
    profiler.dump_stats("allreduce_rank%02d.cprof"%myRank)

    sortby = 'cumulative'
    ps = pstats.Stats(profiler).sort_stats(sortby)
    ps.print_stats()
