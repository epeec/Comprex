import sys
#sys.path.append('/u/l/loroch/PROGRAMMING/comprex/')
import ComprEx as comprex
import pyGPI as gpi
from pyGPI import gaspi_printf
import ctypes
import numpy as np

def print_vector(label, array):
    string=label+": ["
    for i in range(len(array)):
        string = string + "%.2f, "%(array[i])
    string = string[:-2] + "]\n"
    gaspi_printf(string)



# number of elements in array
size = 50

# number of send/receive cycles
num_runs = 3

# define GASPI roles
srcRank = 0
destRank = 1

# communication tag
tag = 1

# initialize GASPI
gaspi_runtime = gpi.Gaspi_Runtime()

with gpi.Gaspi_Context() as gaspi_context:

    print("Transmit values from rank 0 to rank 1. Rank 1 accumulates values.")
    print("Use Gaspi Logger to intermediate outputs. Start it with \'gaspi_logger&\' in your console.")

    # get local rank
    myRank = gaspi_context.getRank()
    print( "Rank %d of %d"%(myRank, gaspi_context.getSize()) )

    #gaspi_context = gpi.Gaspi_Context()
    # getting memory for GASPI
    gaspi_segment = gpi.Gaspi_Segment(2**10) # 1 MB

    # create threshold
    threshold = comprex.ThresholdTopK(0.2)
    gaspi_printf("Using ThresholdTopK")

    # create compressor
    compressor = comprex.CompressorRLE()
    gaspi_printf("Using CompressorRLE")

    # create Comprex
    cmprex = comprex.Comprex(gaspi_runtime.get(), gaspi_context.get(), gaspi_segment.get(), size)
    cmprex.setThreshold(threshold.get())
    cmprex.setCompressor(compressor.get())
    gaspi_printf("Building comprex")


    # create input data
    values = np.ndarray([size], dtype=np.float32)
    for i in range(size):
        values[i] = i * (1-2*(i%2))

    if(myRank == destRank):
        # create buffer for accumulation
        accu = np.ndarray([size], dtype=np.float32)
        gold_result = np.ndarray([size], dtype=np.float32)
        for i in range(size):
            accu[i]=0
            gold_result[i] = values[i]*num_runs

    # main transmission loop
    # +1 runs for flushing rests
    for run in range(num_runs+1):
        gaspi_context.barrier()

        # Source side
        # ------------------------------------------------------------
        if(myRank == srcRank):
            gaspi_printf("Run #%d"%run)
            gaspi_printf("=========================================")

            rests = cmprex.getRests()
            # send data to Receiver side. In last iteration send remaining rests.
            if(run<num_runs):
                # write test data
                print_vector("Source Vector", values)
                print_vector("Rests Vector", rests)

                cmprex.writeRemote(values, destRank, tag)
            else:
                # at the end, flush out the rests
                print_vector("Flush Rests Vector", rests)
                cmprex.flushRests(destRank, tag)
            # print Rests Vector after send, because it should be updated
            rests = cmprex.getRests()
            print_vector("Rests Vector after send", rests)

        # Destination side
        # ------------------------------------------------------------
        if(myRank == destRank):
            # get Data from sender
            gaspi_printf("Dest Rank receiving")
            rxVect = cmprex.readRemote(srcRank, tag)
            print_vector("Received Vector:", rxVect)
            for i in range(size):
                accu[i] += rxVect[i]

        gaspi_context.barrier()

    # check if transmitted values add up to correct values:
    if(myRank == destRank):
        errors=0
        for i in range(size):
            if(accu[i] != gold_result[i]):
                errors += 1
        if(errors==0):
            print("+++PASSED+++")
        else:
            print("---FAILED---")
            print("with %d errors."%errors)
            print("Accu is:")
            print(accu)
            print("expected:")
            print(gold_result)

gaspi_printf("Done.")

