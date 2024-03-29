import sys
sys.path.append('..')
import ComprEx
import gpi
from gpi import gaspi_printf
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
gaspi_environment = gpi.GaspiEnvironment(2**20)


print("Transmit values from rank 0 to rank 1. Rank 1 accumulates values.")
print("Use Gaspi Logger to intermediate outputs. Start it with \'gaspi_logger&\' in your console.")

# get local rank
myRank = gaspi_environment.get_rank()
numRanks = gaspi_environment.get_ranks()
print( "Rank %d of %d"%(myRank, numRanks) )

# create threshold
threshold = ComprEx.ThresholdTopK(0.001)
gaspi_printf("Using ThresholdTopK")

# create compressor
compressor = ComprEx.CompressorIndexPairs()
gaspi_printf("Using CompressorIndexPairs")

# create Comprex
cmprex = ComprEx.Comprex(gaspi_environment, size, size_factor=2.0)
cmprex.setThreshold(threshold)
cmprex.setCompressor(compressor)
gaspi_printf("Built comprex")

# set connection pattern
if myRank==srcRank:
    cmprex.connectTx(destRank, tag)
else:
    cmprex.connectRx(srcRank, tag)
gaspi_printf("Connection established")


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

            cmprex.writeRemote(values)
        else:
            # at the end, flush out the rests
            print_vector("Flush Rests Vector", rests)
            cmprex.flushRests()
        # print Rests Vector after send, because it should be updated
        rests = cmprex.getRests()
        print_vector("Rests Vector after send", rests)

    # Destination side
    # ------------------------------------------------------------
    if(myRank == destRank):
        # get Data from sender
        #gaspi_printf("Dest Rank receiving")
        rxVect = cmprex.readRemote()
        print_vector("Received Vector:", rxVect)
        for i in range(size):
            accu[i] += rxVect[i]

    gaspi_environment.barrier()

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

del cmprex
print("Done.")




