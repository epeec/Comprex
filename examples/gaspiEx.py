import sys
sys.path.append('..')
import GaspiEx
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

gaspiex = GaspiEx.GaspiEx(gaspi_environment, size)

# set connection pattern
if myRank==srcRank:
    gaspiex.connectTx(destRank, tag)
else:
    gaspiex.connectRx(srcRank, tag)
gaspi_printf("Connection established.")


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
for run in range(num_runs):
    # Source side
    # ------------------------------------------------------------
    if(myRank == srcRank):
        gaspi_printf("Run #%d"%run)
        gaspi_printf("=========================================")
        # send data to Receiver side.
        print_vector("Source Vector", values)
        gaspiex.writeRemote(values)

    # Destination side
    # ------------------------------------------------------------
    if(myRank == destRank):
        # get Data from sender
        rxVect = gaspiex.readRemote()
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

# delete GaspiEx before the GPI environment
del gaspiex

print("Done.")




