import sys
sys.path.append('..')
from allreduce import BigRingAllreduce, Comprex_BigRingAllreduce
import gpi
from gpi import GaspiEnvironment, gaspi_printf
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
num_runs = 100

# compression ratio for Comprex
compression_ratio = 1e-2

# define GASPI roles
chiefRank = 0

# communication tag
tag = 0

# initialize GASPI
gaspi_environment = GaspiEnvironment(2**20) # 1 MB

# get local rank
myRank = gaspi_environment.get_rank()
numRanks = gaspi_environment.get_ranks()

# create input data
values = np.ndarray([size], dtype=np.float32)
for i in range(size):
    values[i] = i * (1-2*(i%2))

# create gold result
gold_result = np.ndarray([size], dtype=np.float32)
for i in range(size):
    gold_result[i] = (num_runs*(numRanks-1)+1)*values[i]

# accumulator
accu = np.array(values)

# set up allreduce
allreduce = Comprex_BigRingAllreduce(gaspi_environment)
allreduce.setup_connections(size, tag, compression_ratio, size_factor=3.0)

# main transmission loop
for run in range(num_runs):
    if(myRank != chiefRank):
        accu[:] = values[:]
    allreduce.apply(accu)
if(isinstance(allreduce, Comprex_BigRingAllreduce)):
    allreduce.flush(accu)


# check if transmitted values add up to correct values:
if(myRank == chiefRank):
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

# Delete allreduce, before the gaspi_environment.
del allreduce

print("Done.")