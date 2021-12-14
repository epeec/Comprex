import sys
sys.path.append('..')
from allreduce import RingAllreduce, Comprex_RingAllreduce
import gpi
from gpi import GaspiEnvironment
import numpy as np

# number of elements in array
# size should be divisible by numRanks for test to pass!
size = 100

# number of send/receive cycles
num_runs = 10

# compression ratio for Comprex
compression_ratio = 1e-2

# communication tag
tag = 0

# Gaspi roles
chiefRank = 0

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
allreduce = Comprex_RingAllreduce(gaspi_environment)
allreduce.setup_connections(size, tag, compression_ratio, size_factor=3.0)

# main transmission loop
for run in range(num_runs):
    # Use a rank-specific region (myRank+1) of the results vector to accumulate over several rounds.
    if(myRank < numRanks-1):
        val_start=(myRank+1)*(size/numRanks)
        val_stop=(myRank+2)*(size/numRanks)
    else:
        val_start=(0)*(size/numRanks)
        val_stop=(1)*(size/numRanks)
    for it in range(size):
        if (it>=val_start and it<val_stop):
            continue
        accu[it] = values[it]

    allreduce.apply(accu)
if(isinstance(allreduce, Comprex_RingAllreduce)):
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