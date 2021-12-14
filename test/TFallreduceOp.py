import sys
sys.path.append('..')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import numpy as np
import allreduceOp
from gpi import GaspiEnvironment

from time import sleep


def main():
    # Allreduce type
    allreduce_type = "ring"
    # number of elements in array
    size = 100
    # compression ratio for Comprex
    compression_ratios = [0.1]
    # number of send/receive cycles
    num_runs = len(compression_ratios)

    size_factor=3.0
    # define GASPI roles
    chiefRank = 0
    # communication tag
    tag = 0

    # initialize GASPI
    gaspi_env = GaspiEnvironment(1<<30)

    # get local rank
    myRank = gaspi_env.get_rank()
    numRanks = gaspi_env.get_ranks()

    # create input data
    values = np.ndarray([size], dtype=np.float32)
    for i in range(size):
        values[i] = i * (1-2*(i%2))
        
    weights = tf.Variable(initial_value=values.copy(), trainable=False, dtype=tf.float32)

    # create gold result
    gold_result = np.ndarray([size], dtype=np.float32)
    for i in range(size):
        gold_result[i] = (num_runs*(numRanks-1)+1)*values[i]

    # set up allreduce
    if(allreduce_type == "alltoone"):
        allreduce = allreduceOp.AllToOneAllreduceOp(gaspi_env)
        allreduce.setup(size, tag, chiefRank)
    elif(allreduce_type == "comprex_alltoone"):
        allreduce = allreduceOp.Comprex_AllToOneAllreduceOp(gaspi_env)
        allreduce.setup(size, tag, chiefRank, compression_ratios[0], size_factor)
    # TODO: Ring needs special way how data is reset in the weights between iterations for this test to get anticipated results.
    elif(allreduce_type == "ring"):
        allreduce = allreduceOp.RingAllreduceOp(gaspi_env)
        allreduce.setup(size, tag)
    elif(allreduce_type == "comprex_ring"):
        allreduce = allreduceOp.Comprex_RingAllreduceOp(gaspi_env)
        allreduce.setup(size, tag, compression_ratios[0], size_factor)
    elif(allreduce_type == "bigring"):
        allreduce = allreduceOp.BigRingAllreduceOp(gaspi_env)
        allreduce.setup(size, tag)
    elif(allreduce_type == "comprex_bigring"):
        allreduce = allreduceOp.Comprex_BigRingAllreduceOp(gaspi_env)
        allreduce.setup(size, tag, compression_ratios[0], size_factor)
    

    gaspi_env.barrier()

    # main transmission loop
    for compression in compression_ratios:
        if(myRank != chiefRank):
            # changing `weights` sometimes changes `values` as well (it is random if this bug appears or not)
            for i in range(size):
                values[i] = i * (1-2*(i%2))
            weights.assign(values)
        if("comprex" in allreduce_type):
            allreduce.set_compression_ratio(compression)
        allreduce.apply(weights)
        gaspi_env.barrier()
    if("comprex" in allreduce_type):
        allreduce.flush(weights)

    # check if transmitted values add up to correct values:
    if(myRank == chiefRank):
        errors=0
        for i in range(size):
            if(weights[i] != gold_result[i]):
                errors += 1
        if(errors==0):
            print("+++PASSED+++")
        else:
            print("---FAILED---")
            print("with %d errors."%errors)
            print("Accu is:")
            print(weights)
            print("expected:")
            print(gold_result)

    gaspi_env.barrier()
    del allreduce

if __name__ == "__main__":
    main()
