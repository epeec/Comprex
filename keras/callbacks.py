from GaspiEx import GaspiEx
from gpi import gaspi_printf

import tensorflow as tf
import numpy as np


class BroadcastInitWeights(tf.keras.callbacks.Callback):
    def __init__(self, srcRank, gaspi_env):
        super(BroadcastInitWeights, self).__init__()
        self.srcRank = srcRank
        self.gaspi_env = gaspi_env

    def on_train_begin(self, logs={}):
        self.model_size = self.model.count_params()
        comm = GaspiEx(self.gaspi_env, self.model_size)
        comm_tag = 0
        myRank = self.gaspi_env.get_rank()
        numRanks = self.gaspi_env.get_ranks()

        # chief node
        if(myRank==self.srcRank):
            blob = weights_to_blob(self.model)
            for destRank in range(numRanks):
                if destRank != myRank:
                    comm.connectTx(destRank, comm_tag)
                    comm.writeRemote(blob)
        # worker node
        else:
            comm.connectRx(self.srcRank, comm_tag)
            blob = comm.readRemote()
            blob_to_weights(self.model, blob)



class SynchWeights(tf.keras.callbacks.Callback):
    def __init__(self, intervall, srcRank, gaspi_env):
        super(SynchWeights, self).__init__()
        self.intervall=intervall
        self.srcRank = srcRank
        self.gaspi_env = gaspi_env
        self.comm = None
        self.epoch_counter = 0
        self.comm_tag = 0

    def on_train_begin(self, logs=None):
        # initialize GaspiEx
        if self.comm is None:
            self.model_size = self.model.count_params()
            self.comm = GaspiEx(self.gaspi_env, self.model_size)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_counter += 1
        if self.epoch_counter >= self.intervall:
            self.epoch_counter = 0
            myRank = self.gaspi_env.get_rank()
            numRanks = self.gaspi_env.get_ranks()
            # synchronize
            if(myRank==self.srcRank):
                blob = weights_to_blob(self.model)
                for destRank in range(numRanks):
                    if destRank != myRank:
                        self.comm.connectTx(destRank, self.comm_tag)
                        self.comm.writeRemote(blob)
            else:
                self.comm.connectRx(self.srcRank, self.comm_tag)
                blob = self.comm.readRemote()
                blob_to_weights(self.model, blob)


def weights_to_blob(model):
    model_size = model.count_params()
    blob = np.ndarray([model_size], dtype=np.float32)
    idx=0
    weights = model.get_weights()
    for weight in weights:
        blob[idx:idx+weight.size] = weight.flatten()
        idx += weight.size
    return blob

def blob_to_weights(model, blob):
    idx=0
    new_weights = []
    old_weights = model.get_weights()
    for old_weight in old_weights:
        new = blob[idx:idx+old_weight.size]
        idx += old_weight.size
        new = np.array(new).reshape(old_weight.shape)
        new_weights.append(new)
    model.set_weights(new_weights)