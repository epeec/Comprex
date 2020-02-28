from pyGPI.GaspiEx import GaspiEx
import pyGPI
from pyGPI.Gpi import gaspi_printf

import tensorflow as tf
import numpy as np


class BroadcastInitWeights(tf.keras.callbacks.Callback):
    def __init__(self, srcRank):
        super(BroadcastInitWeights, self).__init__()
        self.srcRank = srcRank

    def on_train_begin(self, logs={}):
        self.model_size = self.model.count_params()
        self.comm = GaspiEx(pyGPI.gaspi_runtime.get(), pyGPI.gaspi_context.get(), pyGPI.gaspi_segment.get())
        self.comm_tag = 0
        self.myRank = pyGPI.gaspi_context.getRank()

        # chief node
        if(self.myRank==self.srcRank):
            blob = self.weights_to_blob()
            #gaspi_printf(str(blob))
            for destRank in range(pyGPI.gaspi_context.getSize()):
                if destRank != self.myRank:
                    #gaspi_printf("Sending blob to %d"%(destRank))
                    self.comm.connectTx(destRank, self.model_size, self.comm_tag)
                    self.comm.writeRemote(blob, destRank, self.comm_tag)
        # worker node
        else:
            #gaspi_printf("Receiving blob from %d"%(self.srcRank))
            self.comm.connectRx(self.srcRank, self.model_size, self.comm_tag)
            blob = self.comm.readRemote(self.model_size, self.srcRank, self.comm_tag)
            self.blob_to_weights(blob)
            #print(self.model.get_weights())

    def weights_to_blob(self):
        model_size = self.model.count_params()
        blob = np.ndarray([model_size], dtype=np.float32)
        idx=0
        weights = self.model.get_weights()
        for weight in weights:
            blob[idx:idx+weight.size] = weight.flatten()
            idx += weight.size
        return blob


    def blob_to_weights(self, blob):
        idx=0
        new_weights = []
        old_weights = self.model.get_weights()
        for old_weight in old_weights:
            new = blob[idx:idx+old_weight.size]
            idx += old_weight.size
            new = np.array(new).reshape(old_weight.shape)
            new_weights.append(new)
        self.model.set_weights(new_weights)