from pyGPI import ComprEx
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
        self.comm = ComprEx.Comprex(pyGPI.gaspi_runtime, pyGPI.gaspi_context, pyGPI.gaspi_segment, self.model_size)
        threshold = ComprEx.ThresholdNone()
        compressor = ComprEx.CompressorNone()
        self.comm.setThreshold(threshold)
        self.comm.setCompressor(compressor)
        self.comm_tag = 0
        self.myRank = pyGPI.gaspi_context.getRank()

        # chief node
        if(self.myRank==self.srcRank):
            blob = self.weights_to_blob()
            # chief: send weights to workers.
            for destRank in range(pyGPI.gaspi_context.getSize()):
                if destRank != self.myRank:
                    self.comm.connectTx(destRank, self.comm_tag, size_factor=2)
                    self.comm.writeRemote(blob)
        else:
            # worker node
            self.comm.connectRx(self.srcRank, self.comm_tag, size_factor=2)
            blob = self.comm.readRemote()
            self.blob_to_weights(blob)

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