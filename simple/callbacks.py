import ComprEx
import GaspiEx
import Gpi
import gpi_env
from Gpi import gaspi_printf

from tensorflow.python.client import timeline

import util

import tensorflow as tf
import numpy as np

class WriteTrace(tf.keras.callbacks.Callback):
    def __init__(self, filename, run_metadata):
        super(self.__class__, self).__init__()
        self.filename = filename
        self.run_metadata = run_metadata
        #print("Write Trace enabled")

    def on_epoch_end(self, batch, logs=None):
        tl = timeline.Timeline(self.run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open(self.filename, 'w') as f:
            print("Writing timeline %s"%self.filename)
            f.write(ctf)


class BroadcastInitWeights(tf.keras.callbacks.Callback):
    def __init__(self, srcRank):
        super(BroadcastInitWeights, self).__init__()
        self.srcRank = srcRank

    def on_train_begin(self, logs={}):
        self.model_size = self.model.count_params()
        #self.threshold = comprex.ThresholdNone()
        #self.compressor = comprex.CompressorNone()
        self.comm = GaspiEx.GaspiEx(gpi_env.gaspi_runtime.get(), gpi_env.gaspi_context.get(), gpi_env.gaspi_segment.get())
        self.comm_tag = 0
        self.myRank = gpi_env.gaspi_context.getRank()

        # chief node
        if(self.myRank==self.srcRank):
            blob = util.weights_to_blob(self.model)
            #gaspi_printf(str(blob))
            for destRank in range(gpi_env.gaspi_context.getSize()):
                if destRank != self.myRank:
                    #gaspi_printf("Sending blob to %d"%(destRank))
                    self.comm.writeRemote(blob, destRank, self.comm_tag)
        # worker node
        else:
            gaspi_printf("Receiving blob from %d"%(self.srcRank))
            blob = self.comm.readRemote(self.model_size, self.srcRank, self.comm_tag)
            util.blob_to_weights(self.model, blob)
            #print(self.model.get_weights())