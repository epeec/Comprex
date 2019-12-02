import sys
sys.path.append('../pyGPI')

import ComprEx as comprex
import Gpi as gpi
import gpi_env
from Gpi import gaspi_printf

import tensorflow as tf
import numpy as np

import data
import model
import callbacks
import util
import gpiOptimizer
from tensorflow.python.client import timeline

TRACE = True

input_shape = [28,28,3]
batch_size = 128

# get local rank
myRank = gpi_env.gaspi_context.getRank()
numRanks = gpi_env.gaspi_context.getSize()
gaspi_printf("Running rank %d of %d"%(myRank, numRanks))

# pin GPUs
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(myRank)

# number of threads
#config.intra_op_parallelism_threads=16
#config.inter_op_parallelism_threads=16

# trace results (optional)
run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE) if TRACE else None
run_metadata= tf.RunMetadata() if TRACE else None

tf.keras.backend.set_session(tf.Session(config=config))

# set up model
mymodel = model.Net(input_shape=input_shape)

model_size = mymodel.count_params()
optimizer = gpiOptimizer.create_distributed_optimizer(tf.keras.optimizers.SGD(lr=0.000001), name=None)
#optimizer = tf.keras.optimizers.SGD(lr=0.000001)
mymodel.compile(optimizer=optimizer,
              loss=tf.keras.losses.mean_squared_error,
              metrics=['accuracy'],
              options=run_options, run_metadata=run_metadata)


# broadcast initialization of rank 0 to all workers
if(myRank==0):
    #print(weights_to_blob(mymodel))
    blob = util.weights_to_blob(mymodel)
    blob[0]=4
    util.blob_to_weights(mymodel, blob)
    #print(mymodel.get_weights())
callbacks_list=[]
callbacks_list.append(callbacks.BroadcastInitWeights(0))
callbacks_list.append(callbacks.WriteTrace("timeline_%02d.json"%(myRank), run_metadata) )

# train
verbose= 1 if myRank==0 else 0
mymodel.fit_generator(data.data_gen(input_shape, batch_size), steps_per_epoch=60, epochs=5, callbacks=callbacks_list, verbose=verbose)

# output trace
'''
if TRACE:
    print("Writing trace for rank %02d."%myRank)
    tl = timeline.Timeline(run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format()
    with open('timeline_%02d.json'%myRank, 'w') as f:
        f.write(ctf)
'''

# test
predictions = mymodel.predict_generator(data.test_gen(input_shape, batch_size), steps=1)

#print(predictions)
#print(mymodel.get_weights())