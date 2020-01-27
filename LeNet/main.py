import sys
sys.path.append('..') # find pyGPI

# LeNet for MNIST using Keras and TensorFlow
import tensorflow as tf

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import mnist
import numpy as np

import pyGPI
import pyGPI.Gpi as gpi
from pyGPI.Gpi import gaspi_printf
import pyGPI.keras.callbacks as callbacks
import pyGPI.keras.gpiOptimizer as gpiOptimizer

from tensorflow.python.client import timeline

import model

class WriteTrace(tf.keras.callbacks.Callback):
    def __init__(self, filename, run_metadata):
        super(self.__class__, self).__init__()
        self.filename = filename
        self.run_metadata = run_metadata
        #print("Write Trace enabled")

    def on_train_end(self, batch, logs=None):
        tl = timeline.Timeline(self.run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open(self.filename, 'w') as f:
            print("Writing timeline %s"%self.filename)
            f.write(ctf)

def main():
    myRank = pyGPI.gaspi_context.getRank()
    numRanks = pyGPI.gaspi_context.getSize()
    #print(numRanks)

    # pin GPUs
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(myRank)
    tf.keras.backend.set_session(tf.Session(config=config))

    # trace results (optional)
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata= tf.RunMetadata()

    # Download the MNIST dataset
    dataset = mnist.load_data()

    train_data = dataset[0][0]
    train_labels = dataset[0][1]

    num_train_samples = train_data.shape[0]
    num_samples_per_rank = num_train_samples // numRanks
    print(num_samples_per_rank)

    train_data = train_data[myRank*num_samples_per_rank : (myRank+1)*num_samples_per_rank-1]
    train_labels = train_labels[myRank*num_samples_per_rank : (myRank+1)*num_samples_per_rank-1]

    test_data = dataset[1][0]
    test_labels = dataset[1][1]

    # Reshape the data to a (70000, 28, 28, 1) tensord
    train_data = train_data.reshape([*train_data.shape,1]) / 255.0

    test_data = test_data.reshape([*test_data.shape,1]) / 255.0

    # Tranform training labels to one-hot encoding
    train_labels = np.eye(10)[train_labels]

    # Tranform test labels to one-hot encoding
    test_labels = np.eye(10)[test_labels]

    lenet = model.LeNet()

    # Compile the network
    optimizer = tf.keras.optimizers.SGD(lr=0.01)
    lenet.compile(
        loss = "categorical_crossentropy",
        optimizer = optimizer,
        metrics = ["accuracy"],
        options=run_options, run_metadata=run_metadata)

    # Callbacks
    callbacks_list=[]
    callbacks_list.append(callbacks.BroadcastInitWeights(0))
    callbacks_list.append(callbacks.WriteTrace("timeline_%02d.json"%(myRank), run_metadata) )

    # Train the model
    lenet.fit(
        train_data,
        train_labels,
        batch_size = 64,
        nb_epoch = 2,
        verbose = 1 if myRank==0 else 0,
        callbacks=callbacks_list)

    # Evaluate the model
    if myRank==0:
        (loss, accuracy) = lenet.evaluate(
            test_data,
            test_labels,
            batch_size = 128,
            verbose = 1)
        # Print the model's accuracy
        print("Test accuracy: %.2f"%(accuracy))

if __name__ == "__main__":
    main()