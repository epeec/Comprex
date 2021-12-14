import sys
sys.path.append('..') # find pyGPI
import os
import tensorflow as tf

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import mnist
import numpy as np

from gpi import GaspiEnvironment, gaspi_printf
import keras.callbacks as callbacks
from keras import comprex_optimizer

import model

LOGGING_ALL_RANKS = False

def main(allreduce_type, compression_ratio, DO_PROFILING, gaspi_env):

    tensorboard_dir = "log/%s_%0.1f/tensorboard"%(allreduce_type, compression_ratio*100)
    profiling_dir = "log/%s_%0.1f/profiling"%(allreduce_type, compression_ratio*100)

    batchsize = 64
    epochs = 5

    myRank = gaspi_env.get_rank()
    numRanks = gaspi_env.get_ranks()

    # logging
    local_dir = tensorboard_dir
    if LOGGING_ALL_RANKS or myRank==0:
        os.makedirs(local_dir, exist_ok = True)
        local_dir = local_dir + "/rank_%02d"%myRank
        os.makedirs(local_dir, exist_ok = True)

    # Download the MNIST dataset
    dataset = mnist.load_data()

    train_data = dataset[0][0]
    train_labels = dataset[0][1]

    num_train_samples = train_data.shape[0]
    num_samples_per_rank = num_train_samples // numRanks
    steps_per_epoch = num_samples_per_rank // batchsize

    train_data = train_data[myRank*num_samples_per_rank : (myRank+1)*num_samples_per_rank-1]
    train_labels = train_labels[myRank*num_samples_per_rank : (myRank+1)*num_samples_per_rank-1]

    test_data = dataset[1][0]
    test_labels = dataset[1][1]

    num_test_samples = test_data.shape[0]
    validation_steps = num_test_samples // batchsize

    # Reshape the data to a (70000, 28, 28, 1) tensord
    train_data = train_data.reshape([*train_data.shape,1]) / 255.0

    test_data = test_data.reshape([*test_data.shape,1]) / 255.0

    # Tranform training labels to one-hot encoding
    train_labels = np.eye(10)[train_labels]

    # Tranform test labels to one-hot encoding
    test_labels = np.eye(10)[test_labels]

    train_tfdata = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).repeat().shuffle(batchsize*10).batch(batchsize, drop_remainder=True).prefetch(10)
    test_tfdata = tf.data.Dataset.from_tensor_slices((test_data, test_labels)).batch(batchsize, drop_remainder=True)

    lenet = model.LeNet()
    # lenet.summary()

    # Compile the network
    optimizer = comprex_optimizer.ComprexOptimizer( gaspi_env
                                                  , learning_rate=0.01*numRanks
                                                  , momentum=0.9
                                                  , compression_ratio = compression_ratio
                                                  , size_factor=3.0
                                                  , allreduce_type = allreduce_type 
                                                  )
    lenet.compile(
        loss = "categorical_crossentropy",
        optimizer = optimizer,
        metrics = ["accuracy"],
        experimental_run_tf_function=False)

    # Callbacks
    callbacks_list=[]
    callbacks_list.append(callbacks.BroadcastInitWeights(0, gaspi_env))

    # training
    # add Tensorboard callback
    if LOGGING_ALL_RANKS or myRank==0:
        callbacks_list.append(tf.keras.callbacks.TensorBoard(log_dir=local_dir, histogram_freq=0, write_graph=False, write_images=False, update_freq='epoch', embeddings_freq=0, profile_batch=0, embeddings_metadata=None))
    lenet.fit(
        train_tfdata,
        steps_per_epoch=steps_per_epoch,
        epochs = epochs,
        validation_steps=validation_steps,
        validation_data=test_tfdata,
        verbose = 1 if myRank==0 else 0,
        callbacks=callbacks_list)

    # Evaluate the model
    if myRank==0:
        (loss, accuracy) = lenet.evaluate(
            test_tfdata,
            verbose = 1)
        # Print the model accuracy
        print("Test accuracy: %.2f"%(accuracy))

    gaspi_env.barrier()


if __name__ == "__main__":

    gaspi_env = GaspiEnvironment(int(1e9))

    compression_ratios = [1e-2, 1e-3]

    # no compression types
    for allreduce_type in ["ring", "bigring"]:
        main(allreduce_type, 1.0, False, gaspi_env)
        tf.keras.backend.clear_session()

    # compression types
    for allreduce_type in ["comprex_ring", "comprex_bigring"]:
        for compression_ratio in compression_ratios:
            main(allreduce_type, compression_ratio, False, gaspi_env)
            tf.keras.backend.clear_session()
    
