import tensorflow as tf
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

def LeNet():
    # Create a sequential model
    model = Sequential()

    # Add the first convolution layer
    model.add(Convolution2D(
        filters = 20,
        kernel_size = (5, 5),
        padding = "same",
        input_shape = (28, 28, 1)))

    # Add a ReLU activation function
    model.add(Activation(
        activation = "relu"))

    # Add a pooling layer
    model.add(MaxPooling2D(
        pool_size = (2, 2),
        strides =  (2, 2)))

    # Add the second convolution layer
    model.add(Convolution2D(
        filters = 50,
        kernel_size = (5, 5),
        padding = "same"))

    # Add a ReLU activation function
    model.add(Activation(
        activation = "relu"))

    # Add a second pooling layer
    model.add(MaxPooling2D(
        pool_size = (2, 2),
        strides = (2, 2)))

    model.add(tf.keras.layers.GlobalAveragePooling2D())

    # Flatten the network
    model.add(Flatten())

    # Add a fully-connected output layer
    model.add(Dense(10))

    # Add a softmax activation function
    model.add(Activation("softmax"))

    return model
