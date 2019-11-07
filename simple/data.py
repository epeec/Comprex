import numpy as np

def data_gen(input_shape, batch_size):
    while True:
        yield np.array([np.ones(input_shape)]*batch_size), np.array([np.zeros([1])]*batch_size)

def test_gen(input_shape, batch_size):
    while True:
        yield np.array([np.ones(input_shape)]*batch_size)