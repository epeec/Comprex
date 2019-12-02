import numpy as np

SCALE = 0.5

def data_gen(input_shape, batch_size):
    while True:
        yield np.array([np.ones(input_shape)*SCALE]*batch_size), np.array([np.zeros([1])]*batch_size)

def test_gen(input_shape, batch_size):
    while True:
        yield np.array([np.ones(input_shape)*SCALE]*batch_size)