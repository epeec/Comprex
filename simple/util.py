import numpy as np
import tensorflow.keras as keras
import tensorflow as tf



def weights_to_blob(_model):
    model_size = _model.count_params()
    blob = np.ndarray([model_size], dtype=np.float32)
    idx=0
    weights = _model.get_weights()
    for weight in weights:
        blob[idx:idx+weight.size] = weight.flatten()
        idx += weight.size
    return blob

def blob_to_weights(_model, blob):
    idx=0
    new_weights = []
    old_weights = _model.get_weights()
    for old_weight in old_weights:
        new = blob[idx:idx+old_weight.size]
        idx += old_weight.size
        new = np.array(new).reshape(old_weight.shape)
        new_weights.append(new)
    _model.set_weights(new_weights)
    #print(_model.get_weights())


