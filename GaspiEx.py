import ctypes
import os
import numpy as np
import numpy.ctypeslib as npct
script_dir = os.path.dirname(__file__)
lib_dir = os.path.join(script_dir,'./lib/libPyGPI.so')

data_t = ctypes.c_float
np_data_t_p = npct.ndpointer(dtype=np.float32, ndim=1, flags='CONTIGUOUS')
libcd = npct.load_library(lib_dir, ".")

######################
# GaspiEx
######################
# GaspiEx<data_t>* GaspiEx_new(GaspiEnvironment* gaspi_env, int size)
libcd.GaspiEx_new.argtypes = [ctypes.c_void_p, ctypes.c_int]
libcd.GaspiEx_new.restype = ctypes.c_void_p
# void GaspiEx_del(GaspiEx<data_t>* self)
libcd.GaspiEx_del.argtypes = [ctypes.c_void_p]
libcd.GaspiEx_del.restype = ctypes.c_void_p
# void writeRemote(GaspiEx<data_t>* self, const data_t* vector, int size)
libcd.GaspiEx_writeRemote.argtypes = [ctypes.c_void_p, np_data_t_p, ctypes.c_int ]
libcd.GaspiEx_writeRemote.restype = ctypes.c_void_p
# void readRemote(GaspiEx<data_t>* self, data_t* vector, int size)
libcd.GaspiEx_readRemote.argtypes = [ctypes.c_void_p, np_data_t_p, ctypes.c_int ]
libcd.GaspiEx_readRemote.restype = ctypes.c_void_p
# void connectTo(GaspiEx<data_t>* self, int srcRank, int targRank, int tag)
libcd.GaspiEx_connectTo.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
libcd.GaspiEx_connectTo.restype = ctypes.c_void_p
# void GaspiEx_connectTx(GaspiEx<data_t>* self, int targRank, int tag)
libcd.GaspiEx_connectTx.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
libcd.GaspiEx_connectTx.restype = ctypes.c_void_p
# void GaspiEx_connectRx(GaspiEx<data_t>* self, int srcRank, int tag)
libcd.GaspiEx_connectRx.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
libcd.GaspiEx_connectRx.restype = ctypes.c_void_p



class Getable(object):
    def get(self):
        return self.obj

###########################
# GaspiEx
###########################
class GaspiEx(object):
    def __init__(self, gaspi_env, size):
        self.obj = libcd.GaspiEx_new(gaspi_env.get(), size)
        self.size = size

    def __del__(self):
        if self.obj is not None:
            libcd.GaspiEx_del(self.obj)
            self.obj = None

    def writeRemote(self, vector):
        if vector.size > self.size:
            raise RuntimeError("Vector is too large to be send with this GaspiEx instance!")
        try:
            libcd.GaspiEx_writeRemote(self.obj, vector, vector.size)
        except:
            raise RuntimeError("GPI error in GaspiEx_writeRemote!")

    def readRemote(self):
        res = np.ndarray([self.size], dtype=np.float32)
        try:
            libcd.GaspiEx_readRemote(self.obj, res, self.size)
        except:
            raise RuntimeError("GPI error in GaspiEx_readRemote!")
        return res

    def readRemote_vec(self, vector, size):
        if size > self.size:
            raise RuntimeError("Too much data to be read with this GaspiEx instance!")
        try:
            libcd.GaspiEx_readRemote(self.obj, vector, size)
        except:
            raise RuntimeError("GPI error in GaspiEx_readRemote!")

    def connectTo(self, srcRank, destRank, tag):
        try:
            libcd.GaspiEx_connectTo(self.obj, srcRank, destRank, tag)
        except:
            raise RuntimeError("GPI error in GaspiEx_connectTo!")

    def connectTx(self, destRank, tag):
        try:
            libcd.GaspiEx_connectTx(self.obj, destRank, tag)
        except:
            raise RuntimeError("GPI error in GaspiEx_connectTx!")

    def connectRx(self, srcRank, tag):
        try:
            libcd.GaspiEx_connectRx(self.obj, srcRank, tag)
        except:
            raise RuntimeError("GPI error in GaspiEx_connectRx!")
