import ctypes
import os
import numpy as np
import numpy.ctypeslib as npct
script_dir = os.path.dirname(__file__)
lib_dir = os.path.join(script_dir,'./build/src/libPyGPI.so')
#libgaspiex = ctypes.cdll.LoadLibrary(lib_dir)


data_t = ctypes.c_float
np_data_t_p = npct.ndpointer(dtype=np.float32, ndim=1, flags='CONTIGUOUS')

libcd = npct.load_library(lib_dir, ".")

######################
# GaspiEx
######################
'''
# GaspiEx* new_wEnv(gaspi::Runtime* runTime, gaspi::Context* context, gaspi::segment::Segment* segment)
libgaspiex.GaspiEx_new.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
libgaspiex.GaspiEx_new.restype = ctypes.c_void_p
# void GaspiEx_del(GaspiEx<data_t>* self)
libgaspiex.GaspiEx_del.argtypes = [ctypes.c_void_p]
libgaspiex.GaspiEx_del.restype = ctypes.c_void_p
# void writeRemote(GaspiEx<data_t>* self, const data_t* vector, int size, int destRank, int tag)
libgaspiex.GaspiEx_writeRemote.argtypes = [ctypes.c_void_p, ctypes.POINTER(data_t), ctypes.c_int , ctypes.c_int, ctypes.c_int]
libgaspiex.GaspiEx_writeRemote.restype = ctypes.c_void_p
# void readRemote(GaspiEx<data_t>* self, data_t* vector, int size, int srcRank, int tag)
libgaspiex.GaspiEx_readRemote.argtypes = [ctypes.c_void_p, ctypes.POINTER(data_t), ctypes.c_int, ctypes.c_int, ctypes.c_int]
libgaspiex.GaspiEx_readRemote.restype = ctypes.c_void_p
'''
# GaspiEx* new_wEnv(gaspi::Runtime* runTime, gaspi::Context* context, gaspi::segment::Segment* segment)
libcd.GaspiEx_new.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
libcd.GaspiEx_new.restype = ctypes.c_void_p
# void GaspiEx_del(GaspiEx<data_t>* self)
libcd.GaspiEx_del.argtypes = [ctypes.c_void_p]
libcd.GaspiEx_del.restype = ctypes.c_void_p
# void writeRemote(GaspiEx<data_t>* self, const data_t* vector, int size, int destRank, int tag)
libcd.GaspiEx_writeRemote.argtypes = [ctypes.c_void_p, np_data_t_p, ctypes.c_int , ctypes.c_int, ctypes.c_int]
libcd.GaspiEx_writeRemote.restype = ctypes.c_void_p
# void readRemote(GaspiEx<data_t>* self, data_t* vector, int size, int srcRank, int tag)
libcd.GaspiEx_readRemote.argtypes = [ctypes.c_void_p, np_data_t_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
libcd.GaspiEx_readRemote.restype = ctypes.c_void_p

class Getable(object):
    def get(self):
        return self.obj

###########################
# GaspiEx
###########################
class GaspiEx(object):
    def __init__(self, runtime, context, segment):
        #self.obj = libgaspiex.GaspiEx_new(runtime, context, segment)
        self.obj = libcd.GaspiEx_new(runtime, context, segment)

    def __del__(self):
        #libgaspiex.GaspiEx_del(self.obj)
        libcd.GaspiEx_del(self.obj)

    def writeRemote(self, vector, destRank, tag):
        try:
            #libgaspiex.GaspiEx_writeRemote(self.obj, vec, len(vector), destRank, tag)
            libcd.GaspiEx_writeRemote(self.obj, vector, vector.size, destRank, tag)
        except:
            raise RuntimeError("GPI error in writeRemote!")

    def readRemote(self, size, srcRank, tag):
        #vec = (data_t*size)()
        #vec = ctypes.POINTER(data_t*self.size)()
        res = np.ndarray([size], dtype=np.float32)
        try:
            #libgaspiex.GaspiEx_readRemote(self.obj, vec, size, srcRank, tag)
            libcd.GaspiEx_readRemote(self.obj, res, size, srcRank, tag)
        except:
            raise RuntimeError("GPI error in readRemote!")
        #res = [vec[i] for i in range(size)]
        #res = [0]*size
        return res
