import ctypes
import os
import numpy as np
import numpy.ctypeslib as npct

script_dir = os.path.dirname(__file__)
lib_dir = os.path.join(script_dir,'../build/src/libComprEx.so')
#libcomprex = ctypes.cdll.LoadLibrary(lib_dir)

data_t = ctypes.c_float

np_data_t_p = npct.ndpointer(dtype=np.float32, ndim=1, flags='CONTIGUOUS')
libcomprex = npct.load_library(lib_dir, ".")

######################
# ComprEx
######################
# ComprEx* new_wEnv(gaspi::Runtime* runTime, gaspi::Context* context, gaspi::segment::Segment* segment, int size)
libcomprex.Comprex_new.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
libcomprex.Comprex_new.restype = ctypes.c_void_p
# void Comprex_del(ComprEx<data_t>* self)
libcomprex.Comprex_del.argtypes = [ctypes.c_void_p]
libcomprex.Comprex_del.restype = ctypes.c_void_p
# void setThreshold(ComprEx<data_t>* self, const ThresholdFunction<data_t>* threshold)
libcomprex.Comprex_setThreshold.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
libcomprex.Comprex_setThreshold.restype = ctypes.c_void_p
# void setCompressor(ComprEx<data_t>* self, const Compressor<data_t>* compressor)
libcomprex.Comprex_setCompressor.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
libcomprex.Comprex_setCompressor.restype = ctypes.c_void_p
# void resetRests(ComprEx<data_t>* self)
libcomprex.Comprex_resetRests.argtypes = [ctypes.c_void_p]
libcomprex.Comprex_resetRests.restype = ctypes.c_void_p
# void flushRests(ComprEx<data_t>* self, int destRank, int tag)
libcomprex.Comprex_flushRests.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
libcomprex.Comprex_flushRests.restype = ctypes.c_void_p
# data_t* getRests(ComprEx<data_t>* self)
libcomprex.Comprex_getRests.argtypes = [ctypes.c_void_p, np_data_t_p, ctypes.c_int]
libcomprex.Comprex_getRests.restype = ctypes.c_void_p
# void writeRemote(ComprEx<data_t>* self, const data_t* vector, int size, int destRank, int tag)
libcomprex.Comprex_writeRemote.argtypes = [ctypes.c_void_p, np_data_t_p, ctypes.c_int , ctypes.c_int, ctypes.c_int]
libcomprex.Comprex_writeRemote.restype = ctypes.c_void_p
# void readRemote(ComprEx<data_t>* self, data_t* vector, int size, int srcRank, int tag)
libcomprex.Comprex_readRemote.argtypes = [ctypes.c_void_p, np_data_t_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
libcomprex.Comprex_readRemote.restype = ctypes.c_void_p
###########################
# Thresholds
###########################
# ThresholdNone
libcomprex.ThresholdNone_new.argtypes = []
libcomprex.ThresholdNone_new.restype = ctypes.c_void_p
libcomprex.ThresholdNone_del.argtypes = [ctypes.c_void_p]
libcomprex.ThresholdNone_del.restype = ctypes.c_void_p
# ThresholdConst
libcomprex.ThresholdConst_new.argtypes = [data_t]
libcomprex.ThresholdConst_new.restype = ctypes.c_void_p
libcomprex.ThresholdConst_del.argtypes = [ctypes.c_void_p]
libcomprex.ThresholdConst_del.restype = ctypes.c_void_p
libcomprex.ThresholdConst_getThreshold.argtypes = [ctypes.c_void_p]
libcomprex.ThresholdConst_getThreshold.restype = np_data_t_p #ctypes.POINTER(data_t)
# ThresholdTopK
libcomprex.ThresholdTopK_new.argtypes = [ctypes.c_float]
libcomprex.ThresholdTopK_new.restype = ctypes.c_void_p
libcomprex.ThresholdTopK_del.argtypes = [ctypes.c_void_p]
libcomprex.ThresholdTopK_del.restype = ctypes.c_void_p
libcomprex.ThresholdTopK_getThreshold.argtypes = [ctypes.c_void_p]
libcomprex.ThresholdTopK_getThreshold.restype = data_t
###########################
# Compressor
###########################
# CompressorNone
libcomprex.CompressorNone_new.argtypes = []
libcomprex.CompressorNone_new.restype = ctypes.c_void_p
libcomprex.CompressorNone_del.argtypes = [ctypes.c_void_p]
libcomprex.CompressorNone_del.restype = ctypes.c_void_p
# CompressorRLE
libcomprex.CompressorRLE_new.argtypes = []
libcomprex.CompressorRLE_new.restype = ctypes.c_void_p
libcomprex.CompressorRLE_del.argtypes = [ctypes.c_void_p]
libcomprex.CompressorRLE_del.restype = ctypes.c_void_p


class Getable(object):
    def get(self):
        return self.obj

###########################
# ComprEx
###########################
class Comprex(object):
    def __init__(self, runtime, context, segment, size):
        self.size = size
        self.obj = libcomprex.Comprex_new(runtime, context, segment, size)

    def __del__(self):
        libcomprex.Comprex_del(self.obj)

    def setThreshold(self, threshold):
        libcomprex.Comprex_setThreshold(self.obj, threshold)

    def setCompressor(self, compressor):
        libcomprex.Comprex_setCompressor(self.obj, compressor)

    def getSize(self):
        return self.size

    def resetRests(self):
        libcomprex.Comprex_resetRests(self.obj)

    def flushRests(self, destRank, tag):
        libcomprex.Comprex_flushRests(self.obj, destRank, tag)

    def getRests(self):
        res = np.ndarray([self.size], dtype=np.float32)
        try:
            libcomprex.Comprex_getRests(self.obj, res, res.size)
        except:
            raise RuntimeError("Error getting Rests!")
        return res

    def writeRemote(self, vector, destRank, tag):
        #vec = ctypes.POINTER(data_t*len(vector))(*vector)
        #vec = (data_t*len(vector))(*vector)
        #vec = ctypes.POINTER(data_t)(vector)
        if vector.dtype is not np.float32:
            vec = np.array(vector, dtype=np.float32)
        else:
            vec = vector
        libcomprex.Comprex_writeRemote(self.obj, vec, vec.size, destRank, tag)

    def readRemote(self, srcRank, tag):
        res = np.ndarray([self.size], dtype=np.float32)
        libcomprex.Comprex_readRemote(self.obj, res, res.size, srcRank, tag)
        return res



###########################
# Thresholds
###########################
class ThresholdNone(Getable):
    def __init__(self):
        self.obj = libcomprex.ThresholdNone_new()
    def __del__(self):
        libcomprex.ThresholdNone_del(self.obj)

class ThresholdConst(Getable):
    def __init__(self, thresh):
        self.obj = libcomprex.ThresholdConst_new(thresh)
    def __del__(self):
        libcomprex.ThresholdConst_del(self.obj)
    def getThreshold(self):
        return libcomprex.ThresholdConst_getThreshold(self.obj)

class ThresholdTopK(Getable):
    def __init__(self, topK):
        self.obj = libcomprex.ThresholdTopK_new(topK)
    def __del__(self):
        libcomprex.ThresholdTopK_del(self.obj)
    def getThreshold(self):
        return libcomprex.ThresholdTopK_getThreshold(self.obj)

###########################
# Compressors
###########################
class CompressorNone(Getable):
    def __init__(self):
        self.obj = libcomprex.CompressorNone_new()
    def __del__(self):
        libcomprex.CompressorNone_del(self.obj)

class CompressorRLE(Getable):
    def __init__(self):
        self.obj = libcomprex.CompressorRLE_new()
    def __del__(self):
        libcomprex.CompressorRLE_del(self.obj)