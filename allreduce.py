import ctypes
import os
import numpy as np
import numpy.ctypeslib as npct
from common import Gettable, Subscribable
#from pyGPI.gpi import gaspi_printf
script_dir = os.path.dirname(__file__)
lib_dir = os.path.join(script_dir,'./lib/libPyGPI.so')

data_t = ctypes.c_float
np_data_t_p = npct.ndpointer(dtype=np.float32, ndim=1, flags='CONTIGUOUS')

lib = npct.load_library(lib_dir, ".")


######################
# AllToOneAllreduce
######################
# AllToOneAllreduce<data_t>* AllToOneAllreduce_new(GaspiEnvironment* gaspi_env)
lib.AllToOneAllreduce_new.argtypes = [ctypes.c_void_p]
lib.AllToOneAllreduce_new.restype = ctypes.c_void_p
# void AllToOneAllreduce_del(AllToOneAllreduce<data_t>* self)
lib.AllToOneAllreduce_del.argtypes = [ctypes.c_void_p]
lib.AllToOneAllreduce_del.restype = ctypes.c_void_p
# void AllToOneAllreduce_setupConnections(AllToOneAllreduce<data_t>* self, int size, int tag, int chiefRank)
lib.AllToOneAllreduce_setupConnections.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.AllToOneAllreduce_setupConnections.restype = ctypes.c_void_p
# AllToOneAllreduce_apply(AllToOneAllreduce<data_t>* self, data_t *data, int size)
lib.AllToOneAllreduce_apply.argtypes = [ctypes.c_void_p, np_data_t_p, ctypes.c_int]
lib.AllToOneAllreduce_apply.restype = ctypes.c_void_p


######################
# Comprex_AllToOneAllreduce
######################
# Comprex_AllToOneAllreduce<data_t>* Comprex_AllToOneAllreduce_new(GaspiEnvironment* gaspi_env)
lib.Comprex_AllToOneAllreduce_new.argtypes = [ctypes.c_void_p]
lib.Comprex_AllToOneAllreduce_new.restype = ctypes.c_void_p
# void Comprex_AllToOneAllreduce_del(Comprex_AllToOneAllreduce<data_t>* self)
lib.Comprex_AllToOneAllreduce_del.argtypes = [ctypes.c_void_p]
lib.Comprex_AllToOneAllreduce_del.restype = ctypes.c_void_p
# void Comprex_AllToOneAllreduce_setupConnections(Comprex_AllToOneAllreduce<data_t>* self, int size, int tag, int chiefRank, float compression_ratio, float size_factor)
lib.Comprex_AllToOneAllreduce_setupConnections.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float]
lib.Comprex_AllToOneAllreduce_setupConnections.restype = ctypes.c_void_p
# Comprex_AllToOneAllreduce_apply(Comprex_AllToOneAllreduce<data_t>* self, data_t *data, int size)
lib.Comprex_AllToOneAllreduce_apply.argtypes = [ctypes.c_void_p, np_data_t_p, ctypes.c_int]
lib.Comprex_AllToOneAllreduce_apply.restype = ctypes.c_void_p
# Comprex_AllToOneAllreduce_flush(Comprex_AllToOneAllreduce<data_t>* self, data_t *data, int size)
lib.Comprex_AllToOneAllreduce_flush.argtypes = [ctypes.c_void_p, np_data_t_p, ctypes.c_int]
lib.Comprex_AllToOneAllreduce_flush.restype = ctypes.c_void_p
# void Comprex_AllToOneAllreduce_reset(Comprex_AllToOneAllreduce<data_t>* self)
lib.Comprex_AllToOneAllreduce_reset.argtypes = [ctypes.c_void_p]
lib.Comprex_AllToOneAllreduce_reset.restype = ctypes.c_void_p
# void Comprex_AllToOneAllreduce_set_compression_ratio(Comprex_AllToOneAllreduce<data_t>* self, float compression_ratio){
lib.Comprex_AllToOneAllreduce_set_compression_ratio.argtypes = [ctypes.c_void_p, ctypes.c_float]
lib.Comprex_AllToOneAllreduce_set_compression_ratio.restype = ctypes.c_void_p

######################
# RingAllreduce
######################
# RingAllreduce<data_t>* RingAllreduce_new(GaspiEnvironment* gaspi_env)
lib.RingAllreduce_new.argtypes = [ctypes.c_void_p]
lib.RingAllreduce_new.restype = ctypes.c_void_p
# void RingAllreduce_del(RingAllreduce<data_t>* self)
lib.RingAllreduce_del.argtypes = [ctypes.c_void_p]
lib.RingAllreduce_del.restype = ctypes.c_void_p
# void RingAllreduce_setupConnections(RingAllreduce<data_t>* self, int size, int tag)
lib.RingAllreduce_setupConnections.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
lib.RingAllreduce_setupConnections.restype = ctypes.c_void_p
# RingAllreduce_apply(RingAllreduce<data_t>* self, data_t *data, int size)
lib.RingAllreduce_apply.argtypes = [ctypes.c_void_p, np_data_t_p, ctypes.c_int]
lib.RingAllreduce_apply.restype = ctypes.c_void_p


######################
# Comprex_RingAllreduce
######################
# Comprex_RingAllreduce<data_t>* Comprex_RingAllreduce_new(GaspiEnvironment* gaspi_env)
lib.Comprex_RingAllreduce_new.argtypes = [ctypes.c_void_p]
lib.Comprex_RingAllreduce_new.restype = ctypes.c_void_p
# void Comprex_RingAllreduce_del(Comprex_RingAllreduce<data_t>* self)
lib.Comprex_RingAllreduce_del.argtypes = [ctypes.c_void_p]
lib.Comprex_RingAllreduce_del.restype = ctypes.c_void_p
# void Comprex_RingAllreduce_setupConnections(Comprex_RingAllreduce<data_t>* self, int size, int tag, float compression_ratio, float size_factor)
lib.Comprex_RingAllreduce_setupConnections.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float]
lib.Comprex_RingAllreduce_setupConnections.restype = ctypes.c_void_p
# Comprex_RingAllreduce_apply(Comprex_RingAllreduce<data_t>* self, data_t *data, int size)
lib.Comprex_RingAllreduce_apply.argtypes = [ctypes.c_void_p, np_data_t_p, ctypes.c_int]
lib.Comprex_RingAllreduce_apply.restype = ctypes.c_void_p
# Comprex_RingAllreduce_flush(Comprex_RingAllreduce<data_t>* self, data_t *data, int size)
lib.Comprex_RingAllreduce_flush.argtypes = [ctypes.c_void_p, np_data_t_p, ctypes.c_int]
lib.Comprex_RingAllreduce_flush.restype = ctypes.c_void_p
# void Comprex_RingAllreduce_reset(Comprex_RingAllreduce<data_t>* self)
lib.Comprex_RingAllreduce_reset.argtypes = [ctypes.c_void_p]
lib.Comprex_RingAllreduce_reset.restype = ctypes.c_void_p
# void Comprex_RingAllreduce_set_compression_ratio(Comprex_RingAllreduce<data_t>* self, float compression_ratio){
lib.Comprex_RingAllreduce_set_compression_ratio.argtypes = [ctypes.c_void_p, ctypes.c_float]
lib.Comprex_RingAllreduce_set_compression_ratio.restype = ctypes.c_void_p

######################
# BigRingAllreduce
######################
# BigRingAllreduce<data_t>* BigRingAllreduce_new(GaspiEnvironment* gaspi_env)
lib.BigRingAllreduce_new.argtypes = [ctypes.c_void_p]
lib.BigRingAllreduce_new.restype = ctypes.c_void_p
# void BigRingAllreduce_del(BigRingAllreduce<data_t>* self)
lib.BigRingAllreduce_del.argtypes = [ctypes.c_void_p]
lib.BigRingAllreduce_del.restype = ctypes.c_void_p
# void BigRingAllreduce_setupConnections(BigRingAllreduce<data_t>* self, int size, int tag)
lib.BigRingAllreduce_setupConnections.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
lib.BigRingAllreduce_setupConnections.restype = ctypes.c_void_p
# BigRingAllreduce_apply(BigRingAllreduce<data_t>* self, data_t *data, int size)
lib.BigRingAllreduce_apply.argtypes = [ctypes.c_void_p, np_data_t_p, ctypes.c_int]
lib.BigRingAllreduce_apply.restype = ctypes.c_void_p


######################
# Comprex_BigRingAllreduce
######################
# Comprex_BigRingAllreduce<data_t>* Comprex_BigRingAllreduce_new(GaspiEnvironment* gaspi_env)
lib.Comprex_BigRingAllreduce_new.argtypes = [ctypes.c_void_p]
lib.Comprex_BigRingAllreduce_new.restype = ctypes.c_void_p
# void Comprex_BigRingAllreduce_del(Comprex_BigRingAllreduce<data_t>* self)
lib.Comprex_BigRingAllreduce_del.argtypes = [ctypes.c_void_p]
lib.Comprex_BigRingAllreduce_del.restype = ctypes.c_void_p
# void Comprex_BigRingAllreduce_setupConnections(Comprex_BigRingAllreduce<data_t>* self, int size, int tag, float compression_ratio, float size_factor)
lib.Comprex_BigRingAllreduce_setupConnections.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float]
lib.Comprex_BigRingAllreduce_setupConnections.restype = ctypes.c_void_p
# Comprex_BigRingAllreduce_apply(Comprex_BigRingAllreduce<data_t>* self, data_t *data, int size)
lib.Comprex_BigRingAllreduce_apply.argtypes = [ctypes.c_void_p, np_data_t_p, ctypes.c_int]
lib.Comprex_BigRingAllreduce_apply.restype = ctypes.c_void_p
# Comprex_BigRingAllreduce_flush(Comprex_BigRingAllreduce<data_t>* self, data_t *data, int size)
lib.Comprex_BigRingAllreduce_flush.argtypes = [ctypes.c_void_p, np_data_t_p, ctypes.c_int]
lib.Comprex_BigRingAllreduce_flush.restype = ctypes.c_void_p
# void Comprex_BigRingAllreduce_reset(Comprex_BigRingAllreduce<data_t>* self)
lib.Comprex_BigRingAllreduce_reset.argtypes = [ctypes.c_void_p]
lib.Comprex_BigRingAllreduce_reset.restype = ctypes.c_void_p
# void Comprex_BigRingAllreduce_set_compression_ratio(Comprex_BigRingAllreduce<data_t>* self, float compression_ratio){
lib.Comprex_BigRingAllreduce_set_compression_ratio.argtypes = [ctypes.c_void_p, ctypes.c_float]
lib.Comprex_BigRingAllreduce_set_compression_ratio.restype = ctypes.c_void_p

###########################
# AllToOneAllreduce
###########################
class AllToOneAllreduce(Gettable):
    def __init__(self, gaspi_env):
        self.size=None
        self.obj = lib.AllToOneAllreduce_new(gaspi_env.get())

    def __del__(self):
        if self.obj is not None:
            lib.AllToOneAllreduce_del(self.obj)
            self.obj=None

    def setup_connections(self, size, tag, chiefRank):
        self.size = size
        lib.AllToOneAllreduce_setupConnections(self.obj, size, tag, chiefRank)

    def apply(self, vector):
        if vector.size != self.size:
            raise RuntimeError("Allreduce input vector of size %d does not match allreduce size %d!"%(vector.size, self.size))
        shape = vector.shape
        vector.shape = [self.size]
        lib.AllToOneAllreduce_apply(self.obj, vector, vector.size)
        vector.shape =  shape
        return vector

    def getSize(self):
        return self.size


###########################
# Comprex_AllToOneAllreduce
###########################
class Comprex_AllToOneAllreduce(Gettable):
    def __init__(self, gaspi_env):
        self.size=None
        self.gaspi_env = gaspi_env
        self.obj = lib.Comprex_AllToOneAllreduce_new(gaspi_env.get())

    def __del__(self):
        if self.obj is not None:
            lib.Comprex_AllToOneAllreduce_del(self.obj)
            self.obj=None

    def setup_connections(self, size, tag, chiefRank, compression_ratio, size_factor=1.0):
        self.size = size
        lib.Comprex_AllToOneAllreduce_setupConnections(self.obj, size, tag, chiefRank, compression_ratio, size_factor)

    def apply(self, vector):
        if vector.size != self.size:
            raise RuntimeError("Allreduce input vector of size %d does not match allreduce size %d!"%(vector.size, self.size))
        shape = vector.shape
        vector.shape = [self.size]
        lib.Comprex_AllToOneAllreduce_apply(self.obj, vector, vector.size)
        vector.shape =  shape
        return vector

    def flush(self, vector):
        vector.resize(self.size)
        lib.Comprex_AllToOneAllreduce_flush(self.obj, vector, vector.size)
        return vector

    def reset(self):
        lib.Comprex_AllToOneAllreduce_reset(self.obj)

    def set_compression_ratio(self, compression_ratio):
        lib.Comprex_AllToOneAllreduce_set_compression_ratio(self.obj, compression_ratio)

    def getSize(self):
        return self.size


###########################
# RingAllreduce
###########################
class RingAllreduce(Gettable):
    def __init__(self, gaspi_env):
        self.size=None
        self.obj = lib.RingAllreduce_new(gaspi_env.get())

    def __del__(self):
        if self.obj is not None:
            lib.RingAllreduce_del(self.obj)
            self.obj=None

    def setup_connections(self, size, tag):
        self.size = size
        lib.RingAllreduce_setupConnections(self.obj, size, tag)

    def apply(self, vector):
        if vector.size != self.size:
            raise RuntimeError("Allreduce input vector of size %d does not match allreduce size %d!"%(vector.size, self.size))
        shape = vector.shape
        vector.shape = [self.size]
        lib.RingAllreduce_apply(self.obj, vector, vector.size)
        vector.shape =  shape
        return vector

    def getSize(self):
        return self.size


###########################
# Comprex_RingAllreduce
###########################
class Comprex_RingAllreduce(Gettable):
    def __init__(self, gaspi_env):
        self.size=None
        self.gaspi_env = gaspi_env
        self.obj = lib.Comprex_RingAllreduce_new(gaspi_env.get())

    def __del__(self):
        if self.obj is not None:
            lib.Comprex_RingAllreduce_del(self.obj)
            self.obj=None

    def setup_connections(self, size, tag, compression_ratio, size_factor=1.0):
        self.size = size
        lib.Comprex_RingAllreduce_setupConnections(self.obj, size, tag, compression_ratio, size_factor)

    def apply(self, vector):
        if vector.size != self.size:
            raise RuntimeError("Allreduce input vector of size %d does not match allreduce size %d!"%(vector.size, self.size))
        shape = vector.shape
        vector.shape = [self.size]
        lib.Comprex_RingAllreduce_apply(self.obj, vector, vector.size)
        vector.shape =  shape
        return vector

    def flush(self, vector):
        vector.resize(self.size)
        lib.Comprex_RingAllreduce_flush(self.obj, vector, vector.size)
        return vector

    def reset(self):
        lib.Comprex_RingAllreduce_reset(self.obj)

    def set_compression_ratio(self, compression_ratio):
        lib.Comprex_RingAllreduce_set_compression_ratio(self.obj, compression_ratio)

    def getSize(self):
        return self.size


###########################
# BigRingAllreduce
###########################
class BigRingAllreduce(Gettable):
    def __init__(self, gaspi_env):
        self.size=None
        self.obj = lib.BigRingAllreduce_new(gaspi_env.get())

    def __del__(self):
        if self.obj is not None:
            lib.BigRingAllreduce_del(self.obj)
            self.obj=None

    def setup_connections(self, size, tag):
        self.size = size
        lib.BigRingAllreduce_setupConnections(self.obj, size, tag)

    def apply(self, vector):
        if vector.size != self.size:
            raise RuntimeError("Allreduce input vector of size %d does not match allreduce size %d!"%(vector.size, self.size))
        shape = vector.shape
        vector.shape = [self.size]
        lib.BigRingAllreduce_apply(self.obj, vector, vector.size)
        vector.shape =  shape
        return vector

    def getSize(self):
        return self.size


###########################
# Comprex_BigRingAllreduce
###########################
class Comprex_BigRingAllreduce(Gettable):
    def __init__(self, gaspi_env):
        self.size=None
        self.gaspi_env = gaspi_env
        self.obj = lib.Comprex_BigRingAllreduce_new(gaspi_env.get())

    def __del__(self):
        if self.obj is not None:
            lib.Comprex_BigRingAllreduce_del(self.obj)
            self.obj=None

    def setup_connections(self, size, tag, compression_ratio, size_factor=1.0):
        self.size = size
        lib.Comprex_BigRingAllreduce_setupConnections(self.obj, size, tag, compression_ratio, size_factor)

    def apply(self, vector):
        if vector.size != self.size:
            raise RuntimeError("Allreduce input vector of size %d does not match allreduce size %d!"%(vector.size, self.size))
        shape = vector.shape
        vector.shape = [self.size]
        lib.Comprex_BigRingAllreduce_apply(self.obj, vector, vector.size)
        vector.shape =  shape
        return vector

    def flush(self, vector):
        vector.resize(self.size)
        lib.Comprex_BigRingAllreduce_flush(self.obj, vector, vector.size)
        return vector

    def reset(self):
        lib.Comprex_BigRingAllreduce_reset(self.obj)

    def set_compression_ratio(self, compression_ratio):
        lib.Comprex_BigRingAllreduce_set_compression_ratio(self.obj, compression_ratio)

    def getSize(self):
        return self.size
