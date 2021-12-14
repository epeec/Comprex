import ctypes
import os
import numpy as np
import numpy.ctypeslib as npct
from common import Gettable, Subscribable
script_dir = os.path.dirname(__file__)
lib_dir = os.path.join(script_dir,'./lib/libPyGPI.so')
data_t = ctypes.c_float
np_data_t_p = npct.ndpointer(dtype=np.float32, ndim=1, flags='CONTIGUOUS')

libgpi = npct.load_library(lib_dir, ".")

#########################
### GaspiCxx Runtime
#########################
# gaspi::Runtime* Gaspi_Runtime_new()
libgpi.Gaspi_Runtime_new.argtypes = []
libgpi.Gaspi_Runtime_new.restype = ctypes.c_void_p
# void Gaspi_Runtime_del(gaspi::Runtime* self)
libgpi.Gaspi_Runtime_del.argtypes = [ctypes.c_void_p]
libgpi.Gaspi_Runtime_del.restype = ctypes.c_void_p
# int Gaspi_Runtime_getGlobalRank(gaspi::Runtime* self)
libgpi.Gaspi_Runtime_getGlobalRank.argtypes = [ctypes.c_void_p]
libgpi.Gaspi_Runtime_getGlobalRank.restype = ctypes.c_int
# int Gaspi_Runtime_getGlobalSize(gaspi::Runtime* self)
libgpi.Gaspi_Runtime_getGlobalSize.argtypes = [ctypes.c_void_p]
libgpi.Gaspi_Runtime_getGlobalSize.restype = ctypes.c_int
# void Gaspi_Runtime_barrier(gaspi::Runtime* self)
libgpi.Gaspi_Runtime_barrier.argtypes = [ctypes.c_void_p]
libgpi.Gaspi_Runtime_barrier.restype = ctypes.c_void_p
# void Gaspi_Runtime_flush(gaspi::Runtime* self)
libgpi.Gaspi_Runtime_flush.argtypes = [ctypes.c_void_p]
libgpi.Gaspi_Runtime_flush.restype = ctypes.c_void_p

#########################
### GaspiCxx Segment
#########################
# gaspi::segment::Segment* Gaspi_Segment_new(int size)
libgpi.Gaspi_Segment_new.argtypes = [ctypes.c_int]
libgpi.Gaspi_Segment_new.restype = ctypes.c_void_p
# void Gaspi_Segment_del(gaspi::segment::Segment* self) 
libgpi.Gaspi_Segment_del.argtypes = [ctypes.c_void_p]
libgpi.Gaspi_Segment_del.restype = ctypes.c_void_p

############################
### GaspiCxx initGaspiCxx()
############################
# void Gaspi_initGaspiCxx()
libgpi.Gaspi_initGaspiCxx.argtypes = []
libgpi.Gaspi_initGaspiCxx.restype = ctypes.c_void_p

#########################
### GaspiEnvironment
#########################
# GaspiEnvironment* GaspiEnvironment_new(int segment_size)
libgpi.GaspiEnvironment_new.argtypes = [ctypes.c_int]
libgpi.GaspiEnvironment_new.restype = ctypes.c_void_p
# void GaspiEnvironment_del(GaspiEnvironment* self)
libgpi.GaspiEnvironment_del.argtypes = [ctypes.c_void_p]
libgpi.GaspiEnvironment_del.restype = ctypes.c_void_p
# int GaspiEnvironment_get_rank(GaspiEnvironment* self)
libgpi.GaspiEnvironment_get_rank.argtypes = [ctypes.c_void_p]
libgpi.GaspiEnvironment_get_rank.restype = ctypes.c_int
# int GaspiEnvironment_get_ranks(GaspiEnvironment* self)
libgpi.GaspiEnvironment_get_ranks.argtypes = [ctypes.c_void_p]
libgpi.GaspiEnvironment_get_ranks.restype = ctypes.c_int
# void GaspiEnvironment_barrier(GaspiEnvironment* self)
libgpi.GaspiEnvironment_barrier.argtypes = [ctypes.c_void_p]
libgpi.GaspiEnvironment_barrier.restype = ctypes.c_void_p
# void GaspiEnvironment_flush(GaspiEnvironment* self)
libgpi.GaspiEnvironment_flush.argtypes = [ctypes.c_void_p]
libgpi.GaspiEnvironment_flush.restype = ctypes.c_void_p

#########################
### GPI Functions
#########################
# void Gaspi_Printf(const char* msg)
libgpi.Gaspi_Printf.argtypes = [ctypes.c_char_p]
libgpi.Gaspi_Printf.restype = ctypes.c_void_p  



###########################
# Gaspi_Runtime
###########################
class Gaspi_Runtime(Gettable, Subscribable):
    def __init__(self):
        super().__init__()
        self.obj = libgpi.Gaspi_Runtime_new()

    def __enter__(self):
        return self

    def __del__(self):
        super().__del__()
        libgpi.Gaspi_Runtime_del(self.obj)
        self.obj = None

    def __exit__(self, exception_type, exception_value, traceback):
        super().__del__()
        libgpi.Gaspi_Runtime_del(self.obj)
        self.obj=None

    def getGlobalRank(self):
        return libgpi.Gaspi_Runtime_getGlobalRank(self.obj)

    def getGlobalSize(self):
        return libgpi.Gaspi_Runtime_getGlobalSize(self.obj)

    def barrier(self):
        libgpi.Gaspi_Runtime_barrier(self.obj)

    def flush(self):
        libgpi.Gaspi_Runtime_flush(self.obj)

###########################
# Gaspi_Segment
###########################
class Gaspi_Segment(Gettable, Subscribable):
    def __init__(self, size):
        super().__init__()
        self.obj = libgpi.Gaspi_Segment_new(size)

    def __enter__(self):
        return self

    def __del__(self):
        if self.obj is not None:
            super().__del__()
            libgpi.Gaspi_Segment_del(self.obj)
            self.obj=None

    def __exit__(self, exception_type, exception_value, traceback):
        libgpi.Gaspi_Segment_del(self.obj)
        self.obj=None


############################
### GaspiCxx initGaspiCxx()
############################
def initGaspiCxx():
    libgpi.Gaspi_initGaspiCxx()

###########################
# GaspiEnvironment
###########################
class GaspiEnvironment(Gettable):
    def __init__(self, size):
        super().__init__()
        self.obj = libgpi.GaspiEnvironment_new(size)

    def __enter__(self):
        return self

    def __del__(self):
        if self.obj is not None:
            libgpi.GaspiEnvironment_del(self.obj)
            self.obj=None

    def __exit__(self, exception_type, exception_value, traceback):
        libgpi.GaspiEnvironment_del(self.obj)
        self.obj=None

    def get_rank(self):
        return libgpi.GaspiEnvironment_get_rank(self.obj)

    def get_ranks(self):
        return libgpi.GaspiEnvironment_get_ranks(self.obj)

    def barrier(self):
        libgpi.GaspiEnvironment_barrier(self.obj)

    def flush(self):
        libgpi.GaspiEnvironment_flush(self.obj)

###########################
# GPI
###########################
def gaspi_printf(message):
    libgpi.Gaspi_Printf(bytes(message,encoding="utf-8"))
