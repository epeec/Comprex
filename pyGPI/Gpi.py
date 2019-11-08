import ctypes
import os
import numpy as np
script_dir = os.path.dirname(__file__)
lib_dir = os.path.join(script_dir,'./build/src/libPyGPI.so')
libgpi = ctypes.cdll.LoadLibrary(lib_dir)


# gaspi::Runtime* Gaspi_Runtime_new()
libgpi.Gaspi_Runtime_new.argtypes = [] #[ctypes.c_void_p]
libgpi.Gaspi_Runtime_new.restype = ctypes.c_void_p
libgpi.Gaspi_Runtime_del.argtypes = [ctypes.c_void_p]
libgpi.Gaspi_Runtime_del.restype = ctypes.c_void_p

# gaspi::Context* Gaspi_Context_new()
libgpi.Gaspi_Context_new.argtypes = []
libgpi.Gaspi_Context_new.restype = ctypes.c_void_p
libgpi.Gaspi_Context_del.argtypes = [ctypes.c_void_p]
libgpi.Gaspi_Context_del.restype = ctypes.c_void_p
# int Gaspi_Context_getRank(gaspi::Context* context)
libgpi.Gaspi_Context_getRank.argtypes = [ctypes.c_void_p]
libgpi.Gaspi_Context_getRank.restype = ctypes.c_int
# int Gaspi_Context_getSize(gaspi::Context* context)
libgpi.Gaspi_Context_getSize.argtypes = [ctypes.c_void_p]
libgpi.Gaspi_Context_getSize.restype = ctypes.c_int
# Gaspi_Context_barrier(gaspi::Context* context)
libgpi.Gaspi_Context_barrier.argtypes = [ctypes.c_void_p]
libgpi.Gaspi_Context_barrier.restype = ctypes.c_void_p

# gaspi::segment::Segment* Gaspi_Segment_new(int size)
libgpi.Gaspi_Segment_new.argtypes = [ctypes.c_int]
libgpi.Gaspi_Segment_new.restype = ctypes.c_void_p
libgpi.Gaspi_Segment_del.argtypes = [ctypes.c_void_p]
libgpi.Gaspi_Segment_del.restype = ctypes.c_void_p

libgpi.Gaspi_Printf.argtypes = [ctypes.c_char_p]
libgpi.Gaspi_Printf.restype = ctypes.c_void_p

# bool Gaspi_isRuntimeAvailable(gaspi::Runtime* self)
libgpi.Gaspi_isRuntimeAvailable.argtypes = [ctypes.c_void_p]
libgpi.Gaspi_isRuntimeAvailable.restype = ctypes.c_bool

# void Gaspi_Allreduce_floatsum(float* output, const float* input, int size, gaspi::Context* context)
libgpi.Gaspi_Allreduce_floatsum.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_void_p]
libgpi.Gaspi_Allreduce_floatsum.restype = ctypes.c_void_p

# int Gaspi_Allreduce_Elem_Max()
libgpi.Gaspi_Allreduce_Elem_Max.argtypes = []
libgpi.Gaspi_Allreduce_Elem_Max.restype = ctypes.c_uint

class Getable(object):
    def get(self):
        return self.obj

###########################
# Gaspi_Runtime
###########################
class Gaspi_Runtime(Getable):
    def __init__(self):
        #gaspi_printf("__init__ called on %s"%self)
        if not Gaspi_Runtime.exists:
            self.obj = libgpi.Gaspi_Runtime_new()
        Gaspi_Runtime.exists=True
    def __enter__(self):
        return self
    def __del__(self):
        #gaspi_printf("__del__ called on %s"%self)
        try:
            if Gaspi_Runtime.exists:
                libgpi.Gaspi_Runtime_del(self.obj)
                Gaspi_Runtime.exists=False
        except:
            pass
    def __exit__(self, exception_type, exception_value, traceback):
        #libgpi.Gaspi_Runtime_del(self.obj)
        pass
Gaspi_Runtime.exists=False

###########################
# Gaspi_Context
###########################
# TODO: Verify if destructor of Runtime destroys gaspi context
class Gaspi_Context(Getable):
    def __init__(self):
        #gaspi_printf("__init__ called on %s"%self)
        self.obj = libgpi.Gaspi_Context_new()
    def __enter__(self):
        #gaspi_printf("__enter__ called")
        return self
    def __del__(self):
        #gaspi_printf("__del__ called on %s"%self)
        #libgpi.Gaspi_Context_del(self.obj)
        pass
    def __exit__(self, exception_type, exception_value, traceback):
        #gaspi_printf("__exit__ called")
        #libgpi.Gaspi_Context_del(self.obj)
        pass

    def getRank(self):
        return libgpi.Gaspi_Context_getRank(self.obj)

    def rank(self):
        return self.getRank()

    def getSize(self):
        return libgpi.Gaspi_Context_getSize(self.obj)

    def size(self):
        return self.getSize()

    def barrier(self):
        #gaspi_printf("barrier")
        libgpi.Gaspi_Context_barrier(self.obj)

###########################
# Gaspi_Segment
###########################
# TODO: Verify if destructor of Runtime destroys gaspi segment
class Gaspi_Segment(Getable):
    def __init__(self, size):
        self.obj = libgpi.Gaspi_Segment_new(size)
    def __enter__(self):
        return self
    def __del__(self):
        #libgpi.Gaspi_Segment_del(self.obj)
        pass
    def __exit__(self, exception_type, exception_value, traceback):
        #libgpi.Gaspi_Segment_del(self.obj)
        pass

###########################
# utilities
###########################

def isRuntimeAvailable():
        return libgpi.Gaspi_isRuntimeAvailable()

###########################
# Gaspi
###########################

def gaspi_printf(message):
    libgpi.Gaspi_Printf(bytes(message,encoding="utf-8"))


def gaspi_allreduce_floatsum(_input, context):
    try:
        old_shape = _input.shape
        array = _input.flatten().tolist()
        size = len(array)
    except:
        raise RuntimeError("gaspi_allreduce_floatsum expects numpy array as input!")
    #print("Transmission size %d"%(size))
    in_vec = (ctypes.c_float*size)(*array)
    out_vec = (ctypes.c_float*size)()
    try:
        libgpi.Gaspi_Allreduce_floatsum(out_vec, in_vec, size, context)
    except:
        raise RuntimeError("Gaspi Allreduce error!")
    output = np.array([ out_vec[i] for i in range(size)],dtype=np.float32)
    output = np.reshape(output, old_shape)
    return output

def gaspi_allreduce_elem_max():
    return libgpi.Gaspi_Allreduce_Elem_Max()