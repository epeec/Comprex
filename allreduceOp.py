import tensorflow as tf
import os
import allreduce
script_dir = os.path.dirname(__file__)
lib_dir = os.path.join(script_dir,'./lib/libtfGPIOps.so')
gpiops_module = tf.load_op_library(lib_dir)

class AllreduceOp():
    def __init__(self):
        self.size = None

    def __call__(self, inputs):
        return self.apply(inputs)

    def get_size(self):
        return self.size

    def apply(self, inputs):
        raise NotImplementedError()


# AllToOneAllreduce
class AllToOneAllreduceOp(AllreduceOp):
    def __init__(self, gaspi_env):
        super().__init__()
        self.gaspi_allreduce = allreduce.AllToOneAllreduce(gaspi_env)

    def setup(self, size, tag=0, chiefRank=0):
        if self.size is None:
            self.gaspi_allreduce.setup_connections(size, tag, chiefRank)
            self.size = size
        elif self.size < size:
            raise RuntimeError("tried to setup Comprex a second time with a bigger size!")

    def apply(self, inputs):
        return gpiops_module.alltoone_allreduce_apply(inputs, self.gaspi_allreduce.get())


# Comprex_AllToOneAllreduce
class Comprex_AllToOneAllreduceOp(AllreduceOp):
    def __init__(self, gaspi_env):
        super().__init__()
        self.gaspi_allreduce = allreduce.Comprex_AllToOneAllreduce(gaspi_env)

    def setup(self, size, tag=0, chiefRank=0, compression_ratio=0.01, size_factor=1.0):
        if self.size is None:
            self.gaspi_allreduce.setup_connections(size, tag, chiefRank, compression_ratio, size_factor)
            self.size = size
        elif self.size < size:
            raise RuntimeError("tried to setup Comprex a second time with a bigger size!")

    def apply(self, inputs):
        return gpiops_module.comprex_alltoone_allreduce_apply(inputs, self.gaspi_allreduce.get())

    def flush(self, inputs):
        return gpiops_module.comprex_alltoone_allreduce_flush(inputs, self.gaspi_allreduce.get())

    def set_compression_ratio(self, compression_ratio):
        self.gaspi_allreduce.set_compression_ratio(compression_ratio)


# RingAllreduce
class RingAllreduceOp(AllreduceOp):
    def __init__(self, gaspi_env):
        super().__init__()
        self.gaspi_allreduce = allreduce.RingAllreduce(gaspi_env)

    def setup(self, size, tag=0):
        if self.size is None:
            self.gaspi_allreduce.setup_connections(size, tag)
            self.size = size
        elif self.size < size:
            raise RuntimeError("tried to setup Comprex a second time with a bigger size!")

    def apply(self, inputs):
        return gpiops_module.ring_allreduce_apply(inputs, self.gaspi_allreduce.get())


# Comprex_RingAllreduce
class Comprex_RingAllreduceOp(AllreduceOp):
    def __init__(self, gaspi_env):
        super().__init__()
        self.gaspi_allreduce = allreduce.Comprex_RingAllreduce(gaspi_env)

    def setup(self, size, tag=0, compression_ratio=0.01, size_factor=1.0):
        if self.size is None:
            self.gaspi_allreduce.setup_connections(size, tag, compression_ratio, size_factor)
            self.size = size
        elif self.size < size:
            raise RuntimeError("tried to setup Comprex a second time with a bigger size!")

    def reset(self):
        self.gaspi_allreduce.reset()

    def apply(self, inputs):
        return gpiops_module.comprex_ring_allreduce_apply(inputs, self.gaspi_allreduce.get())

    def flush(self, inputs):
        return gpiops_module.comprex_ring_allreduce_flush(inputs, self.gaspi_allreduce.get())

    def set_compression_ratio(self, compression_ratio):
        self.gaspi_allreduce.set_compression_ratio(compression_ratio)

# BigRingAllreduce
class BigRingAllreduceOp(AllreduceOp):
    def __init__(self, gaspi_env):
        super().__init__()
        self.gaspi_allreduce = allreduce.BigRingAllreduce(gaspi_env)

    def setup(self, size, tag=0):
        if self.size is None:
            self.gaspi_allreduce.setup_connections(size, tag)
            self.size = size
        elif self.size < size:
            raise RuntimeError("tried to setup Comprex a second time with a bigger size!")

    def apply(self, inputs):
        return gpiops_module.bigring_allreduce_apply(inputs, self.gaspi_allreduce.get())


# Comprex_BigRingAllreduce
class Comprex_BigRingAllreduceOp(AllreduceOp):
    def __init__(self, gaspi_env):
        super().__init__()
        self.gaspi_allreduce = allreduce.Comprex_BigRingAllreduce(gaspi_env)

    def setup(self, size, tag=0, compression_ratio=0.01, size_factor=1.0):
        if self.size is None:
            self.gaspi_allreduce.setup_connections(size, tag, compression_ratio, size_factor)
            self.size = size
        elif self.size < size:
            raise RuntimeError("tried to setup Comprex a second time with a bigger size!")

    def apply(self, inputs):
        return gpiops_module.comprex_bigring_allreduce_apply(inputs, self.gaspi_allreduce.get())

    def flush(self, inputs):
        return gpiops_module.comprex_bigring_allreduce_flush(inputs, self.gaspi_allreduce.get())

    def set_compression_ratio(self, compression_ratio):
        self.gaspi_allreduce.set_compression_ratio(compression_ratio)


