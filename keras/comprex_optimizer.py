
import tensorflow as tf
import allreduceOp
from gpi import gaspi_printf
import numpy as np


class ComprexOptimizer(tf.keras.optimizers.Optimizer):

    _HAS_AGGREGATE_GRAD = True
    _AGGREGATE_BLOB = True

    def __init__(self,
                gaspi_env,
                learning_rate=0.01,
                momentum=0.9,
                compression_ratio = 0.01,
                nesterov=False,
                name="ComprexOptimizer",
                allreduce_type="comprex_bigring",
                size_factor=1.0,
                logging=False,
				pre_clipnorm=5.0,
                weight_decay=0.0,
                local_momentum=True,
                **kwargs):
        super(ComprexOptimizer, self).__init__(name, **kwargs)
        if isinstance(momentum, (int, float)) and (momentum < 0 or momentum > 1):
            raise ValueError("`momentum` must be between [0, 1].")
        self._set_hyper("learning_rate", learning_rate)
        self._set_hyper("momentum_correction", 1.0)
        self._set_hyper("momentum_factor", momentum)
        self.nesterov = nesterov
        self.gaspi_env = gaspi_env
        self.compression_ratio = compression_ratio
        self.size_factor = size_factor
        self.allreduce_type = allreduce_type
        self.logging=logging
        self.comprex_allreducers = {}
        self.pre_clipnorm = pre_clipnorm
        self.weight_decay = weight_decay
        self.local_momentum = local_momentum

    def set_compression_ratio(self, compression_ratio):
        self.compression_ratio = compression_ratio
        for _,comprex in self.comprex_allreducers.items():
            comprex.set_compression_ratio(compression_ratio)

    def create_allreduce_op(self, name, size):
        if(name == "alltoone"):
            allreduce = allreduceOp.AllToOneAllreduceOp(self.gaspi_env)
            allreduce.setup(size, tag=1, chiefRank=0)
        elif(name == "comprex_alltoone"):
            allreduce = allreduceOp.Comprex_AllToOneAllreduceOp(self.gaspi_env)
            allreduce.setup(size, tag=1, chiefRank=0, compression_ratio = self.compression_ratio, size_factor = self.size_factor)
        elif(name == "ring"):
            allreduce = allreduceOp.RingAllreduceOp(self.gaspi_env)
            allreduce.setup(size, tag=1)
        elif(name == "comprex_ring"):
            allreduce = allreduceOp.Comprex_RingAllreduceOp(self.gaspi_env)
            allreduce.setup(size, tag=1, compression_ratio = self.compression_ratio, size_factor = self.size_factor)
        elif(name == "bigring"):
            allreduce = allreduceOp.BigRingAllreduceOp(self.gaspi_env)
            allreduce.setup(size, tag=1)
        elif(name == "comprex_bigring"):
            allreduce = allreduceOp.Comprex_BigRingAllreduceOp(self.gaspi_env)
            allreduce.setup(size, tag=1, compression_ratio = self.compression_ratio, size_factor = self.size_factor)
        else:
            raise ValueError("AllreduceOp with name %s is not known!"%(name))
        return allreduce

    def _aggregate_gradients(self, grads_and_vars):
            averaged_gradients = []

            if self.gaspi_env.get_ranks() > 1:
                with tf.name_scope(self._name + "_ComprexAllreduce"):
                    if not self._AGGREGATE_BLOB:
                        for idx, (grad, var) in enumerate(grads_and_vars):
                            self.add_slot(var, "momentum")
                            grad = tf.clip_by_norm(grad, self.pre_clipnorm)
                            if(self.local_momentum):
                                momentum_var = self.get_slot(var, "momentum")
                                momentum_factor = self._get_hyper("momentum_factor")  
                                learning_rate = self._get_hyper("learning_rate")
                                momentum_correction = self._get_hyper("momentum_correction")
                                mom = momentum_factor * momentum_correction * momentum_var + learning_rate * grad
                                momentum_var.assign(mom)
                            else:
                                mom = grad
                            size = grad.shape.num_elements()
                            if idx not in self.comprex_allreducers:
                                self.comprex_allreducers[idx] = self.create_allreduce_op(self.allreduce_type, size)
                            mom = tf.convert_to_tensor(mom)
                            avg_grad = self.comprex_allreducers[idx].apply(mom)
                            avg_grad = avg_grad/self.gaspi_env.get_ranks()
                            averaged_gradients.append(avg_grad)
                    else: # aggregate as blob
                        moments = []
                        blob = []
                        # make gradient blob
                        for grad, var in grads_and_vars:
                            self.add_slot(var, "momentum")
                            grad = tf.clip_by_norm(grad, self.pre_clipnorm)
                            if(self.local_momentum):
                                momentum_var = self.get_slot(var, "momentum")
                                momentum_factor = self._get_hyper("momentum_factor")  
                                learning_rate = self._get_hyper("learning_rate")
                                momentum_correction = self._get_hyper("momentum_correction")
                                mom = momentum_factor * momentum_correction * momentum_var + learning_rate * grad
                                momentum_var.assign(mom)
                            else:
                                mom = grad
                            moments.append(mom)
                            blob.append(tf.reshape(mom, [mom.shape.num_elements()]))
                        blob = tf.concat(blob, axis=0)
                        #exchange blob
                        if isinstance(blob, tf.IndexedSlices):
                            blob = tf.convert_to_tensor(blob)
                        blob_size = blob.shape.num_elements()
                        # apply allreduce op
                        if 0 not in self.comprex_allreducers:
                            self.comprex_allreducers[0] = self.create_allreduce_op(self.allreduce_type, blob_size)
                        avg_blob = self.comprex_allreducers[0].apply(blob)
                        avg_blob = avg_blob/self.gaspi_env.get_ranks()
                        # reshape blob into gradients
                        blob_idx=0
                        for mom in moments:
                            mom_size = mom.shape.num_elements()
                            avg_mom = tf.convert_to_tensor(avg_blob[blob_idx:blob_idx+mom_size])
                            avg_mom = tf.reshape(avg_mom, mom.shape)
                            blob_idx += mom_size
                            # TODO: in TF version 2.4 the aggregate function is supposed to return a list of tuples (reduced_grads, vars).
                            averaged_gradients.append(avg_mom)
            else: # single worker, not distributed
                for grad, var in grads_and_vars:
                    self.add_slot(var, "momentum")
                    momentum_var = self.get_slot(var, "momentum")
                    momentum_factor = self._get_hyper("momentum_factor")
                    mom = momentum_factor * momentum_var + grad
                    momentum_var.assign(mom)
                    averaged_gradients.append(mom)
            return averaged_gradients

    def apply_gradients(self, *args, **kwargs):
        return super(ComprexOptimizer, self).apply_gradients(*args, **kwargs)

    def _create_slots(self, var_list):
        pass

    def _resource_apply_dense(self, grad, var, apply_state=None):
        # `grad` already contains momentum term!
        learning_rate = self._get_hyper("learning_rate")
        momentum_var = self.get_slot(var, "momentum")
        momentum_factor = self._get_hyper("momentum_factor")
        momentum_correction = self._get_hyper("momentum_correction")
        if not self.local_momentum:
            mom = momentum_factor * momentum_correction * momentum_var + learning_rate * grad
            momentum_var.assign(mom)
        else:
            mom = grad
        var_update = (1-self.weight_decay) * var - mom
        var.assign(var_update)
        return var_update

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
       raise NotImplementedError("Sparse Optimizer not implemented!")

    def get_config(self):
        config = super(ComprexOptimizer, self).get_config()
        return config
