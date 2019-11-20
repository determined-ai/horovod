# Copyright 2017 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import json
import os
import threading

import horovod.tensorflow as hvd
import tensorflow as tf
from horovod.common.gradient_aggregation import LocalGradientAggregationHelper


def _make_allreduce_grads_fn(device_dense, device_sparse, compression):
    def allreduce_grads(grads):
        averaged_gradients = []
        for idx, grad in enumerate(grads):
            if grad is not None:
                avg_grad = hvd.allreduce(grad,
                                         device_dense=device_dense,
                                         device_sparse=device_sparse,
                                         compression=compression)
                averaged_gradients.append(avg_grad)
            else:
                averaged_gradients.append(None)
        return averaged_gradients

    return allreduce_grads


def create_distributed_optimizer(keras, optimizer, name, device_dense, device_sparse,
                                 compression, sparse_as_dense, aggregation_frequency,
                                 grad_updated_sizes_dict, profile_frequency, profile_filename):
    class _DistributedOptimizer(keras.optimizers.Optimizer):
        def __init__(self, name, device_dense, device_sparse, compression, sparse_as_dense,
                     config, aggregation_frequency, grad_updated_sizes_dict, profile_frequency,
                     profile_filename):
            if name is None:
                name = "Distributed%s" % self.__class__.__base__.__name__
            self._name = name
            self._device_dense = device_dense
            self._device_sparse = device_sparse
            self._compression = compression
            self._sparse_as_dense = sparse_as_dense
            self._get_gradients_used = False

            self._profile_frequency = profile_frequency
            assert self._profile_frequency >= 0

            self._profile_filename = profile_filename

            # Used to know when to add profile logging. We profile when `self.profile_counter`
            # is equal to `self._profile_frequency`.
            self.profile_counter = None

            # Used to know start of a batch for duration profiling.
            self.start_timestamp = None

            self._agg_helper = LocalGradientAggregationHelper(
                aggregation_frequency,
                _make_allreduce_grads_fn(device_dense, device_sparse, compression),
                sparse_as_dense,
                grad_updated_sizes_dict
            )

            super(self.__class__, self).__init__(**config)

        def get_gradients(self, loss, params):
            """
            Compute gradients of all trainable variables.

            See Optimizer.get_gradients() for more info.

            In DistributedOptimizer, get_gradients() is overriden to also
            allreduce the gradients before returning them.
            """

            def init_profile_vars():
                v = tf.get_collection('aggregation_variables')
                vars_init_op = tf.variables_initializer(v)
                sess = tf.keras.backend.get_session(op_input_list=())

                with tf.variable_scope("aggregation_variables"):
                    self.profile_counter = tf.get_variable(
                        "profile_counter", shape=(), dtype=tf.int32,
                        trainable=False, initializer=tf.zeros_initializer())
                    vars_init_op = tf.variables_initializer([self.profile_counter])
                    sess.run(vars_init_op)

            def trim_last_curly_brace(s):
                if s[-1] != "}":
                    raise AssertionError(
                        f'Expected last character in "{s}" to be "}}", but got "{s[-1]}"'
                    )
                return s[:-1]

            def profile_start():
                if not self._profile_frequency or not self._profile_filename:
                    return tf.no_op()
                return tf.cond(
                    tf.equal(self.profile_counter, self._profile_frequency - 1),
                    log_comm_start,
                    tf.no_op,
                )

            def log_comm_start():
                """
                Log communication end profiling information.

                Returns a tf.print operation that writes profiling information
                for the start of communication to a file in the chrome://tracing
                format.
                """
                profile_base_info = trim_last_curly_brace(
                    get_profile_info("communication", "B")
                )
                # The chrome://tracing utility uses milliseconds since epoch
                # but timestamp is seconds since epoch. Multiply by 1e6 to get
                # milliseconds.
                self.start_timestamp = tf.timestamp() * 1e6
                return tf.print(
                    profile_base_info,
                    ', "ts": ',
                    self.start_timestamp,
                    "}",
                    sep="",
                    output_stream=f"file://{self._profile_filename}",
                )

            def get_profile_info(name, phase, **kwargs):
                info = {
                    "name": name,
                    "pid": os.getpid(),
                    "tid": threading.current_thread().ident,
                    "ph": phase,
                }
                info.update(kwargs)
                return json.dumps(info)

            def profile_end():
                if not self._profile_frequency or not self._profile_filename:
                    return tf.no_op()
                cond = tf.cond(
                    tf.equal(self.profile_counter, self._profile_frequency - 1),
                    log_comm_end,
                    tf.no_op,
                )
                with tf.control_dependencies([cond]):
                    return tf.cond(
                        tf.equal(self.profile_counter, self._profile_frequency - 1),
                        clear_profile_counter,
                        increment_profile_counter,
                    )

            def log_comm_end():
                """
                Log communication end profiling information.

                Returns a tf.print operation that writes profiling information
                for the end of communication to a file in the chrome://tracing
                format.
                """
                profile_base_info = trim_last_curly_brace(
                    get_profile_info("communication", "E")
                )
                # The chrome://tracing utility uses milliseconds since epoch
                # but timestamp is seconds since epoch.  Multiply by 1e6 to get
                # milliseconds.
                end_timestamp = tf.timestamp() * 1e6
                duration = end_timestamp - self.start_timestamp
                return tf.print(
                    profile_base_info,
                    ', "ts": ',
                    end_timestamp,
                    ', "duration": ',
                    duration,
                    "}",
                    sep="",
                    output_stream=f"file://{self._profile_filename}",
                )

            def clear_profile_counter():
                return self.profile_counter.assign(0)

            def increment_profile_counter():
                return self.profile_counter.assign_add(1)

            self._get_gradients_used = True
            self.grads = super(
                self.__class__, self).get_gradients(loss, params)
            init_profile_vars()
            if hvd.size() > 1:
                self._agg_helper.init_aggregation_vars(
                    self.grads,
                    sess=tf.keras.backend.get_session(op_input_list=()),
                )
                with tf.control_dependencies([profile_start()]):
                    allreduced_grads = self._agg_helper.compute_gradients(tuple(self.grads))
                with tf.control_dependencies(allreduced_grads):
                    comm_end = profile_end()
                with tf.control_dependencies([comm_end]):
                    return [tf.identity(grad) for grad in allreduced_grads]
            else:
                return self.grads

        def apply_gradients(self, *args, **kwargs):
            if not self._get_gradients_used:
                raise Exception('`apply_gradients()` was called without a call to '
                                '`get_gradients()`. If you\'re using TensorFlow 2.0, '
                                'please specify `experimental_run_tf_function=False` in '
                                '`compile()`.')
            return self._agg_helper.apply_gradients(
                lambda: super(self.__class__, self).apply_gradients(*args, **kwargs),
                *args,
                **kwargs,
            )

        @classmethod
        def from_config(cls, cfg):
            return cls(name, device_dense, device_sparse, compression, sparse_as_dense, cfg)

    # We dynamically create a new class that inherits from the optimizer that was passed in.
    # The goal is to override get_gradients() method with an allreduce implementation.
    # This class will have the same name as the optimizer it's wrapping, so that the saved
    # model could be easily restored without Horovod.
    cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
               dict(_DistributedOptimizer.__dict__))
    return cls(name, device_dense, device_sparse, compression, sparse_as_dense,
               optimizer.get_config(), aggregation_frequency, grad_updated_sizes_dict, profile_frequency,
               profile_filename)


def _eval(backend, op_or_result):
    if hvd._executing_eagerly():
        return op_or_result
    else:
        return backend.get_session().run(op_or_result)


if hasattr(hvd, 'broadcast_global_variables'):
    def broadcast_global_variables(backend, root_rank):
        return _eval(backend, hvd.broadcast_global_variables(root_rank))


def allreduce(backend, value, name, average):
    return _eval(backend, hvd.allreduce(tf.constant(value, name=name), average=average))


def allgather(backend, value, name):
    return _eval(backend, hvd.allgather(tf.constant(value, name=name)))


def broadcast(backend, value, root_rank, name):
    return _eval(backend, hvd.broadcast(tf.constant(value, name=name), root_rank))


def load_model(keras, wrap_optimizer, filepath, custom_optimizers, custom_objects):
    horovod_objects = {
        subclass.__name__.lower(): wrap_optimizer(subclass)
        for subclass in keras.optimizers.Optimizer.__subclasses__()
        if subclass.__module__ == keras.optimizers.Optimizer.__module__
    }

    if custom_optimizers is not None:
        horovod_objects.update({
            cls.__name__: wrap_optimizer(cls)
            for cls in custom_optimizers
        })

    if custom_objects is not None:
        horovod_objects.update(custom_objects)

    return keras.models.load_model(filepath, custom_objects=horovod_objects)
