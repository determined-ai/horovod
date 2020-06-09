import functools
import os
import tensorflow as tf
from distutils import util
from packaging import version


def tf_function_before_tf_2_2(func):
    if version.parse(tf.__version__) < version.parse("2.2.0") \
            or util.strtobool(os.getenv("ENABLE_TF_FUNCTION_FOR_ALLREDUCE", "False")):
        @functools.wraps(func)
        @tf.function
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    else:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper


class LocalGradientAggregationHelperEager:
    def __init__(
        self,
        aggregation_frequency,
        allreduce_func,
        sparse_as_dense,
        average_aggregated_gradients,
    ):
        self._allreduce_grads = allreduce_func

        # How often are parameters synchronized.
        self.aggregation_frequency = aggregation_frequency
        assert self.aggregation_frequency > 0

        # Should the aggregated parameters be averaged.
        self.average_aggregated_gradients = average_aggregated_gradients

        # This is going to be N data structure holding the aggregated gradient updates
        # for parameter updates. N is the number of parameters.
        self.shadow_var = {}

        # Used to know when to allreduce and apply gradients. We allreduce when `self.counter`
        # is equal to `self.aggregation_frequency`. We apply gradients when `self.counter` is
        # equal to 0.
        self.counter = 0

        self._sparse_as_dense = sparse_as_dense

        # Used to keep track of the number of None gradient updates.
        self.num_none_grad_updates = 0

    @tf_function_before_tf_2_2
    def compute_gradients(self, grads):
        if self.aggregation_frequency == 1:
            return self._allreduce_helper(grads)

        resulting_grads = []
        for idx, grad in enumerate(grads):
            if idx not in self.shadow_var.keys():
                if grad is not None:
                    self.shadow_var[idx] = tf.Variable(
                        initial_value=tf.zeros_like(grad),
                        trainable=False,
                        dtype=grad.dtype,
                    )
                else:
                    self.num_none_grad_updates += 1
                    continue
            if grad is not None:
                self.shadow_var[idx].assign_add(grad)
                resulting_grads.append(self.shadow_var[idx].read_value())

        assert len(self.shadow_var) + self.num_none_grad_updates == len(grads)

        self.counter += 1
        if tf.equal(self.counter, self.aggregation_frequency):
            resulting_grads = self._allreduce_helper(resulting_grads)
            assert len(resulting_grads) == len(self.shadow_var)
            resulting_grads = [
                resulting_grads[idx] if idx in self.shadow_var else None
                for idx in range(len(resulting_grads) + self.num_none_grad_updates)
            ]
            assert (
                len(resulting_grads)
                == len(self.shadow_var) + self.num_none_grad_updates
            )
            self._clear_vars()

        return resulting_grads

    def _allreduce_helper(self, grads):
        allreduced_grads = self._allreduce_grads(grads)
        if self.aggregation_frequency > 1:
            gradient_divisor = (
                self.aggregation_frequency if self.average_aggregated_gradients else 1
            )
            allreduced_grads = [
                grad / gradient_divisor if grad is not None else None for grad in grads
            ]
        return allreduced_grads

    def _clear_vars(self):
        self.counter = 0
        for idx in self.shadow_var.keys():
            self.shadow_var[idx].assign_add(-1 * self.shadow_var[idx])

    @tf_function_before_tf_2_2
    def apply_gradients(self, apply_grads_closure, *args, **kwargs):
        if self.counter == 0:
            apply_grads_closure()
