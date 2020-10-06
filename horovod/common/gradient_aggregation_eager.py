from distutils.version import LooseVersion

import tensorflow as tf

_POST_TF_2_4_0 = LooseVersion(tf.__version__) >= LooseVersion('2.4.0')


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
        self.counter = tf.Variable(initial_value=0)

        self._sparse_as_dense = sparse_as_dense

        # Used to keep track of the number of None gradient updates.
        self.num_none_grad_updates = 0

    @tf.function
    def compute_gradients(self, grads):
        resulting_grads = []
        for idx, grad in enumerate(grads):
            if isinstance(grad, tf.IndexedSlices):
                raise AssertionError(
                    "IndexedSlices are not supported when "
                    "`self._aggregation_frequency` > 1 and "
                    "`sparse_as_dense` is False"
                )

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

        self.counter.assign_add(1)
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

        scaled_allreduced_grads = []
        gradient_divisor = (
            self.aggregation_frequency if self.average_aggregated_gradients else 1
        )
        for grad in allreduced_grads:
            if grad is None or isinstance(grad, tf.IndexedSlices):
                scaled_allreduced_grads.append(grad)
                continue

            scaled_allreduced_grads.append(grad / gradient_divisor)

        return scaled_allreduced_grads

    def _clear_vars(self):
        self.counter.assign(0)
        for idx in self.shadow_var.keys():
            self.shadow_var[idx].assign_add(-1 * self.shadow_var[idx])

    def apply_gradients(self, apply_grads_closure, optimizer, *args, **kwargs):
        def increment_optimizer_iteration():
            if hasattr(optimizer, "_iterations") and optimizer._iterations is not None:
                return optimizer._iterations.assign_add(1).op
            return tf.no_op()

        def non_aggregation_step():
            if _POST_TF_2_4_0:
                # In TF 2.4+ `_aggregate_gradients()` is called from inside of `apply_gradients()`.
                # We account for this by calling `_aggregate_gradients()` for steps where we do
                # not call `apply_gradients()`.
                transformed_grads_and_vars = optimizer._transform_unaggregated_gradients(args[0])
                _ = optimizer._aggregate_gradients(transformed_grads_and_vars)

            return increment_optimizer_iteration()

        def is_aggregation_step():
            if _POST_TF_2_4_0:
                # In TF 2.4+ we evaluate whether it's time to aggregated gradients before
                # calling `_aggregate_gradients()`.
                return tf.equal(tf.add(self.counter, 1), self.aggregation_frequency)

            return tf.equal(self.counter, 0)

        return tf.cond(
            pred=is_aggregation_step(),
            true_fn=apply_grads_closure,
            false_fn=non_aggregation_step,
        )
