import tensorflow as tf


class LocalGradientAggregationHelper:
    def __init__(self, aggregation_frequency, allreduce_func):
        self._allreduce_grads = allreduce_func

        # How often are parameters synchronized
        self.aggregation_frequency = aggregation_frequency
        assert self.aggregation_frequency > 0

        # This is going to be N data structure holding the aggregated gradient updates
        # for parameter updates. N is the number of parameters.
        self.gpu_shadow_vars = []

        # Used to know when to allreduce and apply gradients. We allreduce when `self.counter`
        # is equal to `self.aggregation_frequency`. We apply gradients when `self.counter` is
        # equal to 0.
        self.counter = None

    def init_aggregation_vars(self, grads):
        with tf.variable_scope("aggregation_variables"):
            self.counter = tf.get_variable(
                "aggregation_counter", shape=(), dtype=tf.int32,
                trainable=False, initializer=tf.zeros_initializer())
            if self.aggregation_frequency > 1:
                for idx, grad in enumerate(grads):
                    grad_aggregation_variable_name = str(idx)
                    grad_aggregation_variable = tf.get_variable(
                        grad_aggregation_variable_name, shape=grad.get_shape().as_list(),
                        trainable=False, initializer=tf.zeros_initializer(), dtype=grad.dtype,
                        collections=[tf.GraphKeys.LOCAL_VARIABLES, "aggregating_collection"])
                    self.gpu_shadow_vars.append(grad_aggregation_variable)
                assert len(self.gpu_shadow_vars) == len(grads)

    def _clear_grads(self):
        clear_ops_list = []
        for idx, grad_aggregator in enumerate(self.gpu_shadow_vars):
            clear_op = grad_aggregator.assign(
                grad_aggregator.initial_value)
            clear_ops_list.append(clear_op)
        return tf.group(*clear_ops_list)

    def _aggregate_grads(self, grads):
        aggregation_ops_list = []
        if self.aggregation_frequency > 1:
            for idx, grad in enumerate(grads):
                grad_aggregator = self.gpu_shadow_vars[idx]
                updated_grad_aggregator = grad_aggregator.assign_add(grad)
                aggregation_ops_list.append(updated_grad_aggregator)
        return aggregation_ops_list

    def _allreduce_grads_helper(self, grads):
        if self.aggregation_frequency > 1:
            # Read in latest variables values.
            aggregated_grads = []
            aggregation_read_ops_list = []
            for idx, grad_aggregator in enumerate(self.gpu_shadow_vars):
                aggregated_grads.append(
                    grad_aggregator.read_value())
                aggregation_read_ops_list.append(
                    aggregated_grads[idx])
            aggregation_read_ops = tf.group(
                *aggregation_read_ops_list)
        else:
            aggregated_grads = grads
            aggregation_read_ops = tf.no_op()

        with tf.control_dependencies([aggregation_read_ops]):
            averaged_gradients = self._allreduce_grads(aggregated_grads)
            with tf.control_dependencies([g.op for g in averaged_gradients]):
                reset_op = self.counter.assign(
                    tf.constant(0), use_locking=True)
            with tf.control_dependencies([reset_op]):
                if self.aggregation_frequency > 1:
                    return tuple(tf.divide(g, self.aggregation_frequency) for g in averaged_gradients)
                else:
                    # When grad updates are represented in `IndexedSlices`, we can not divide
                    # them by int. Currently aggregation_frequency > 1 is not supported
                    # `IndexedSlices`.
                    return tuple(tf.identity(g) for g in averaged_gradients)

    def compute_gradients(self, grads):
        if self.aggregation_frequency > 1:
            clear_op = tf.cond(tf.equal(self.counter, 0), lambda: self._clear_grads(), tf.no_op)
            with tf.control_dependencies([clear_op]):
                aggregation_ops_list = self._aggregate_grads(grads)

            aggregation_ops = tf.group(*aggregation_ops_list)
            with tf.control_dependencies([aggregation_ops]):
                update_counter = self.counter.assign_add(tf.constant(1))
        else:
            update_counter = tf.no_op()

        with tf.control_dependencies([update_counter]):
            if self.aggregation_frequency > 1:
                allreduced_grads = tf.cond(
                    tf.equal(self.counter, self.aggregation_frequency),
                    lambda: self._allreduce_grads_helper(grads),
                    lambda: grads,
                )
            else:
                allreduced_grads = self._allreduce_grads_helper(grads)

        with tf.control_dependencies([tf.group(*allreduced_grads)]):
            return tuple(tf.identity(grad) for grad in allreduced_grads)

    def apply_gradients(self, apply_grads_closure, *args, **kwargs):
        flattended_args0 = [item for tup in args[0] for item in tup]
        with tf.control_dependencies([tf.group(*flattended_args0)]):
            return tf.cond(tf.equal(self.counter, 0), apply_grads_closure, tf.no_op)
