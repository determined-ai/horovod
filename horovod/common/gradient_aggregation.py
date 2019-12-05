import tensorflow as tf


def apply_op_to_not_none_tensors(tensor_op, tensors, *args):
    return [tensor_op(tensor, *args) if tensor is not None else tensor for tensor in tensors]


def get_not_none_from_list(tensor_list):
    return [x for x in tensor_list if x is not None]


class LocalGradientAggregationHelper:
    def __init__(self, aggregation_frequency, allreduce_func, sparse_as_dense,
                 grad_updated_sizes_dict):
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

        self._sparse_as_dense = sparse_as_dense

        # A dictionary containing the shape of each grad.
        # This is used when gradient aggregation frequency > 1 and
        # there are grads that have dynamically set shapes.
        self.grad_updated_sizes_dict = grad_updated_sizes_dict

        # Contains the mapping of indexes of grad updates that are
        # not None to their index in gpu shadow vars which only
        # contains not None gradients. When performing gradient
        # aggregation we have to remove them from the list of grads
        # prior passing them into a tf.cond().
        self.not_none_indexes = {}
        self.num_none_grad_updates = 0

    def init_aggregation_vars(self, grads, sess=None):
        with tf.variable_scope("aggregation_variables"):
            self.counter = tf.get_variable(
                "aggregation_counter", shape=(), dtype=tf.int32,
                trainable=False, initializer=tf.zeros_initializer())
            if self.aggregation_frequency > 1:
                for idx, grad in enumerate(grads):
                    if self._sparse_as_dense and isinstance(grad, tf.IndexedSlices):
                        grad = tf.convert_to_tensor(grad)
                    elif isinstance(grad, tf.IndexedSlices):
                        raise AssertionError(
                            "IndexedSlices are not supported when "
                            "`self._aggregation_frequency` > 1 and "
                            "`self._sparse_as_dense` is False"
                        )
                    if grad is None:
                        self.num_none_grad_updates += 1
                        continue
                    self.not_none_indexes[idx] = len(self.gpu_shadow_vars)

                    if self.grad_updated_sizes_dict:
                        if str(idx) not in self.grad_updated_sizes_dict:
                            raise AssertionError
                        tensor_shape = self.grad_updated_sizes_dict[str(idx)]
                    else:
                        tensor_shape = grad.get_shape().as_list()
                    grad_aggregation_variable_name = str(idx)
                    grad_aggregation_variable = tf.get_variable(
                        grad_aggregation_variable_name, shape=tensor_shape,
                        trainable=False, initializer=tf.zeros_initializer(), dtype=grad.dtype,
                        collections=[tf.GraphKeys.LOCAL_VARIABLES, "aggregating_collection"],
                    )
                    self.gpu_shadow_vars.append(grad_aggregation_variable)
                assert len(self.gpu_shadow_vars) + self.num_none_grad_updates == len(grads)

        # We expect to get a `sess` when we need to manually do a `sess.run(...)`
        # for the variables to be initialized.
        if sess:
            vars_init_op = tf.variables_initializer(
                [self.counter, *get_not_none_from_list(self.gpu_shadow_vars)]
            )
            sess.run(vars_init_op)

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
            grads = get_not_none_from_list(grads)
            assert len(grads) == len(self.gpu_shadow_vars)
            for idx, grad in enumerate(grads):
                if self._sparse_as_dense and isinstance(grad, tf.IndexedSlices):
                    grad = tf.convert_to_tensor(grad)
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
            with tf.control_dependencies([g.op for g in averaged_gradients if g is not None]):
                reset_op = self.counter.assign(
                    tf.constant(0), use_locking=True)
            with tf.control_dependencies([reset_op]):
                if self.aggregation_frequency > 1:
                    averaged_gradients = apply_op_to_not_none_tensors(
                        tf.divide,
                        averaged_gradients,
                        self.aggregation_frequency,
                    )
                    return averaged_gradients
                else:
                    # When grad updates are represented in `IndexedSlices`, we can not divide
                    # them by int. Currently aggregation_frequency > 1 is not supported
                    # `IndexedSlices`.
                    averaged_gradients = apply_op_to_not_none_tensors(
                        tf.identity,
                        averaged_gradients,
                    )
                    return averaged_gradients

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
                grads = get_not_none_from_list(grads)
                assert len(grads) == len(self.gpu_shadow_vars)
                allreduced_grads = tf.cond(
                    tf.equal(self.counter, self.aggregation_frequency),
                    lambda: self._allreduce_grads_helper(grads),
                    lambda: grads,
                )
                assert len(allreduced_grads) == len(self.gpu_shadow_vars)
                allreduced_grads = [
                    allreduced_grads[self.not_none_indexes[idx]] if idx in self.not_none_indexes else None
                    for idx in range(len(self.gpu_shadow_vars) + self.num_none_grad_updates)
                ]
                assert len(allreduced_grads) == len(self.gpu_shadow_vars) + self.num_none_grad_updates
            else:
                allreduced_grads = self._allreduce_grads_helper(grads)

        with tf.control_dependencies([tf.group(*get_not_none_from_list(allreduced_grads))]):
            allreduced_grads = apply_op_to_not_none_tensors(
                tf.identity,
                allreduced_grads,
            )
            return tuple(allreduced_grads)

    def apply_gradients(self, apply_grads_closure, *args, **kwargs):
        flattended_args0 = [item for tup in args[0] for item in tup]
        with tf.control_dependencies([tf.group(*get_not_none_from_list(flattended_args0))]):
            return tf.cond(tf.equal(self.counter, 0), apply_grads_closure, tf.no_op)
