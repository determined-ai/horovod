import tensorflow as tf


def apply_op_to_not_none_tensors(tensor_op, tensors, *args):
    return [tensor_op(tensor, *args) if tensor is not None else tensor for tensor in tensors]


def get_not_none_from_list(tensor_list):
    return [x for x in tensor_list if x is not None]


class LocalGradientAggregationHelper:

    _OPTIMIZER_TYPE_KERAS = "optimizer_type_keras"
    _OPTIMIZER_TYPE_LEGACY = "optimizer_type_legacy"

    def __init__(self, aggregation_frequency, allreduce_func, sparse_as_dense,
                 average_aggregated_gradients, optimizer_type):
        self._allreduce_grads = allreduce_func

        # How often are parameters synchronized
        self.aggregation_frequency = aggregation_frequency
        assert self.aggregation_frequency > 0

        # Should the aggregated parameters be averaged.
        self.average_aggregated_gradients = average_aggregated_gradients

        # This is going to be N data structure holding the aggregated gradient updates
        # for parameter updates. N is the number of parameters.
        self.gpu_shadow_vars = []

        # Used to know when to allreduce and apply gradients. We allreduce when `self.counter`
        # is equal to `self.aggregation_frequency`. We apply gradients when `self.counter` is
        # equal to 0.
        self.counter = None

        self._sparse_as_dense = sparse_as_dense
        self.optimizer_type = optimizer_type

        # Contains the mapping of indexes of grad updates that are
        # not None to their index in gpu shadow vars which only
        # contains not None gradients. When performing gradient
        # aggregation we have to remove them from the list of grads
        # prior passing them into a tf.cond().
        self.not_none_indexes = {}
        self.num_none_grad_updates = 0

    def init_aggregation_vars(self, grads, sess=None):
        with tf.compat.v1.variable_scope("aggregation_variables"):
            self.counter = tf.compat.v1.get_variable(
                "aggregation_counter", shape=(), dtype=tf.int32,
                trainable=False, initializer=tf.compat.v1.zeros_initializer())
            for idx, grad in enumerate(grads):
                if self._sparse_as_dense and isinstance(grad, tf.IndexedSlices):
                    grad = tf.convert_to_tensor(value=grad)
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
                grad_aggregation_variable_name = str(idx)
                grad_aggregation_variable = tf.compat.v1.get_variable(
                    grad_aggregation_variable_name, shape=grad.get_shape().as_list(),
                    trainable=False, initializer=tf.compat.v1.zeros_initializer(), dtype=grad.dtype,
                    collections=[tf.compat.v1.GraphKeys.LOCAL_VARIABLES, "aggregating_collection"],
                )
                self.gpu_shadow_vars.append(grad_aggregation_variable)
            assert len(self.gpu_shadow_vars) + self.num_none_grad_updates == len(grads)

        # We expect to get a `sess` when we need to manually do a `sess.run(...)`
        # for the variables to be initialized. This is the `tf.keras`
        # optimizers.
        if self.optimizer_type == self._OPTIMIZER_TYPE_KERAS:
            session = tf.compat.v1.keras.backend.get_session(op_input_list=())
            vars_init_op = tf.compat.v1.variables_initializer(
                [self.counter, *get_not_none_from_list(self.gpu_shadow_vars)]
            )
            session.run(vars_init_op)

    def _clear_grads(self):
        clear_ops_list = []
        for idx, grad_aggregator in enumerate(self.gpu_shadow_vars):
            clear_op = grad_aggregator.assign(
                grad_aggregator.initial_value)
            clear_ops_list.append(clear_op)
        return tf.group(*clear_ops_list)

    def _aggregate_grads(self, grads):
        aggregation_ops_list = []
        grads = get_not_none_from_list(grads)
        assert len(grads) == len(self.gpu_shadow_vars)
        for idx, grad in enumerate(grads):
            if self._sparse_as_dense and isinstance(grad, tf.IndexedSlices):
                grad = tf.convert_to_tensor(value=grad)
            grad_aggregator = self.gpu_shadow_vars[idx]
            updated_grad_aggregator = grad_aggregator.assign_add(grad)
            aggregation_ops_list.append(updated_grad_aggregator)
        return aggregation_ops_list

    def _allreduce_grads_helper(self, grads):
        # Read in latest variables values.
        aggregated_grads = []
        aggregation_read_ops_list = []
        for idx, grad_aggregator in enumerate(self.gpu_shadow_vars):
            aggregated_grads.append(
                grad_aggregator.read_value())
            aggregation_read_ops_list.append(
                aggregated_grads[idx])
        aggregation_read_ops = tf.group(*aggregation_read_ops_list)

        with tf.control_dependencies([aggregation_read_ops]):
            averaged_gradients = self._allreduce_grads(aggregated_grads)
            with tf.control_dependencies([g.op for g in averaged_gradients if g is not None]):
                reset_op = self.counter.assign(
                    tf.constant(0), use_locking=True)
            with tf.control_dependencies([reset_op]):
                gradient_divisor = self.aggregation_frequency if \
                    self.average_aggregated_gradients else 1
                averaged_gradients = apply_op_to_not_none_tensors(
                    tf.divide,
                    averaged_gradients,
                    gradient_divisor,
                )
                return averaged_gradients

    def compute_gradients(self, grads):
        clear_op = tf.cond(pred=tf.equal(self.counter, 0), true_fn=lambda: self._clear_grads(), false_fn=tf.no_op)
        with tf.control_dependencies([clear_op]):
            aggregation_ops_list = self._aggregate_grads(grads)

        aggregation_ops = tf.group(*aggregation_ops_list)
        with tf.control_dependencies([aggregation_ops]):
            update_counter = self.counter.assign_add(tf.constant(1))

        with tf.control_dependencies([update_counter]):
            grads = get_not_none_from_list(grads)
            assert len(grads) == len(self.gpu_shadow_vars)
            allreduced_grads = tf.cond(
                pred=tf.equal(self.counter, self.aggregation_frequency),
                true_fn=lambda: self._allreduce_grads_helper(grads),
                false_fn=lambda: grads,
            )
            if not isinstance(allreduced_grads, (list, tuple)):
                allreduced_grads = (allreduced_grads,)
            assert len(allreduced_grads) == len(self.gpu_shadow_vars)
            allreduced_grads = [
                allreduced_grads[self.not_none_indexes[idx]] if idx in self.not_none_indexes else None
                for idx in range(len(self.gpu_shadow_vars) + self.num_none_grad_updates)
            ]
            assert len(allreduced_grads) == len(self.gpu_shadow_vars) + self.num_none_grad_updates

        return allreduced_grads

    def apply_gradients(self, apply_grads_closure, optimizer, *args, **kwargs):
        flattended_args0 = [item for tup in args[0] for item in tup]

        def increment_global_step_counter():
            global_step_counter = tf.compat.v1.train.get_global_step()
            if global_step_counter is None:
                return tf.no_op()
            return global_step_counter.assign_add(
                tf.constant(1, dtype=tf.int64),
                use_locking=True,
                read_value=False
            )

        def increment_optimizer_iteration():
            if hasattr(optimizer, "_iterations"):
                return optimizer._iterations.assign_add(1).op
            return tf.no_op()

        cond_increment_global_step_counter = tf.cond(
            pred=tf.equal(self.counter, 0), true_fn=tf.no_op, false_fn=increment_global_step_counter)
        flattended_args0.append(cond_increment_global_step_counter)

        with tf.control_dependencies([tf.group(*get_not_none_from_list(flattended_args0))]):
            return tf.cond(
                pred=tf.equal(self.counter, 0),
                true_fn=apply_grads_closure,
                false_fn=increment_optimizer_iteration
            )
