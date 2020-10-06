import tensorflow as tf


def apply_op_to_not_none_tensors(tensor_op, tensors, *args):
    return [
        tensor_op(
            tensor,
            *
            args) if tensor is not None else tensor for tensor in tensors]


def get_not_none_from_list(tensor_list):
    return [x for x in tensor_list if x is not None]


class LocalGradientAggregationHelper:
    """
    LocalGradientAggregationHelper aggregates gradient updates locally,
    and communicates the updates across machines only once every
    backward_passes_per_step. Only supports graph mode execution.
    """

    def __init__(
            self,
            backward_passes_per_step,
            allreduce_func,
            sparse_as_dense,
            average_aggregated_gradients,
            rank):
        self._allreduce_grads = allreduce_func

        # backward_passes_per_step controls how often gradient updates are
        # synchronized.
        self.backward_passes_per_step = backward_passes_per_step
        assert self.backward_passes_per_step > 0

        # average_aggregated_gradients controls whether gradient updates that are
        # aggregated, should be divided by `backward_passes_per_step`.
        self.average_aggregated_gradients = average_aggregated_gradients

        # This is going to be [N] data structure holding the aggregated gradient updates
        # N is the number of parameters.
        self.locally_aggregated_grads = []

        # Used to know when to allreduce and apply gradients. We allreduce when `self.counter`
        # is equal to `self.backward_passes_per_step`. We apply gradients when `self.counter` is
        # equal to 0.
        self.counter = None

        self.sparse_as_dense = sparse_as_dense
        self.rank = rank

        # Contains the mapping of indexes of grad updates that are not None to their index in
        # locally_aggregated_grads which only contains not None gradients. When performing
        # gradient aggregation we have to remove them from the list of grads prior to passing
        # the list into a tf.cond().
        self.not_none_indexes = {}
        self.num_none_grad_updates = 0

    def init_aggregation_vars(self, grads, sess=None):
        """
        Initializes the counter that is used when to communicate and aggregate gradients
        and the tensorflow variables that store the locally aggregated gradients.
        """

        variable_scope_name = "aggregation_variables_" + str(self.rank)
        with tf.compat.v1.variable_scope(variable_scope_name, reuse=tf.compat.v1.AUTO_REUSE):
            self.counter = tf.compat.v1.get_variable(
                "aggregation_counter", shape=(), dtype=tf.int32,
                trainable=False, initializer=tf.compat.v1.zeros_initializer(),
            )
            for idx, grad in enumerate(grads):
                # Handle IndexedSlices.
                if self.sparse_as_dense and isinstance(grad, tf.IndexedSlices):
                    grad = tf.convert_to_tensor(grad)
                elif isinstance(grad, tf.IndexedSlices):
                    raise AssertionError(
                        "IndexedSlices are not supported when "
                        "`backward_passes_per_step` > 1 and "
                        "`sparse_as_dense` is False."
                    )

                # Handle grads that are None.
                if grad is None:
                    self.num_none_grad_updates += 1
                    continue
                self.not_none_indexes[idx] = len(self.locally_aggregated_grads)

                # Create shadow variable.
                grad_aggregation_variable_name = str(idx)
                grad_aggregation_variable = tf.compat.v1.get_variable(
                    grad_aggregation_variable_name,
                    shape=grad.get_shape().as_list(),
                    trainable=False,
                    initializer=tf.zeros_initializer(),
                    dtype=grad.dtype,
                    collections=[
                        tf.compat.v1.GraphKeys.LOCAL_VARIABLES,
                        "aggregating_collection"],
                )
                self.locally_aggregated_grads.append(grad_aggregation_variable)
            assert len(self.locally_aggregated_grads) + \
                self.num_none_grad_updates == len(grads)

        # We expect to get a `sess` when we need to manually do a `sess.run(...)`
        # for the variables to be initialized. This is the `tf.keras`
        # optimizers.
        if sess:
            vars_init_op = tf.compat.v1.variables_initializer(
                [self.counter, *get_not_none_from_list(self.locally_aggregated_grads)]
            )
            sess.run(vars_init_op)

    def _clear_grads(self):
        clear_ops_list = []
        for idx, grad_aggregator in enumerate(self.locally_aggregated_grads):
            clear_op = grad_aggregator.assign(grad_aggregator.initial_value)
            clear_ops_list.append(clear_op)
        return tf.group(*clear_ops_list)

    def _aggregate_grads(self, grads):
        aggregation_ops_list = []
        grads = get_not_none_from_list(grads)
        assert len(grads) == len(self.locally_aggregated_grads)

        # Apply new gradient updates to the locally copy.
        for idx, grad in enumerate(grads):
            if self.sparse_as_dense and isinstance(grad, tf.IndexedSlices):
                grad = tf.convert_to_tensor(grad)

            updated_grad_aggregator = self.locally_aggregated_grads[idx].assign_add(
                grad)
            aggregation_ops_list.append(updated_grad_aggregator)

        return aggregation_ops_list

    def _allreduce_grads_helper(self, grads):
        # Read in latest variables values.
        aggregated_grads = []
        aggregation_read_ops_list = []
        for idx, locally_aggregated_grad in enumerate(
                self.locally_aggregated_grads):
            aggregated_grads.append(locally_aggregated_grad.read_value())
            aggregation_read_ops_list.append(aggregated_grads[idx])
        aggregation_read_ops = tf.group(*aggregation_read_ops_list)

        with tf.control_dependencies([aggregation_read_ops]):
            averaged_gradients = self._allreduce_grads(aggregated_grads)

            # Reset counter.
            with tf.control_dependencies([g.op for g in averaged_gradients if g is not None]):
                reset_op = self.counter.assign(
                    tf.constant(0), use_locking=True)

            # Divide by backward_passes_per_step if
            # average_aggregated_gradients is True.
            with tf.control_dependencies([reset_op]):
                gradient_divisor = self.backward_passes_per_step if \
                    self.average_aggregated_gradients else 1

                averaged_gradients = apply_op_to_not_none_tensors(
                    tf.divide,
                    averaged_gradients,
                    gradient_divisor,
                )
                return averaged_gradients

    def compute_gradients(self, grads):
        """
        Applies the new gradient updates the locally aggregated gradients, and
        performs cross-machine communication every backward_passes_per_step
        times it is called.
        """
        # Clear the locally aggregated gradients when the counter is at zero.
        clear_op = tf.cond(
            tf.equal(
                self.counter,
                0),
            lambda: self._clear_grads(),
            tf.no_op)

        # Add new gradients to the locally aggregated gradients.
        with tf.control_dependencies([clear_op]):
            aggregation_ops_list = self._aggregate_grads(grads)

        # Increment the counter once new gradients have been applied.
        aggregation_ops = tf.group(*aggregation_ops_list)
        with tf.control_dependencies([aggregation_ops]):
            update_counter = self.counter.assign_add(tf.constant(1))

        with tf.control_dependencies([update_counter]):
            grads = get_not_none_from_list(grads)
            assert len(grads) == len(self.locally_aggregated_grads)

            # Allreduce locally aggregated gradients when the counter is equivalent to
            # `backward_passes_per_step`. This the condition is true, it also resets
            # the counter back to 0.
            allreduced_grads = tf.cond(
                tf.equal(self.counter, self.backward_passes_per_step),
                lambda: self._allreduce_grads_helper(grads),
                lambda: grads,
            )

            # Handle case where there is only one variable.
            if not isinstance(allreduced_grads, (list, tuple)):
                allreduced_grads = (allreduced_grads,)
            assert len(allreduced_grads) == len(self.locally_aggregated_grads)

            # Insert gradients that are None back in.
            allreduced_grads = [
                allreduced_grads[self.not_none_indexes[idx]] if idx in self.not_none_indexes else None
                for idx in range(len(self.locally_aggregated_grads) + self.num_none_grad_updates)
            ]
            assert len(allreduced_grads) == len(
                self.locally_aggregated_grads) + self.num_none_grad_updates

        # If gradients have not been allreduced this batch, we return the gradients
        # that were submitted as the updates (the input).
        return allreduced_grads

    def apply_gradients(self, apply_grads_closure, *args, **kwargs):
        """
        Apply updates every backward_passes_per_step, which lines up with
        the batches on which we communicated the locally aggregated gradients.
        """
        flattended_args0 = [item for tup in args[0] for item in tup]

        # Since we skip applying updates when the counter is not at zero we
        # still want to increment the global step if it is being tracked
        # (e.g., Tensorflow Estimators).
        def increment_global_step_counter():
            global_step_counter = tf.compat.v1.train.get_global_step()
            if global_step_counter is None:
                return tf.no_op()
            return global_step_counter.assign_add(
                tf.constant(1, dtype=tf.int64),
                use_locking=True,
                read_value=False
            )

        cond_increment_global_step_counter = tf.cond(
            tf.equal(self.counter, 0), tf.no_op, increment_global_step_counter)
        flattended_args0.append(cond_increment_global_step_counter)

        with tf.control_dependencies([tf.group(*get_not_none_from_list(flattended_args0))]):
            return tf.cond(
                tf.equal(
                    self.counter,
                    0),
                apply_grads_closure,
                tf.no_op)
