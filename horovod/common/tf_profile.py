import json
import os
import threading

import tensorflow as tf


class TFProfileHelper:
    def __init__(self, profile_frequency, profile_filename):
        self._profile_frequency = profile_frequency
        assert self._profile_frequency >= 0

        self._profile_filename = profile_filename

        # Used to know when to add profile logging. We profile when `self._profile_counter`
        # is equal to `self._profile_frequency`.
        self._profile_counter = None

        # Used to know start of a batch for duration profiling.
        self._start_timestamp = None

    def init_profile_vars(self, sess=None):
        with tf.variable_scope("profile_variables"):
            self._profile_counter = tf.get_variable(
                "profile_counter", shape=(), dtype=tf.int32,
                trainable=False, initializer=tf.zeros_initializer())
        if sess:
            sess.run(tf.variables_initializer([self._profile_counter]))

    def _trim_last_curly_brace(self, s):
        if s[-1] != "}":
            raise AssertionError(
                f'Expected last character in "{s}" to be "}}", but got "{s[-1]}"'
            )
        return s[:-1]

    def profile_start(self):
        if not self._profile_frequency or not self._profile_filename:
            return tf.no_op()
        return tf.cond(
            tf.equal(self._profile_counter, self._profile_frequency - 1),
            lambda: self._log_comm_start(),
            tf.no_op,
        )

    def _log_comm_start(self):
        """
        Log communication end profiling information.

        Returns a tf.print operation that writes profiling information
        for the start of communication to a file in the chrome://tracing
        format.
        """
        profile_base_info = self._trim_last_curly_brace(
            self._get_profile_info("communication", "B")
        )
        # The chrome://tracing utility uses milliseconds since epoch
        # but timestamp is seconds since epoch. Multiply by 1e6 to get
        # milliseconds.
        self._start_timestamp = tf.timestamp() * 1e6
        return tf.print(
            profile_base_info,
            ', "ts": ',
            self._start_timestamp,
            "}",
            sep="",
            output_stream=f"file://{self._profile_filename}",
        )

    def _get_profile_info(self, name, phase, **kwargs):
        info = {
            "name": name,
            "pid": os.getpid(),
            "tid": threading.current_thread().ident,
            "ph": phase,
        }
        info.update(kwargs)
        return json.dumps(info)

    def profile_end(self):
        if not self._profile_frequency or not self._profile_filename:
            return tf.no_op()
        cond = tf.cond(
            tf.equal(self._profile_counter, self._profile_frequency - 1),
            lambda: self._log_comm_end(),
            tf.no_op,
        )
        with tf.control_dependencies([cond]):
            return tf.cond(
                tf.equal(self._profile_counter, self._profile_frequency - 1),
                lambda: self._clear_profile_counter(),
                lambda: self._increment_profile_counter(),
            )

    def _log_comm_end(self):
        """
        Log communication end profiling information.

        Returns a tf.print operation that writes profiling information
        for the end of communication to a file in the chrome://tracing
        format.
        """
        profile_base_info = self._trim_last_curly_brace(
            self._get_profile_info("communication", "E")
        )
        # The chrome://tracing utility uses milliseconds since epoch
        # but timestamp is seconds since epoch.  Multiply by 1e6 to get
        # milliseconds.
        end_timestamp = tf.timestamp() * 1e6
        duration = end_timestamp - self._start_timestamp
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

    def _clear_profile_counter(self):
        return self._profile_counter.assign(0)

    def _increment_profile_counter(self):
        return self._profile_counter.assign_add(1)
