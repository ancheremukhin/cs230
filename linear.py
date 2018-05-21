import tensorflow as tf
import tensorflow.contrib.layers as layers

from utils.general import get_logger
from utils.test_env import EnvTest
from core.deep_q_learning import DQN
from schedule import LinearExploration, LinearSchedule

from config.config import config


class Linear(DQN):
    """
    Fully Connected with Tensorflow
    """
    def add_placeholders_op(self):
        """
        Adds placeholders to the graph
        """
        state_shape = list(self.env.observation_space.shape)

        self.s = tf.placeholder(
            dtype=tf.uint8,
            shape=(None, state_shape[0], state_shape[1], state_shape[2] * self.config.state_history)
        )
        self.a = tf.placeholder(
            dtype=tf.int32,
            shape=(None,)
        )
        self.r = tf.placeholder(
            dtype=tf.float32,
            shape=(None,)
        )
        self.sp = tf.placeholder(
            dtype=tf.uint8,
            shape=(None, state_shape[0], state_shape[1], state_shape[2] * self.config.state_history)
        )
        self.done_mask = tf.placeholder(
            dtype=tf.bool,
            shape=(None,)
        )
        self.lr = tf.placeholder(dtype=tf.float32)

    def get_q_values_op(self, state, scope, reuse=False):
        """
        Returns Q values for all actions

        Args:
            state: (tf tensor)
                shape = (batch_size, img height, img width, nchannels)
            scope: (string) scope name, that specifies if target network or not
            reuse: (bool) reuse of variables in the scope

        Returns:
            out: (tf tensor) of shape = (batch_size, num_actions)
        """
        num_actions = self.env.action_space.n
        out = state

        out = tf.contrib.layers.fully_connected(
            tf.contrib.layers.flatten(state, scope=scope),
            num_actions,
            activation_fn=None,
            scope=scope,
            reuse=reuse
            )

        return out

    def add_update_target_op(self, q_scope, target_q_scope):
        """
        update_target_op will be called periodically
        to copy Q network weights to target Q network

        Args:
            q_scope: (string) name of the scope of variables for q
            target_q_scope: (string) name of the scope of variables
                        for the target network
        """

        q_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope=q_scope
        )

        target_q_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope=target_q_scope
        )

        self.update_target_op = tf.group(
            map(
                lambda x: tf.assign(x[1], x[0]),
                zip(q_vars, target_q_vars)
            )
        )

    def add_loss_op(self, q, target_q):
        """
        Sets the loss of a batch, self.loss is a scalar

        Args:
            q: (tf tensor) shape = (batch_size, num_actions)
            target_q: (tf tensor) shape = (batch_size, num_actions)
        """
        num_actions = self.env.action_space.n

        batch_size = tf.shape(q)[0]
        next_q = tf.multiply(
            tf.cast(tf.logical_not(self.done_mask), dtype=tf.float32),
            tf.reduce_max(target_q, axis=1)
        )

        samp_q = tf.add(
            self.r,
            self.config.gamma * next_q
        )

        self.loss = tf.reduce_sum(
            tf.square(
                tf.multiply(
                    tf.one_hot(self.a, num_actions),
                    tf.subtract(tf.reshape(samp_q, [-1, 1]), q)
                )
            )
        )
        self.loss = tf.div(self.loss, tf.cast(batch_size, dtype=tf.float32))

    def add_optimizer_op(self, scope):
        """
        Set self.train_op and self.grad_norm
        """

        var_list = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope=scope
        )

        optimizer = tf.train.AdamOptimizer(self.lr)
        grads_and_vars = optimizer.compute_gradients(self.loss, var_list)

        grads = map(lambda x: x[0], grads_and_vars)
        vars = map(lambda x: x[1], grads_and_vars)

        if self.config.grad_clip:
            grads = map(lambda x: tf.clip_by_norm(x, self.config.clip_val), grads)

        self.train_op = optimizer.apply_gradients(zip(grads, vars))
        self.grad_norm = tf.global_norm(grads)


if __name__ == '__main__':
    env = EnvTest((5, 5, 1))

    # exploration strategy
    exp_schedule = LinearExploration(
        env,
        config.eps_begin,
        config.eps_end,
        config.eps_nsteps
    )

    # learning rate schedule
    lr_schedule  = LinearSchedule(
        config.lr_begin,
        config.lr_end,
        config.lr_nsteps
    )

    # train model
    model = Linear(env, config)
    model.run(exp_schedule, lr_schedule)
