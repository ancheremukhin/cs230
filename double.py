import gym
from utils.preprocess import greyscale
from utils.wrappers import PreproWrapper, MaxAndSkipEnv

import tensorflow as tf
import tensorflow.contrib.layers as layers

from utils.general import get_logger
from utils.test_env import EnvTest
from q1_schedule import LinearExploration, LinearSchedule
from q3_nature import NatureQN

from configs.q6_bonus_question import config


class MyDQN(NatureQN):
    """
    Double Q learning
    see https://arxiv.org/pdf/1509.06461.pdf
    """
    def add_loss_op(self, q, target_q):
        """
        Sets the loss of a batch, self.loss is a scalar

        Double DQN

        Args:
            q: (tf tensor) shape = (batch_size, num_actions)
            target_q: (tf tensor) shape = (batch_size, num_actions)
        """
        num_actions = self.env.action_space.n
        batch_size = tf.shape(self.q_sp)[0]

        indexes = tf.argmax(self.q_sp, axis=1)
        indexes += tf.cast(tf.range(batch_size) * num_actions, tf.int64)

        next_q = tf.multiply(
            tf.cast(tf.logical_not(self.done_mask), dtype=tf.float32),
            tf.gather(tf.reshape(target_q, [-1]), indexes)
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

