import tensorflow as tf
import tensorflow.contrib.layers as layers

from utils.general import get_logger
from utils.test_env import EnvTest
from q1_schedule import LinearExploration, LinearSchedule
from q2_linear import Linear


from configs.q3_nature import config


class NatureQN(Linear):
    """
    Implementing DeepMind's Nature paper. Here are the relevant urls.
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
    """
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
        out = tf.contrib.layers.conv2d(
            out,
            num_outputs=32,
            kernel_size=[8, 8],
            stride=4,
            activation_fn=tf.nn.relu,
            scope=scope + "/c-1",
            reuse=reuse
        )
        out = tf.contrib.layers.conv2d(
            out,
            num_outputs=64,
            kernel_size=[4, 4],
            stride=2,
            activation_fn=tf.nn.relu,
            scope=scope + "/c-2",
            reuse=reuse
        )
        out = tf.contrib.layers.conv2d(
            out,
            num_outputs=64,
            kernel_size=[3, 3],
            stride=1,
            activation_fn=tf.nn.relu,
            scope=scope + "/c-3",
            reuse=reuse
        )

        out = tf.contrib.layers.flatten(out)
        out = tf.contrib.layers.fully_connected(
            out,
            512,
            activation_fn=tf.nn.relu,
            scope=scope + "/fc-1",
            reuse=reuse
        )
        out = tf.contrib.layers.fully_connected(
            out,
            num_actions,
            activation_fn=None,
            scope=scope + "/fc-2",
            reuse=reuse
        )

        return out


"""
Use deep Q network for test environment.
"""
if __name__ == '__main__':
    env = EnvTest((80, 80, 1))

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin,
            config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

    # train model
    model = NatureQN(env, config)
    model.run(exp_schedule, lr_schedule)
