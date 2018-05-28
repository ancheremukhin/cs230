#!/usr/bin/env python
import os
import numpy as np

import tensorflow as tf
import tensorflow.contrib.layers as layers
from keras.models import Sequential
from keras.layers import Dense, Activation

from par_env import ParEnv
from config.config import config

env = ParEnv('user_trajectories.csv', gamma=config.gamma)
model = Sequential()

model.add(Dense(units=16, activation='relu', input_dim=env.input_dim()))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=1, activation='relu'))

model.compile(loss='mean_squared_error', optimizer='sgd')

print("Fitting model...")

model.fit(
    np.array(env.sa, dtype=np.float),
    np.array(env.r, dtype=np.float),
    epochs=5, batch_size=config.batch_size
)
print(model.summary())
