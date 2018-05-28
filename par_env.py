#!/usr/bin/env python
import csv
import numpy as np


class ParEnv(object):
    '''
    Personalized Ads Recommendation environment
    '''
    SA_FIELDS = [
        'visit', 'visit_time_recency', 'success',
        'success_time_recency', 'purchased_count',
        'purchased_money', 'age', 'sex', 'reg_date',
        'discount_rate', 'list_price', 'discount_price',
        'sales_release_date', 'sales_end_date',
        'sales_period',
    ]

    def __init__(self, path, gamma=0.0):
        # discount factor, by default agent is myopic
        self.gamma = gamma
        # states, actions, rewards
        self.sa = np.array((None, len(ParEnv.SA_FIELDS)), dtype=np.float)
        self.r = np.array((None, ), dtype=np.float)

        trajectories = list()
        with open(path, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            header = next(reader)

            customer_id = None
            for row in reader:
                interaction = dict()
                for (k, v) in zip(header, row):
                    interaction[k] = v

                if customer_id != interaction['customer_id']:
                    customer_id = interaction['customer_id']
                    trajectories.append(list())

                trajectories[-1].append(interaction)

        for trajectory in trajectories:
            assert len(set(map(lambda x: x['customer_id'], trajectory))) == 1

            rewards = list()
            for interaction in trajectory:

                # state and action
                self.sa = np.append(self.sa, [float(interaction[k]) for k in ParEnv.SA_FIELDS], axis=0)
                # reward
                rewards.append(float(interaction['purchased']))

            for i in range(len(rewards)):
                reward = np.sum([
                    rewards[i + j] * (self.gamma ** j) for j in range(len(rewards[i:]))
                ])
                self.r = np.append(self.r, reward)

    def input_dim(self):
        return self.sa.shape[1]


if __name__ == '__main__':
    par_env = ParEnv('user_trajectories.csv', gamma=1.0)
    print(len(par_env.action_space))
    assert len(par_env.s) == len(par_env.a)
    assert len(par_env.s) == len(par_env.r)
