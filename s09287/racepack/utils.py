import pandas as pd
import numpy as np
from random import sample
import random

import torch
import torch.nn as nn

# # reference from https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
# class QLearningTable:
#     def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
#         self.actions = actions  # a list
#         self.lr = learning_rate
#         self.gamma = reward_decay
#         self.epsilon = e_greedy
#         #dataframe 대신 numpy배열을 쓰는 경우도 많다
#         self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
#
#     def choose_action(self, observation):
#         self.check_state_exist(observation)
#
#         if np.random.uniform() < self.epsilon:
#             # choose best action
#             # state_action = self.q_table.ix[observation, :]
#             state_action = self.q_table.loc[observation, :]
#
#             # some actions have the same value
#             # permutation -> 순열을 바꾸고 random하게 선택하도록 하는 부분
#             state_action = state_action.reindex(np.random.permutation(state_action.index))
#
#             action = state_action.idxmax()
#         else:
#             # choose random action
#             action = np.random.choice(self.actions)
#
#         return action
#
#     def learn(self, s, a, r, s_):
#         self.check_state_exist(s_)
#         self.check_state_exist(s)
#
#         # q_predict = self.q_table.ix[s, a]
#         q_predict = self.q_table.loc[s, a]
#         # q_target = r + self.gamma * self.q_table.ix[s_, :].max()
#         q_target = r + self.gamma * self.q_table.loc[s_, :].max()
#
#         # update
#         # self.q_table.ix[s, a] += self.lr * (q_target - q_predict)
#         self.q_table.loc[s, a] += self.lr * (q_target - q_predict)
#
#     def check_state_exist(self, state):
#         if state not in self.q_table.index:
#             # append new state to q table
#             self.q_table = self.q_table.append(
#                 pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))


class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.disallowed_actions = {}

    def choose_action(self, observation, excluded_actions=[]):
        self.check_state_exist(observation)

        self.disallowed_actions[observation] = excluded_actions

        #state_action = self.q_table.ix[observation, :]
        #state_action = self.q_table.loc[observation, self.q_table.columns[:]]
        state_action = self.q_table.loc[observation, :]

        for excluded_action in excluded_actions:
            del state_action[excluded_action]

        if np.random.uniform() < self.epsilon:
            # some actions have the same value
            state_action = state_action.reindex(np.random.permutation(state_action.index))

            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(state_action.index)

        return action

    def learn(self, s, a, r, s_):
        if s == s_:
            return

        self.check_state_exist(s_)
        self.check_state_exist(s)

        #q_predict = self.q_table.ix[s, a]
        q_predict = self.q_table.loc[s, a]

        #s_rewards = self.q_table.ix[s_, :]
        #s_rewards = self.q_table.loc[s_, self.q_table.columns[:]]
        s_rewards = self.q_table.loc[s_, :]

        if s_ in self.disallowed_actions:
            for excluded_action in self.disallowed_actions[s_]:
                del s_rewards[excluded_action]

        if s_ != 'terminal':
            q_target = r + self.gamma * s_rewards.max()
        else:
            q_target = r  # next state is terminal

        # update
        #self.q_table.ix[s, a] += self.lr * (q_target - q_predict)
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))


class ExperienceReplayMemory:
    def __init__(self, max_size):
        # deque object that we've used for 'episodic_memory' is not suitable for random sampling
        # here, we instead use a fix-size array to implement 'buffer'
        self.buffer = [None] * max_size
        self.max_size = max_size
        self.index = 0
        self.size = 0

    def push(self, obj):
        self.buffer[self.index] = obj
        self.size = min(self.size + 1, self.max_size)
        self.index = (self.index + 1) % self.max_size

    def sample(self, batch_size):
        indices = sample(range(self.size), batch_size)
        return [self.buffer[index] for index in indices]

    def __len__(self):
        return self.size


class NaiveMultiLayerPerceptron(nn.Module):

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 num_neurons: list = [64, 32],
                 hidden_act_func: str = 'ReLU',
                 out_act_func: str = 'Identity'):
        super(NaiveMultiLayerPerceptron, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_neurons = num_neurons
        self.hidden_act_func = getattr(nn, hidden_act_func)()
        self.out_act_func = getattr(nn, out_act_func)()

        input_dims = [input_dim] + num_neurons
        output_dims = num_neurons + [output_dim]

        self.layers = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(input_dims, output_dims)):
            is_last = True if i == len(input_dims) - 1 else False
            self.layers.append(nn.Linear(in_dim, out_dim))
            if is_last:
                self.layers.append(self.out_act_func)
            else:
                self.layers.append(self.hidden_act_func)

    def forward(self, xs):
        for layer in self.layers:
            xs = layer(xs)
        return xs


class DQN(nn.Module):

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 qnet: nn.Module,
                 qnet_target: nn.Module,
                 lr: float,
                 gamma: float,
                 epsilon: float):
        """
        :param state_dim: input state dimension
        :param action_dim: action dimension
        :param qnet: main q network
        :param qnet_target: target q network
        :param lr: learning rate
        :param gamma: discount factor of MDP
        :param epsilon: E-greedy factor
        """

        super(DQN, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.qnet = qnet
        self.lr = lr
        self.gamma = gamma
        self.opt = torch.optim.Adam(params=self.qnet.parameters(), lr=lr)
        self.register_buffer('epsilon', torch.ones(1) * epsilon)
        self.cum_loss = 0.0

        # target network related
        qnet_target.load_state_dict(qnet.state_dict())
        self.qnet_target = qnet_target
        self.criteria = nn.SmoothL1Loss()

        self.count_action_random = 0
        self.count_action_select = 0

    def choose_action(self, state):
        qs = self.qnet(state)
        #prob = np.random.uniform(0.0, 1.0, 1)
        #if torch.from_numpy(prob).float() <= self.epsilon:  # random
        if random.random() <= self.epsilon: # random
            action = np.random.choice(range(self.action_dim))
            self.count_action_random += 1
        else:  # greedy
            action = qs.argmax(dim=-1)
            self.count_action_select += 1
        return int(action)

    def learn(self, state, action, reward, next_state, done):
        s, a, r, ns = state, action, reward, next_state

        # compute Q-Learning target with 'target network'
        with torch.no_grad():
            q_max, _ = self.qnet_target(ns).max(dim=-1, keepdim=True)
            q_target = r + self.gamma * q_max * (1 - done)

        q_val = self.qnet(s).gather(1, a)
        loss = self.criteria(q_val, q_target)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.loss = loss.item()

    def _get_loss_(self):
        return self.loss

    def _get_action_ratio(self):
        selected_ratio = self.count_action_select/(self.count_action_select+self.count_action_random)
        return selected_ratio


def prepare_training_inputs(sampled_exps, device='cpu'):
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []
    for sampled_exp in sampled_exps:
        states.append(sampled_exp[0])
        actions.append(sampled_exp[1])
        rewards.append(sampled_exp[2])
        next_states.append(sampled_exp[3])
        dones.append(sampled_exp[4])

    states = torch.cat(states, dim=0).float().to(device)
    actions = torch.cat(actions, dim=0).to(device)
    rewards = torch.cat(rewards, dim=0).float().to(device)
    next_states = torch.cat(next_states, dim=0).float().to(device)
    dones = torch.cat(dones, dim=0).float().to(device)
    return states, actions, rewards, next_states, dones


# if __name__ == '__main__':
#     net = NaiveMultiLayerPerceptron(10, 1, [20, 12], 'ReLU', 'Identity')
#     print(net)
#
#     xs = torch.randn(size=(12, 10))
#     ys = net(xs)
#     print(ys)