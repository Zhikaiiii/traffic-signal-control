import logging
import os
import torch
import numpy as np
import random
from torch import nn
from torch import optim
from Replay_Buffer import Replay_Buffer
from model import Model

class Base_Agent:
    def __init__(self, config):

        self.seed = config['sim_seed']
        # self.max_step = config['max_step']
        self.total_episodes = config['total_episodes']
        self.update_step = config['update_step']
        self.curr_step = 0

        # 超参数
        self.tau = config['tau']
        self.gamma = config['gamma']
        self.epsilon_init = config['epsilon_init']
        self.epsilon = self.epsilon_init
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # 状态数和行动数
        self.num_states_obs = config['num_states_obs']
        self.num_states_phase = config['num_states_phase']
        self.num_states_lanes = config['num_states_lanes']
        self.num_actions = config['num_actions']

        # Replay Buffer相关参数
        self.batch_size = config['batch_size']
        self.buffer_size = config['buffer_size']
        self.replay_buffer = Replay_Buffer(self.buffer_size, self.batch_size, self.seed)

    def set_random_seeds(self, random_seed):
        """Sets all possible random seeds so results can be reproduced"""
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
            torch.cuda.manual_seed(random_seed)

    # 需要由子类重写
    def step(self, state):
        """Takes a step in the game. This method must be overriden by any agent"""
        raise ValueError("Step needs to be implemented by the agent")

    # 创建模型
    def create_model(self, input_dim, output_dim):
        return Model(input_dim, output_dim)

    def select_actions(self, actions):
        actions = actions.numpy()
        if np.random.random() < self.epsilon:
            selected_action = np.random.randint(0, self.num_actions)
        else:
            selected_action = np.argmax(actions)
        return selected_action

    # experience replay
    def store_experience(self, state, action, reward, next_state, is_done):
        self.replay_buffer.store_experience(state, action, reward, next_state, is_done)

    def sample_experience(self):
        states, actions, rewards, next_states, is_dones = self.replay_buffer.sample_experience()
        return states, actions, rewards, next_states, is_dones

    def update_epsilon_exploration(self, current_episode):
        if self.epsilon <= 0.02:
            return
        else:
            self.epsilon = self.epsilon_init * (1 - 2 * current_episode/ self.total_episodes)

    def soft_update_of_target_network(self, current_model, target_model):
        """Updates the target network in the direction of the local network but by taking a step size
        less than one so the target network's parameter values trail the local networks. This helps stabilise training"""
        for target_param, current_param in zip(target_model.parameters(), current_model.parameters()):
            target_param.data.copy_(self.tau * current_param.data + (1.0 - self.tau) * target_param.data)

    def reset(self):
        self.curr_step = 0

    # 复制网络结构
    @staticmethod
    def copy_network(from_model, to_model):
        """Copies model parameters from from_model to to_model"""
        for to_model, from_model in zip(to_model.parameters(), from_model.parameters()):
            to_model.data.copy_(from_model.data.clone())
