import torch
import numpy as np
from Replay_Buffer import Replay_Buffer
from Models.Basic_Model import Embedding_Layer
import torch.nn as nn


class Base_Learner:
    def __init__(self, config):

        self.seed = config['sim_seed']
        # self.max_step = config['max_step']
        self.total_episodes = config['total_episodes']
        self.update_step = config['update_step']
        self.curr_step = 0

        # self.agent_type = config['agent_type']
        # 超参数
        self.tau = config['tau']
        self.gamma = config['gamma']
        self.epsilon_init = config['epsilon_init']
        self.epsilon = self.epsilon_init
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # 状态数和行动数
        # self.num_states_obs = config['num_states_obs']
        # self.num_states_phase = config['num_states_phase']
        # self.num_states_lanes = config['num_states_lanes']
        # self.num_actions = config['num_actions']
        self.input_dim = config['hidden_dim']
        self.output_dim = config['num_actions'] + 1

        # Replay Buffer相关参数
        # self.batch_size = config['batch_size']
        # self.buffer_size = config['buffer_size']
        # self.replay_buffer = Replay_Buffer(self.buffer_size, self.batch_size)

    # 需要由子类重写
    def step(self, state):
        """Takes a step in the game. This method must be overriden by any agent"""
        raise ValueError("Step needs to be implemented by the agent")

    # 创建模型
    # Learner只包含最终的Q网络
    def create_model(self, input_dim, output_dim):
        # if self.agent_type == 'IQL':
        model = nn.Sequential(nn.ReLU(), nn.Linear(input_dim, output_dim)).to(self.device)
        # model = Embedding_Layer(input_dim, output_dim).to(self.device)
        # else:
        #     model = Model(embedding, attention, input_dim, output_dim).to(self.device)
        return model

    def select_actions(self, actions):
        actions = actions.cpu().numpy()
        if np.random.random() < self.epsilon:
            selected_action = np.random.randint(0, self.num_actions)
        else:
            selected_action = np.argmax(actions)
        return selected_action

    # experience replay
    # def store_experience(self, state, action, reward, next_state, is_done):
    #     self.replay_buffer.store_experience(state, action, reward, next_state, is_done)
    #
    # def sample_experience(self):
    #     states, actions, rewards, next_states, is_dones = self.replay_buffer.sample_experience()
    #     return states, actions, rewards, next_states, is_dones

    def update_epsilon_exploration(self, current_episode):
        self.epsilon = max(0.01, 0.01 + self.epsilon_init * (1 - 2 * current_episode/ self.total_episodes))

    # def reset(self):
    #     self.curr_step = 0

    # 复制网络结构
    @staticmethod
    def copy_network(from_model, to_model):
        """Copies model parameters from from_model to to_model"""
        for to_para, from_para in zip(to_model.parameters(), from_model.parameters()):
            to_para.data.copy_(from_para.data.clone())

    @staticmethod
    def soft_update_of_target_network(current_model, target_model, tau):
        for target_param, current_param in zip(target_model.parameters(), current_model.parameters()):
            target_param.data.copy_(tau * current_param.data + (1.0 - tau) * target_param.data)