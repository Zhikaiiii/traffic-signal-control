from Learners.Dueling_DDQN_Learner import Dueling_DDQN_Learner
from Replay_Buffer import Replay_Buffer
from Models.Basic_Model import Embedding_Layer
import torch
from itertools import chain
from torch import optim
import torch.nn.functional as F
import numpy as np


# Agents的集合
class Basic_Agents:
    def __init__(self, config, num_agents, input_dim, hidden_dim, output_dim):

        self.num_agents = num_agents
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.config = config

        # Replay Buffer相关参数
        self.batch_size = config['batch_size']
        self.buffer_size = config['buffer_size']
        self.buffer = Replay_Buffer(self.buffer_size, self.batch_size)

        self.lr = config['lr']
        self.tau = config['tau']
        self.agents = []

        self.update_step = config['update_step']
        self.curr_step = 0
        self._init_agents()

    def _init_agents(self):
        self.embedding = Embedding_Layer(self.input_dim, self.hidden_dim).to(self.device)
        self.embedding_target = Embedding_Layer(self.input_dim, self.hidden_dim).to(self.device)
        Dueling_DDQN_Learner.copy_network(self.embedding, self.embedding_target)

        self.share_para = self.embedding.parameters()
        self.all_para = self.embedding.parameters()
        # init the optimizer
        for i in range(self.num_agents):
            self.agents.append(Dueling_DDQN_Learner(self.config))
            self.all_para = chain(self.all_para, self.agents[i].get_q_network().parameters())
            # para = chain(self.embedding.parameters(), self.agents[i].get_q_network().parameters())
            # self.optimizer.append(optim.Adam(self.agents[i].get_q_network().parameters(), lr=1e-3))
        # self.all_para = chain(self.all_para)
        self.share_optimizer = optim.RMSprop(self.all_para, lr=self.lr, weight_decay=1e-4)

    def get_agent(self, i):
        return self.agents[i]

    def step(self, state, test=False):
        state_embedding = self._get_embedding(state)
        action = []
        for i in range(self.num_agents):
            action.append(self.agents[i].step(state_embedding[:, i], test))
        action = np.asarray(action)
        self.curr_step += 1
        return action

    def learn(self):
        # if self.curr_step > 0 and self.curr_step % self.update_step == 0:
        for i in range(self.update_step):
            states, actions, rewards, next_states, is_dones = self.sample_experience()
            actions = torch.from_numpy(actions).long().to(self.device)
            rewards = torch.from_numpy(rewards).float().to(self.device)
            is_dones = torch.from_numpy(is_dones).float().to(self.device)
            states_embedding = self._get_embedding(states)
            next_states_embedding = self._get_embedding(next_states)
            next_states_embedding_target = self._get_embedding_target(next_states)
            total_loss = 0
            for i in range(self.num_agents):
                actions_values_current = self.agents[i].cal_current_actions_value(next_states_embedding[:, i],
                                                                                  next_states_embedding_target[:, i],
                                                                                  rewards[:, i], is_dones)
                actions_values_expected = self.agents[i].cal_expected_actions_value(states_embedding[:, i], actions[:, i])
                loss = F.mse_loss(actions_values_expected, actions_values_current)
                # loss.backward(retain_graph=True)
                total_loss += loss
                # 反向传播
                # self.optimizer[i].zero_grad()
            self.share_optimizer.zero_grad()
            total_loss.backward()
            # self._scale_shared_grads()
            torch.nn.utils.clip_grad_value_(self.all_para, 1)
            self.share_optimizer.step()
            for i in range(self.num_agents):
                # torch.nn.utils.clip_grad_value_(self.agents[i].q_network_current.parameters(), 1)
                # self.optimizer[i].step()
                # 更新target net
                Dueling_DDQN_Learner.soft_update_of_target_network(self.agents[i].q_network_current,
                                                                   self.agents[i].q_network_target, self.tau)
            self._update_sharing_target_network()
            # self.share_optimizer.zero_grad()

    def get_share_para(self):
        return dict(self.embedding.named_parameters())

    def store_experience(self, states, actions, rewards, next_states, is_dones):
        self.buffer.store_experience(states, actions, rewards, next_states, is_dones)

    def sample_experience(self):
        states, actions, rewards, next_states, is_dones = self.buffer.sample_experience()
        return states, actions, rewards, next_states, is_dones

    def _get_embedding(self, state):
        return self.embedding(state)

    def _get_embedding_target(self, state):
        return self.embedding_target(state)

    def _update_sharing_target_network(self):
        Dueling_DDQN_Learner.soft_update_of_target_network(self.embedding, self.embedding_target, self.tau)

    def get_attention_score(self, i):
        return -1

    def _scale_shared_grads(self):
        """
        Scale gradients for parameters that are shared since they accumulate
        gradients from the critic loss function multiple times
        """
        for p in self.share_para:
            p.grad.data.mul_(1. / self.num_agents)

    def save_model(self, path):
        share_model_name = path + '/share_model.pkl'
        torch.save(self.embedding.state_dict(), share_model_name)
        for i in range(self.num_agents):
            unique_model_name = path + '/q_network_%d.pkl' % i
            torch.save(self.agents[i].q_network_current.state_dict(), unique_model_name)

    def load_model(self, path):
        share_model_name = path + '/share_model.pkl'
        self.embedding.load_state_dict(torch.load(share_model_name, map_location=self.device))
        for i in range(self.num_agents):
            unique_model_name = path + '/q_network_%d.pkl' % i
            self.agents[i].q_network_current.load_state_dict(torch.load(unique_model_name, map_location=self.device))