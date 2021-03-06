from Agents.Basic_Agents import Basic_Agents
from Models.Double_Attention_Model import Double_Attention_Model
from Models.Attention_Model import Attention_Model
from Models.Basic_Model import Embedding_Layer, Attention_Model
from Learners.Dueling_DDQN_Learner import Dueling_DDQN_Learner
import torch
from itertools import chain
from torch import optim
import numpy as np


# 所有交叉口的Agent的集合
class Attention_Agents(Basic_Agents):
    def __init__(self, config, num_agents, input_dim, hidden_dim, output_dim, neighbor_map, node_name):
        super().__init__(config, num_agents, input_dim, hidden_dim, output_dim)
        # self.q_network = Attention_Model(self.input_dim, self.output_dim, self.hidden_dim).to(self.device)
        # self.q_network_target = Attention_Model(self.input_dim, self.output_dim, self.hidden_dim).to(self.device)
        self._init_agents()

        self.adj = self._get_adj(neighbor_map, node_name)

    def _init_agents(self):
        # parameter sharing
        self.n_heads = self.config['n_heads']
        self.embedding = Embedding_Layer(self.input_dim, self.hidden_dim).to(self.device)
        self.attention = Attention_Model(self.hidden_dim, self.n_heads).to(self.device)
        self.embedding_target = Embedding_Layer(self.input_dim, self.hidden_dim).to(self.device)
        self.attention_target = Attention_Model(self.hidden_dim, self.n_heads).to(self.device)
        Dueling_DDQN_Learner.copy_network(self.embedding, self.embedding_target)
        Dueling_DDQN_Learner.copy_network(self.attention, self.attention_target)
        # self.share_optimizer = optim.Adam(chain(self.embedding.parameters(), self.attention.parameters()), lr=1e-3)
        self.share_para = chain(self.embedding.parameters(), self.attention.parameters())
        self.all_para = chain(self.embedding.parameters(), self.attention.parameters())
        # init the optimizer
        for i in range(self.num_agents):
            self.agents.append(Dueling_DDQN_Learner(self.config))
            self.all_para = chain(self.all_para, self.agents[i].get_q_network().parameters())
        # self.all_para = chain(self.all_para)
        self.share_optimizer = optim.RMSprop(self.all_para, lr=self.lr, weight_decay=1e-4)

    def _get_embedding(self, state):
        state_embedding = self.embedding(state)
        state_attention = self.attention(state_embedding, self.adj)
        return state_attention

    def _get_embedding_target(self, state):
        state_embedding_target = self.embedding_target(state)
        state_attention_target = self.attention_target(state_embedding_target, self.adj)
        return state_attention_target

    def _update_sharing_target_network(self):
        Dueling_DDQN_Learner.soft_update_of_target_network(self.embedding, self.embedding_target, self.tau)
        Dueling_DDQN_Learner.soft_update_of_target_network(self.attention, self.attention_target, self.tau)

    def _get_adj(self, neighbor_map, node_name):
        adj = np.zeros((self.num_agents, self.num_agents), dtype=bool)
        for i, node in enumerate(node_name):
            adj[i][i] = True
            for neighbor in neighbor_map[node]:
                idx = node_name.index(neighbor)
                # idx = int(neighbor[2:] - 1)
                adj[i][idx] = True
        return adj

    def get_attention_score(self, i):
        att = self.attention.get_attention_score(i, self.adj)
        return att

    def get_share_para(self):
        dic1 = dict(self.embedding.named_parameters())
        dic2 = dict(self.attention.named_parameters())
        return dict(dic1, **dic2)























class Double_Attention_Agents(Attention_Agents):
    def __init__(self, config, num_agents, input_dim, hidden_dim, output_dim):
        super().__init__(config, num_agents, input_dim, hidden_dim, output_dim)
        self._init_agents()

    def _init_agents(self):
        self.embedding = Embedding_Layer(self.input_dim, self.hidden_dim).to(self.device)
        self.attention = Attention_Model(self.hidden_dim).to(self.device)
        self.temporal_attention = Attention_Model(self.hidden_dim).to(self.device)
        self.embedding_target = Embedding_Layer(self.input_dim, self.hidden_dim).to(self.device)
        self.attention_target = Attention_Model(self.hidden_dim).to(self.device)
        self.temporal_attention_target = Attention_Model(self.hidden_dim).to(self.device)
        Dueling_DDQN_Learner.copy_network(self.embedding, self.embedding_target)
        Dueling_DDQN_Learner.copy_network(self.attention, self.attention_target)
        Dueling_DDQN_Learner.copy_network(self.temporal_attention, self.temporal_attention_target)
        for i in range(self.num_agents):
            q_network = Double_Attention_Model(self.input_dim, self.output_dim, self.hidden_dim).to(self.device)
            q_network_target = Double_Attention_Model(self.input_dim, self.output_dim, self.hidden_dim).to(self.device)
            q_network.set_layer_para(self.embedding, self.attention, self.temporal_attention)
            q_network_target.set_layer_para(self.embedding_target, self.attention_target, self.temporal_attention_target)
            self.agents[i].set_q_network(q_network, q_network_target)

# def change_mode(self):
        # self.q_network.change_mode()
        # self.q_network_target.change_mode()
        # for i in range(self.num_agents):
        #     self.agents[i].q_network_current.change_mode()
        #     self.agents[i].q_network_target.change_mode()
        # self.embedding = Embedding_Layer(self.input_dim, self.hidden_dim[0])
        # self.attention = Attention_Layer(self.hidden_dim[0], self.hidden_dim[1], self.hidden_dim[2])
        # self.embedding_target = Embedding_Layer(self.input_dim, self.hidden_dim[0])
        # self.attention_target = Attention_Layer(self.hidden_dim[0], self.hidden_dim[1], self.hidden_dim[2])
        # Dueling_DDQN_Learner.copy_network(self.embedding, self.embedding_target)
        # Dueling_DDQN_Learner.copy_network(self.attention, self.attention_target)
    # def get_action(self, i, obs):
    #     return self.agents[i].step(obs)
    # def store_experience(self, i, obs, action, reward, next_obs, is_done):
    #     self.agents[i].store_experience(obs, action, reward, next_obs, is_done)