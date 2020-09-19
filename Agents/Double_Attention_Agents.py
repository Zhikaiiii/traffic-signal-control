from Agents.Basic_Agents import Basic_Agents
from Models.Basic_Model import Double_Attention_Model, Embedding_Layer
from Learners.Dueling_DDQN_Learner import Dueling_DDQN_Learner
from itertools import chain
from torch import optim
import numpy as np
import torch


class Double_Attention_Agents(Basic_Agents):
    def __init__(self, config, num_agents, input_dim, hidden_dim, output_dim, seq_len, neighbor_map, node_name):
        super().__init__(config, num_agents, input_dim, hidden_dim, output_dim)
        self.adj = self._get_adj(neighbor_map, node_name)
        # self.n_heads = config['n_heads']

    def _init_agents(self):
        # parameter sharing
        self.n_heads = self.config['n_heads']
        self.embedding = Embedding_Layer(self.input_dim, self.hidden_dim).to(self.device)
        self.attention = Double_Attention_Model(self.hidden_dim, self.n_heads).to(self.device)
        self.embedding_target = Embedding_Layer(self.input_dim, self.hidden_dim).to(self.device)
        self.attention_target = Double_Attention_Model(self.hidden_dim, self.n_heads).to(self.device)
        Dueling_DDQN_Learner.copy_network(self.embedding, self.embedding_target)
        Dueling_DDQN_Learner.copy_network(self.attention, self.attention_target)
        self.share_para = chain(self.embedding.parameters(), self.attention.parameters())
        self.all_para = chain(self.embedding.parameters(), self.attention.parameters())
        # init the optimizer
        for i in range(self.num_agents):
            self.agents.append(Dueling_DDQN_Learner(self.config))
            self.all_para = chain(self.all_para, self.agents[i].get_q_network().parameters())
        self.share_optimizer = optim.RMSprop(self.all_para, lr=self.lr, weight_decay=1e-4)

    def _get_embedding(self, state):
        if len(state.shape) == 4:
            state = np.expand_dims(state, axis=0)
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

    def store_experience(self, states, actions, rewards, next_states, is_dones):
        states = np.expand_dims(states, axis=0)
        next_states = np.expand_dims(next_states, axis=0)
        self.buffer.store_experience(states, actions, rewards, next_states, is_dones)

    def _get_adj(self, neighbor_map, node_name):
        adj = np.zeros((self.num_agents, self.num_agents), dtype=bool)
        for i, node in enumerate(node_name):
            adj[i][i] = True
            for neighbor in neighbor_map[node]:
                idx = node_name.index(neighbor)
                adj[i][idx] = True
        return adj

    def get_share_para(self):
        dic1 = dict(self.embedding.named_parameters())
        dic2 = dict(self.attention.named_parameters())
        return dict(dic1, **dic2)

    def get_attention_score(self, i):
        return self.attention.get_attention_score(i)