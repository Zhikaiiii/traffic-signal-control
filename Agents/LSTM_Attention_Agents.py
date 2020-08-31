from Learners.Dueling_DDQN_Learner import Dueling_DDQN_Learner
from Agents.Attention_Agents import Attention_Agents
from torch import optim
from Models.Basic_Model import Embedding_Layer, RNN_Model, Attention_Model
from itertools import chain
import numpy as np


class LSTM_Attention_Agents(Attention_Agents):
    def __init__(self, config, num_agents, input_dim, hidden_dim, output_dim, neighbor_map, node_name):
        super().__init__(config, num_agents, input_dim, hidden_dim, output_dim, neighbor_map, node_name)
        self._init_agents()
        self.hidden = np.zeros((self.num_agents, self.hidden_dim))
        self.hidden_target = np.zeros((self.num_agents, self.hidden_dim))
        self.cell = np.zeros((self.num_agents, self.hidden_dim))
        self.cell_target = np.zeros((self.num_agents, self.hidden_dim))

    def _init_agents(self):
        # parameter sharing
        self.embedding = Embedding_Layer(self.input_dim, self.hidden_dim).to(self.device)
        self.rnn = RNN_Model(self.hidden_dim, self.num_agents).to(self.device)
        self.attention = Attention_Model(self.hidden_dim).to(self.device)
        self.embedding_target = Embedding_Layer(self.input_dim, self.hidden_dim).to(self.device)
        self.rnn_target = RNN_Model(self.hidden_dim, self.num_agents).to(self.device)
        self.attention_target = Attention_Model(self.hidden_dim).to(self.device)
        Dueling_DDQN_Learner.copy_network(self.embedding, self.embedding_target)
        Dueling_DDQN_Learner.copy_network(self.rnn, self.rnn_target)
        Dueling_DDQN_Learner.copy_network(self.attention, self.attention_target)
        self.share_para = chain(self.embedding.parameters(), self.attention.parameters(), self.rnn.parameters())
        self.all_para = chain(self.embedding.parameters(), self.attention.parameters(), self.rnn.parameters())
        # init the optimizer
        for i in range(self.num_agents):
            self.agents.append(Dueling_DDQN_Learner(self.config))
            self.all_para = chain(self.all_para, self.agents[i].get_q_network().parameters())
        # self.all_para = chain(self.all_para)
        self.share_optimizer = optim.Adam(self.all_para, lr=1e-3)

    def _get_embedding(self, state):
        state_embedding = self.embedding(state)
        batch_size = state.shape[0]
        if batch_size == 1:
            # get hidden state to store
            self.hidden, self.cell = self.rnn.get_hidden_state()
            self.hidden_target, self.cell_target = self.rnn_target.get_hidden_state()
            state_hidden, _ = self.rnn(state_embedding)
        else:
            state_hidden, _ = self.rnn(state_embedding, self.hidden, self.cell)
        state_attention, _ = self.attention(state_hidden, self.adj)
        return state_attention

    def _get_embedding_target(self, state):
        state_embedding_target = self.embedding_target(state)
        batch_size = state.shape[0]
        if batch_size == 1:
            state_hidden_target, _ = self.rnn_target(state_embedding_target)
        else:
            state_hidden_target, _ = self.rnn_target(state_embedding_target, self.hidden_target, self.cell_target)
        state_attention_target, _ = self.attention_target(state_hidden_target, self.adj)
        return state_attention_target

    def _update_sharing_target_network(self):
        Dueling_DDQN_Learner.soft_update_of_target_network(self.embedding, self.embedding_target, self.tau)
        Dueling_DDQN_Learner.soft_update_of_target_network(self.rnn, self.rnn_target, self.tau)
        Dueling_DDQN_Learner.soft_update_of_target_network(self.attention, self.attention_target, self.tau)

    def store_experience(self, states, actions, rewards, next_states, is_dones):
        hidden = np.stack((self.hidden, self.hidden_target, self.cell, self.cell_target), axis=1)
        self.buffer.store_experience(states, actions, rewards, next_states, is_dones, hidden)

    def sample_experience(self):
        states, actions, rewards, next_states, is_dones, hidden = self.buffer.sample_experience()
        # get  hidden state
        self.hidden = hidden[:, 0]
        self.hidden_target = hidden[:, 1]
        self.cell = hidden[:, 2]
        self.cell_target = hidden[:, 3]
        return states, actions, rewards, next_states, is_dones

