import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from Models.Basic_Model import RNN_Model, Embedding_Layer, Attention_Model


class LSTM_Attention_Model(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.embedding = Embedding_Layer(self.input_dim, self.hidden_dim)
        self.rnn = RNN_Model(self.hidden_dim, self.hidden_dim)
        self.attention = Attention_Model(self.hidden_dim)
        self.relu = nn.ReLU()
        self.linear_out = nn.Linear(self.hidden_dim, self.output_dim)
        self.hidden, self.cell = None, None

    def forward(self, state):
        agent_state, neighbors_state = state[:, :, :, 0], state[:, :, :, 0:]
        neighbors_state = np.transpose(neighbors_state, (0, 3, 1, 2))
        # agent_h_state, neighbors_h_state = h_state[:, :, :, 0], h_state[:, :, :, 0:]
        agent_embedding = self.embedding(agent_state)
        agent_hidden, agent_cell = self.rnn(agent_state, agent_h_state)
        agent_hidden = agent_hidden.unsqueeze(1)
        neighbors_num = neighbors_state.shape[1]
        neighbors_hidden = None
        for i in range(neighbors_num):
            if i == 0:
                neighbors_hidden = self.rnn(self.embedding(neighbors_state[:, i]),
                                               neighbors_h_state[:, i]).unsqueeze(1)
            else:
                neighbors_hidden = torch.cat((neighbors_hidden, self.rnn(self.embedding(neighbors_state[:, i]),
                                                                         neighbors_h_state[:, i]).unsqueeze(1)), dim=1)
        # 如果batch_size不为1要处理
        out = self.attention(agent_hidden, neighbors_hidden)
        out = self.linear1(out)
        # store the hidden state and cell state
        self.hidden = agent_hidden
        self.cell = agent_cell
        return out

    def set_layer_para(self, embedding_layer=None, attention_layer=None, rnn_layer=None):
        if embedding_layer is not None:
            self.embedding = embedding_layer
        if attention_layer is not None:
            self.attention = attention_layer
        if rnn_layer is not None:
            self.rnn = rnn_layer

    def get_state(self):
        hidden_state = self.hidden.cpu().detach().numpy()
        cell_state = self.cell.cpu().detach().numpy()
        return hidden_state, cell_state
