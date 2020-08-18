import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from Models.Basic_Model import RNN_Layer


class AttL_Model(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=None, embedding_layer=None, attention_layer=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.pre_train = True
        # if embedding_layer:
        #     self.embedding = embedding_layer
        #     self.attention = attention_layer
        #     self.linear1 = nn.Linear(self.input_dim, self.output_dim)
        # else:
        self.embedding = Embedding_Layer(self.input_dim, self.hidden_dim[0])
        self.rnn = RNN_Layer(self.hidden_dim[0], self.hidden_dim[1], self.hidden_dim[2])
        self.attention = Attention_Layer(self.hidden_dim[2], self.hidden_dim[3], self.hidden_dim[4])
        self.linear1 = nn.Linear(self.hidden_dim[4], self.output_dim)
        for para in self.attention.parameters():
            para.requires_grad = False

    def forward(self, state, h_state):
        agent_state, neighbors_state = state[:, :, :, 0], state[:, :, :, 0:]
        neighbors_state = np.transpose(neighbors_state, (0, 3, 1, 2))

        agent_h_state, neighbors_h_state = h_state[:, :, :, 0], h_state[:, :, :, 0:]
        agent_embedding = self.embedding(agent_state)
        agent_hidden = self.rnn(agent_state, agent_h_state)
        if self.pre_train:
            out = self.linear1(agent_hidden)
            return out
        agent_hidden = agent_hidden.unsqueeze(1)
        neighbos_num = neighbors_state.shape[1]
        neighbors_hidden = None
        for i in range(neighbos_num):
            if i == 0:
                neighbors_hidden = self.rnn(self.embedding(neighbors_state[:, i]),
                                               neighbors_h_state[:, i]).unsqueeze(1)
            else:
                neighbors_hidden = torch.cat((neighbors_hidden, self.rnn(self.embedding(neighbors_state[:, i]),
                                                                         neighbors_h_state[:, i]).unsqueeze(1)), dim=1)
        # 如果batch_size不为1要处理
        out = self.attention(agent_hidden, neighbors_hidden)
        out = self.linear1(out)
        return out

    def change_mode(self):
        self.pre_train = False
        for para in self.embedding.parameters():
            para.requires_grad = False
        for para in self.attention.parameters():
            para.requires_grad = True
