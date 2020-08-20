import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from Models.Basic_Model import Embedding_Layer, Multi_Attention_Layer


class Attention_Model(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.pre_train = False
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # if embedding_layer:
        #     self.embedding = embedding_layer
        #     self.attention = attention_layer
        #     self.linear1 = nn.Linear(self.input_dim, self.output_dim)
        # else:
        self.embedding = Embedding_Layer(self.input_dim, self.hidden_dim)
        self.attention = Multi_Attention_Layer(self.hidden_dim)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(self.hidden_dim, self.output_dim)
            # for para in self.attention.parameters():
            #     para.requires_grad = False

    '''
    input:
    state: [batch, lane_num+1, feature_dim, neighbors_num]
    output:
    [batch, feature_out]
    '''
    def forward(self, state):
        agent_state, neighbors_state = state[:, :, :, 0], state[:, :, :, 0:]
        neighbors_state = np.transpose(neighbors_state, (0, 3, 1, 2))
        batch_size = state.shape[0]
        neighbors_num = state.shape[-1]
        # agent_state = torch.from_numpy(agent_state).float().to(self.device).unsqueeze(1).view(batch_size, 1, -1)
        # neighbors_state = torch.from_numpy(np.transpose(neighbors_state, (0, 3, 1, 2))).\
        #     float().to(self.device).contiguous().view(batch_size, neighbors_num, -1)
        agent_embedding = self.embedding(agent_state)
        # if self.pre_train:
        #     out = self.linear1(agent_embedding)
        #     return out
        agent_embedding = agent_embedding.unsqueeze(1)
        neighbors_embedding = None
        for i in range(neighbors_num):
            if i == 0:
                neighbors_embedding = self.embedding(neighbors_state[:, i]).unsqueeze(1)
            else:
                neighbors_embedding = torch.cat((neighbors_embedding,
                                                 self.embedding(neighbors_state[:, i]).unsqueeze(1)), dim=1)
        agent_embedding = agent_embedding.permute((1, 0, 2))
        neighbors_embedding = neighbors_embedding.permute((1, 0, 2))
        out = self.attention(agent_embedding, neighbors_embedding)
        out = out.squeeze(0)
        out = self.relu(out)
        # out = self.attention(agent_state, neighbors_state)
        out = self.linear1(out)
        return out

    def change_mode(self):
        self.pre_train = False
        for para in self.embedding.parameters():
            para.requires_grad = False
        for para in self.attention.parameters():
            para.requires_grad = True

    def set_layer_para(self, embedding_layer=None, attention_layer=None):
        if embedding_layer is not None:
            self.embedding = embedding_layer
        if attention_layer is not None:
            self.attention = attention_layer
