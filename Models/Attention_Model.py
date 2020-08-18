import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from Models.Basic_Model import Embedding_Layer, Multi_Attention_Layer

# class Embedding_Layer(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(Embedding_Layer, self).__init__()
#         self.phase_dim, self.state_dim, self.lane_num = input_dim[0], input_dim[1], input_dim[2]
#         self.output_dim = output_dim
#         self.linear_phase = nn.Linear(self.phase_dim, 32)
#         self.linear_state = nn.Linear(self.state_dim, 32)
#         self.linear_final = nn.Linear(32 * (self.lane_num + 1), self.output_dim)
#         self.relu1 = nn.ReLU()
#         self.relu2 = nn.ReLU()
#         self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
#
#     def forward(self, x):
#         obs, phase = x[:, :-1], x[:, -1:]
#         obs = torch.from_numpy(obs).float().to(self.device)
#         phase = torch.from_numpy(phase).float().to(self.device)
#         batch_size = obs.shape[0]
#         x1 = self.linear_state(obs)
#         x2 = self.linear_phase(phase)
#         x1 = x1.view(batch_size, -1)
#         x2 = x2.view(batch_size, -1)
#         x = torch.cat((x1, x2), dim=1)
#         # out = self.relu1(x)
#         out = self.linear_final(x)
#         # out = self.relu2(out)
#         return out
#


class Model(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=None, embedding_layer=None, attention_layer=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.pre_train = True
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if embedding_layer:
            self.embedding = embedding_layer
            self.attention = attention_layer
            self.linear1 = nn.Linear(self.input_dim, self.output_dim)
        else:
            self.embedding = Embedding_Layer(self.input_dim, self.hidden_dim)
            self.attention = Multi_Attention_Layer(self.hidden_dim)
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
        # out = self.attention(agent_state, neighbors_state)
        out = self.linear1(out)
        return out

    def change_mode(self):
        self.pre_train = False
        for para in self.embedding.parameters():
            para.requires_grad = False
        for para in self.attention.parameters():
            para.requires_grad = True
