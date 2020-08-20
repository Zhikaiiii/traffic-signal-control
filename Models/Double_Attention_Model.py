import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from Models.Basic_Model import Embedding_Layer, Multi_Attention_Layer
from Models.Attention_Model import Attention_Model


class Double_Attention_Model(Attention_Model):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__(input_dim, output_dim, hidden_dim)
        self.temporal_attention = Multi_Attention_Layer(self.hidden_dim)
        # self.seq_len = seq_len

    '''
    input:
    state: [batch, seq_len, lane_num+1, feature_dim, neighbors_num]
    output:
    [batch, feature_out]
    '''
    def forward(self, state):
        # [batch, neighbors_num, seq_len, lane_num+1, feature_dim]
        state = np.transpose(state, (0, 4, 1, 2, 3))
        batch_size, neighbor_num, seq_len = state.shape[0], state.shape[1], state.shape[2]
        # self-current state embedding
        # [seq_len, batch, feature_dim]
        agent_embedding = self.embedding(state[:, 0, -1]).unsqueeze(1).permute((1, 0, 2))
        spatial_embedding = torch.zeros((neighbor_num, batch_size, self.hidden_dim)).to(self.device)
        # temporal attention
        for i in range(neighbor_num):
            temporal_embedding = torch.zeros((seq_len, batch_size, self.hidden_dim)).to(self.device)
            for j in range(seq_len):
                temporal_embedding[j, :] = self.embedding(state[:, i, j]).unsqueeze(1).permute((1, 0, 2))
            spatial_embedding[i, :] = self.temporal_attention(agent_embedding, temporal_embedding)
        out = self.attention(agent_embedding, spatial_embedding).squeeze(0)
        out = self.relu(out)
        out = self.linear1(out)
        return out

    def set_layer_para(self, embedding_layer=None, spatial_attention_layer=None, temporal_attention_layer=None):
        if embedding_layer is not None:
            self.embedding = embedding_layer
        if spatial_attention_layer is not None:
            self.attention = spatial_attention_layer
        if temporal_attention_layer is not None:
            self.temporal_attention = temporal_attention_layer