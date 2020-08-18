import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


class Embedding_Layer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Embedding_Layer, self).__init__()
        self.phase_dim, self.state_dim, self.lane_num = input_dim[0], input_dim[1], input_dim[2]
        self.output_dim = output_dim
        self.linear_phase = nn.Linear(self.phase_dim, 20)
        self.linear_state = nn.Linear(self.state_dim, 20)
        self.linear_final = nn.Linear(20 * (self.lane_num + 1), self.output_dim)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def forward(self, x):
        obs, phase = x[0], x[1]
        obs = torch.from_numpy(obs).float().to(self.device)
        phase = torch.from_numpy(phase).float().to(self.device)
        if len(obs.shape) == 2:
            obs = torch.unsqueeze(obs, 0)
            phase = torch.unsqueeze(phase, 0)
        batch_size = obs.shape[0]
        x1 = self.linear_state(obs)
        x2 = self.linear_phase(phase)
        x1 = x1.view(batch_size, -1)
        x = torch.cat((x1,x2), dim=1)
        out = self.relu1(x)
        out = self.linear_final(out)
        out = self.relu2(out)
        return out


# Single Attention Layer
class Attention_Layer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads=1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.W_k, self.W_q, self.W_v = [], [], []
        for _ in range(self.n_heads):
            self.W_k.append(nn.Linear(input_dim, hidden_dim, bias=False))
            self.W_q.append(nn.Linear(input_dim, hidden_dim, bias=False))
            self.W_v.append(nn.Linear(input_dim, hidden_dim, bias=False))
        self.out_layer = nn.Sequential(nn.Linear(hidden_dim, output_dim), nn.LeakyReLU())

    def forward(self, x, neighbors):
        # calculate Q、K、V
        all_head_keys = [W_k(neighbors) for W_k in self.W_k]
        all_head_values = [W_v(neighbors) for W_v in self.W_v]
        all_head_queries = [W_q(x) for W_q in self.W_q]
        all_attention_scores = []
        batch_size = x.shape[0]
        # calculate attention score
        res = torch.zeros((batch_size, 1, self.hidden_dim))
        for curr_head_key, curr_head_value, curr_head_query in zip(all_head_keys, all_head_values, all_head_queries):
            attention_logits = torch.matmul(curr_head_query, curr_head_key.permute(0, 2, 1))
            attention_scores = F.softmax(attention_logits, dim=2)
            all_attention_scores.append(attention_scores)
            res += torch.matmul(attention_scores, curr_head_value)
        res = res / self.n_heads
        res = self.out_layer(res)
        res = res.squeeze(1)
        return res


class Model(nn.Module):
    def __init__(self, embedding_layer, attention_layer, input_dim, output_dim):
        super().__init__()
        self.embedding = embedding_layer
        self.attention = attention_layer
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, state):
        self_state, neighbors_state = state[0], state[1]
        self_embedding = self.embedding(self_state).unsqueeze(1)
        neighbors_embedding = None
        for neighbor in neighbors_state:
            if neighbors_embedding is None:
                neighbors_embedding = self.embedding(neighbor).unsqueeze(1)
            else:
                neighbors_embedding = torch.cat((neighbors_embedding, self.embedding(neighbor).unsqueeze(1)), dim=1)
        # 如果batch_size不为1要处理
        out = self.attention(self_embedding, neighbors_embedding)
        out = self.linear1(out)
        out = self.relu(out)
        return out

