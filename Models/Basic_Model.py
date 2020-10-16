import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

'''
x:[batch_size, num_agents, num_phases, num_states]
'''
class Base_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.linear3 = nn.Linear(output_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.linear4 = nn.Linear(hidden_dim, output_dim + 1)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def forward(self, x):
        x = torch.from_numpy(x).float().to(self.device)
        x1 = self.linear1(x)
        x2 = self.relu1(x1)
        x3 = self.linear2(x2)
        x3 = x3.squeeze(dim=-1)
        score = self.softmax(x3)
        out = self.linear3(score)
        out = self.relu2(out)
        out = self.linear4(out)
        return out


class Embedding_Layer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Embedding_Layer, self).__init__()
        self.phase_dim, self.state_dim, self.lane_num = input_dim[0], input_dim[1], input_dim[2]
        self.output_dim = output_dim
        self.linear_phase = nn.Linear(self.phase_dim, 32, bias=False)
        self.linear_state = nn.Linear(self.state_dim, 32, bias=False)
        self.linear_final = nn.Linear(32 * (self.lane_num + 1), self.output_dim, bias=False)
        self.relu1 = nn.ReLU()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    '''
    包含时间维度:
    x: [batch_size, seq_len, node_num, input_dim]
    不包含：
    x: [batch_size, node_num, input_dim]
    '''

    def forward(self, x):
        if len(x.shape) == 4:
            obs, phase = x[:, :, :-1], x[:, :, -1:]
            batch_size = obs.shape[0]
            node_num = obs.shape[1]
            obs = torch.from_numpy(obs).float().to(self.device)
            phase = torch.from_numpy(phase).float().to(self.device)
            x1 = self.linear_state(obs)
            x2 = self.linear_phase(phase)
            x1 = x1.view(batch_size, node_num, -1)
            x2 = x2.view(batch_size, node_num, -1)
            x = torch.cat((x1, x2), dim=2)
        else:
            obs, phase = x[:, :, :, :-1], x[:, :, :, -1:]
            batch_size = obs.shape[0]
            seq_len = obs.shape[1]
            node_num = obs.shape[2]
            obs = torch.from_numpy(obs).float().to(self.device)
            phase = torch.from_numpy(phase).float().to(self.device)
            x1 = self.linear_state(obs)
            x2 = self.linear_phase(phase)
            x1 = x1.view(batch_size, seq_len, node_num, -1)
            x2 = x2.view(batch_size, seq_len, node_num, -1)
            x = torch.cat((x1, x2), dim=3)
        # if len(obs.shape) == 2:
        #     obs = torch.unsqueeze(obs, 0)
        #     phase = torch.unsqueeze(phase, 0)

        out = self.relu1(x)
        out = self.linear_final(out)
        return out


class RNN_Model(nn.Module):
    def __init__(self, hidden_dim, num_agents):
        super(RNN_Model, self).__init__()
        # self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.num_agents = num_agents
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTMCell(hidden_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.h = torch.zeros((1, self.num_agents, self.hidden_dim)).to(self.device)
        self.c = torch.zeros((1, self.num_agents, self.hidden_dim)).to(self.device)

    def forward(self, x, h=None, c=None):
        if h is None:
            h = self.h
        else:
            h = torch.from_numpy(h).float().to(self.device)
        if c is None:
            c = self.c
        else:
            c = torch.from_numpy(c).float().to(self.device)
        batch_size = x.shape[0]
        h_out = torch.zeros_like(x)
        c_out = torch.zeros_like(x)
        # x = self.relu1(self.linear1(x))
        for i in range(self.num_agents):
            h_out[:, i], c_out[:, i] = self.lstm(x[:, i], (h[:, i], c[:, i]))
        if batch_size == 1:
            self.h = h_out.clone()
            self.c = c_out.clone()
        return h_out, c_out

    def get_hidden_state(self):
        return self.h.cpu().detach().numpy(), self.c.cpu().detach().numpy()


class Double_Attention_Model(nn.Module):
    def __init__(self, embed_dim, n_heads=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.temporal_attention = nn.MultiheadAttention(self.embed_dim, self.n_heads, bias=False)
        self.spatial_attention = nn.MultiheadAttention(self.embed_dim, self.n_heads, bias=False)
        self.attention_score = [torch.zeros(1)]

    '''
    x: [batch_size, seq_len, num_agents, embedding_dim]
    adj:[num_agents, num_agents] 
    '''

    def forward(self, x, adj):
        # current state embedding
        # curr_embed = x[:, :, -1]
        batch_size = x.shape[0]
        num_agents = x.shape[2]
        out = torch.zeros((x.shape[2], x.shape[0], x.shape[3])).to(self.device)
        self.attention_score = [torch.zeros(1)] * num_agents
        # attention_score = []
        # attention_score = torch.zeros((batch_size, num_agents, num_agents))
        for i in range(num_agents):
            temporal_embedding = []
            curr_embed = x[:, -1, i].unsqueeze(0)
            # curr_embed = curr_embed.permute((2, 0, 1))
            for j in range(num_agents):
                if adj[i][j]:
                    neighbor_embed = x[:, :, j].permute((1, 0, 2))
                    temporal_embedding.append(self.temporal_attention(curr_embed, neighbor_embed, neighbor_embed)[0])
            temporal_embedding = torch.cat(tuple(temporal_embedding), dim=0)
            out[i: i + 1, :], att = self.spatial_attention(curr_embed, temporal_embedding, temporal_embedding)
            self.attention_score[i] = att
        out = out.permute((1, 0, 2))
        return out

    def get_attention_score(self, i):
        return self.attention_score[i].squeeze(1).cpu().detach().numpy()


class Attention_Model(nn.Module):
    def __init__(self, embed_dim, n_heads=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.attention = nn.MultiheadAttention(self.embed_dim, self.n_heads)
        self.attention_score = [torch.zeros(1)]

    def _get_key_padding_mask(self, key):
        key_len = key.shape[0]
        batch_size = key.shape[1]
        key_padding_mask = torch.zeros((batch_size, key_len), dtype=torch.uint8).to(self.device)
        for i in range(batch_size):
            for j in range(key_len):
                if torch.equal(key[j, i, :], torch.zeros_like(key[j, i, :])):
                    key_padding_mask[i, j] = True
        return key_padding_mask

    def forward(self, x, adj):
        # [seq_len, batch_size, embedding_dim]
        x = x.permute((1, 0, 2))
        adj = ~adj
        adj = torch.from_numpy(adj).byte().to(self.device)
        target_len = x.shape[0]
        batch_size = x.shape[1]
        out = torch.zeros_like(x)
        self.attention_score = [torch.zeros(1)] * target_len
        # attention_score = torch.zeros((batch_size, target_len, target_len))
        for i in range(target_len):
            mask = adj[i].repeat(batch_size, 1)
            out[i: i + 1], att = self.attention(x[i].unsqueeze(0).clone(),
                                                x, x, key_padding_mask=mask)
            self.attention_score[i] = att
        # [batch_size, seq_len, embedding_dim]
        out = out.permute((1, 0, 2))
        return out

    def get_attention_score(self, i, adj):
        att = self.attention_score[i].cpu().detach().numpy()
        att = att[:, :, adj[i]]
        att = np.squeeze(att, axis=1)
        return att
        # return self.attention_score[i].squeeze(1).cpu().detach().numpy()
    # def forward(self, x, neighbors):
    #     key_padding_mask = self._get_key_padding_mask(neighbors)
    #     if key_padding_mask.all():
    #         return torch.zeros_like(x).to(self.device), None
    #     out, attention_score = self.attention(x, neighbors, neighbors, key_padding_mask=key_padding_mask)
    #     return out, attention_score

# # Single Attention Layer
# class Attention_Layer(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, n_heads=1):
#         super().__init__()
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.output_dim = output_dim
#         self.n_heads = n_heads
#         self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
#         self.W_k, self.W_q, self.W_v = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
#         for _ in range(self.n_heads):
#             self.W_k.append(nn.Linear(input_dim, hidden_dim, bias=False))
#             self.W_q.append(nn.Linear(input_dim, hidden_dim, bias=False))
#             self.W_v.append(nn.Linear(input_dim, hidden_dim, bias=False))
#         self.out_layer = nn.Sequential(nn.Linear(hidden_dim, output_dim), nn.LeakyReLU())
#
#     """
#     input:
#     x: [batch, 1, feature]
#     neighbors: [batch, neighbors_num, feature]
#     output:
#     attention_scores: [batch, 1, neighbors_num]
#     res: [batch, output_feature]
#     """
#     def forward(self, x, neighbors):
#         # calculate Q、K、V
#         all_head_keys = [W_k(neighbors) for W_k in self.W_k]
#         all_head_values = [W_v(neighbors) for W_v in self.W_v]
#         all_head_queries = [W_q(x) for W_q in self.W_q]
#         all_attention_scores = []
#         batch_size = x.shape[0]
#         # calculate attention score
#         res = torch.zeros((batch_size, 1, self.hidden_dim)).to(self.device)
#         for curr_head_key, curr_head_value, curr_head_query in zip(all_head_keys, all_head_values, all_head_queries):
#             attention_logits = torch.matmul(curr_head_query, curr_head_key.permute(0, 2, 1))
#             attention_logits = attention_logits / np.sqrt(self.hidden_dim)
#             attention_scores = F.softmax(attention_logits, dim=2)
#             all_attention_scores.append(attention_scores)
#             res += torch.matmul(attention_scores, curr_head_value)
#         res = res / self.n_heads
#         res = self.out_layer(res)
#         res = res.squeeze(1)
#         # print(torch.std(x))
#         return res
