import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.phase_dim, self.state_dim, self.lane_num = input_dim[0], input_dim[1], input_dim[2]
        self.output_dim = output_dim
        self.linear_phase = nn.Linear(self.phase_dim, 20)
        self.linear_state = nn.Linear(self.state_dim, 20)
        self.linear_final = nn.Linear(20 * (self.lane_num + 1), self.output_dim)
        self.relu = nn.ReLU()
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
        x = self.relu(x)
        out = self.linear_final(x)
        return out