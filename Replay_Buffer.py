import torch
import random
import numpy as np
from collections import deque


# Replay_Buffer:存储过去的experience
class Replay_Buffer:
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        # self.seed = seed
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def store_experience(self, state, action, reward, next_state, is_done):
        experience = [state, action, reward, next_state, is_done]
        self.memory.append(experience)

    def sample_experience(self, batch_size=None):
        if not batch_size:
            batch_size = self.batch_size
        experiences = random.sample(self.memory, k=batch_size)
        states, actions, rewards, next_states, is_dones = [], [], [], [], []
        for e in experiences:
            states.append(e[0])
            actions.append(e[1])
            rewards.append(e[2])
            next_states.append(e[3])
            is_dones.append(e[4])
        states = np.concatenate(states)
        actions = np.asarray(actions)
        rewards = np.asarray(rewards)
        next_states = np.concatenate(next_states)
        is_dones = np.asarray(is_dones)
        return states, actions, rewards, next_states, is_dones