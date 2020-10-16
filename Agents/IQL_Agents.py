import torch
from itertools import chain
from torch import optim
import torch.nn.functional as F
import numpy as np
from Agents.Basic_Agents import Basic_Agents
from Learners.Dueling_DDQN_Learner import Dueling_DDQN_Learner
from Models.Basic_Model import Base_Model

# IQL_Agents with all parameters sharing
class IQL_Agents(Basic_Agents):
    def __init__(self, config, num_agents, input_dim, hidden_dim, output_dim):
        super().__init__(config, num_agents, input_dim, hidden_dim, output_dim)
        self._init_agents()

    def _init_agents(self):
        # parameter sharing
        self.embedding = Base_Model(self.input_dim, self.hidden_dim, self.output_dim).to(self.device)
        self.embedding_target = Base_Model(self.input_dim, self.hidden_dim, self.output_dim).to(self.device)
        # Dueling_DDQN_Learner.copy_network(self.embedding, self.embedding_target)
        # init the optimizer
        self.learner = Dueling_DDQN_Learner(self.config)
        # for i in range(self.num_agents):
        #     self.agents.append(Dueling_DDQN_Learner(self.config))
        self.learner.set_q_network(self.embedding, self.embedding_target)
        self.all_para = self.embedding.parameters()
        self.share_optimizer = optim.RMSprop(self.all_para, lr=self.lr, weight_decay=1e-4)

    def learn(self):
        # if self.curr_step > 0 and self.curr_step % self.update_step == 0:
        for i in range(self.update_step):
            states, actions, rewards, next_states, is_dones = self.sample_experience()
            actions = torch.from_numpy(actions).long().to(self.device)
            rewards = torch.from_numpy(rewards).float().to(self.device)
            is_dones = torch.from_numpy(is_dones).float().to(self.device)
            # states_embedding = self._get_embedding(states)
            # next_states_embedding = self._get_embedding(next_states)
            # next_states_embedding_target = self._get_embedding_target(next_states)
            total_loss = 0
            for i in range(self.num_agents):
                actions_values_current = self.learner.cal_current_actions_value(states[:, i], next_states[:, i],
                                                                                  rewards[:, i], is_dones)
                actions_values_expected = self.learner.cal_expected_actions_value(states[:, i], actions[:, i])
                loss = F.mse_loss(actions_values_expected, actions_values_current)
                # loss.backward(retain_graph=True)
                total_loss += loss
                # 反向传播
                # self.optimizer[i].zero_grad()
            total_loss /= self.num_agents
            self.share_optimizer.zero_grad()
            total_loss.backward()
            # self._scale_shared_grads()
            torch.nn.utils.clip_grad_value_(self.all_para, 1)
            self.share_optimizer.step()
            Dueling_DDQN_Learner.soft_update_of_target_network(self.embedding, self.embedding_target, self.tau)

    def step(self, state, test=False):
        action = []
        for i in range(self.num_agents):
            action.append(self.learner.step(state[:, i], test))
        action = np.asarray(action)
        self.curr_step += 1
        return action
