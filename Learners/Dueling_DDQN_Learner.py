import torch
from torch import optim
import torch.nn.functional as F
from Learners.Base_Learner import Base_Learner


class Dueling_DDQN_Learner(Base_Learner):
    def __init__(self, config):
        Base_Learner.__init__(self, config)
        # 创建Q和Q’
        # self.q_network_current = self.create_model(embedding, attention, 20, self.num_actions+1)
        # self.q_network_target = self.create_model(embedding, attention, 20, self.num_actions+1)
        self.q_network_current = self.create_model([self.num_states_phase, self.num_states_obs, self.num_states_lanes], self.num_actions+1)
        self.q_network_target = self.create_model([self.num_states_phase, self.num_states_obs, self.num_states_lanes], self.num_actions+1)
        # 复制网络
        self.optimizer = optim.Adam(self.q_network_current.parameters(), lr=1e-3)
        self.copy_network(self.q_network_current, self.q_network_target)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', patience=3, threshold=1e-6, factor=0.2)

    # 执行一个step，返回action
    def step(self, state):
        self.q_network_current.eval()
        with torch.no_grad():
            out = self.q_network_current(state)
            advantages_values, state_values = out[:, :-1], out[:, -1]
        self.q_network_current.train()
        action = self.select_actions(advantages_values)
        self.curr_step += 1
        return action

    # 通过experience replay 进行学习
    def learn(self):
        # 得到的是一个mini_batch的experience
        # 每隔C步进行学习
        if self.curr_step > 0 and self.curr_step % self.update_step == 0:
            states, actions, rewards, next_states, is_dones = self.sample_experience()
            actions = torch.from_numpy(actions).long().to(self.device)
            rewards = torch.from_numpy(rewards).float().to(self.device)
            is_dones = torch.from_numpy(is_dones).float().to(self.device)
            actions_values_current = self.cal_current_actions_value(next_states, rewards, is_dones)
            actions_values_expected = self.cal_expected_actions_value(states, actions)
            loss = F.mse_loss(actions_values_expected, actions_values_current)
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.q_network_current.parameters(), 1)
            self.optimizer.step()
            # 更新target net
            self.soft_update_of_target_network(self.q_network_current, self.q_network_target)

    def cal_current_actions_value(self, next_states, rewards, is_dones):
        # self.q_network_current.eval()
        # with torch.no_grad():
        out_next = self.q_network_current(next_states).detach()
        advantages_values_next, state_values_next = out_next[:, :-1], out_next[:, -1]
        # 利用Q选出行动价值最大的行动
        selected_actions_next = advantages_values_next.argmax(1)
        # 利用Q'计算行动价值
        # self.q_network_target.eval()
        # with torch.no_grad():
        out_target = self.q_network_target(next_states)
        advantages_values_target, state_values_target = out_target[:, :-1], out_target[:, -1]
        avg_advantages_values_target = torch.mean(advantages_values_target, dim=1)
        actions_values_target = torch.unsqueeze(state_values_target, 1) + advantages_values_target - \
                                torch.unsqueeze(avg_advantages_values_target, dim=1)
        action_values_next = torch.gather(actions_values_target, 1, torch.unsqueeze(selected_actions_next,1))
        action_values_current = rewards.unsqueeze(1) + self.gamma*action_values_next*(1-is_dones.unsqueeze(1))
        return action_values_current

    def cal_expected_actions_value(self, states, actions):
        # with torch.no_grad():
        out = self.q_network_current(states)
        advantages_values, state_values = out[:, :-1], out[:, -1]
        avg_advantages_values = torch.mean(advantages_values, dim=1)
        actions_values = torch.unsqueeze(state_values, 1) + advantages_values - \
                                torch.unsqueeze(avg_advantages_values, dim=1)
        action_values_expected = torch.gather(actions_values, 1, torch.unsqueeze(actions, 1))
        return action_values_expected

    # 当奖励函数不再上升时降低学习率
    def update_lr(self, metric):
        self.scheduler.step(metric)

    def load_q_network(self, model_name):
        self.q_network_current = torch.load(model_name)
        self.copy_network(self.q_network_current, self.q_network_target)
        self.epsilon = 0.01

    def set_q_network(self, q_network_current, q_network_target):
        self.q_network_current = q_network_current
        self.q_network_target = q_network_target
        # 复制网络
        self.optimizer = optim.RMSprop(self.q_network_current.parameters(), lr=1e-3)
        self.copy_network(self.q_network_current, self.q_network_target)

    def get_q_network(self):
        return self.q_network_current
