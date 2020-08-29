import torch
import traci
import traci.constants as tc
import numpy as np
import pandas as pd
import logging
from sumolib import checkBinary
from utils import get_configuration
from tqdm import tqdm
from Agents.Attention_Agents import Attention_Agents
from Agents.Basic_Agents import Basic_Agents
from Agents.Double_Attention_Agents import Double_Attention_Agents
import random
import matplotlib.pyplot as plt
import warnings


# 路网结构
NEIGHBOR_MAP = {'I0': ['I1', 'I3'],
                'I1': ['I0', 'I2', 'I4'],
                'I2': ['I1', 'I5'],
                'I3': ['I0', 'I4', 'I6'],
                'I4': ['I1', 'I3', 'I5', 'I7'],
                'I5': ['I2', 'I4', 'I8'],
                'I6': ['I3', 'I7'],
                'I7': ['I4', 'I6', 'I8'],
                'I8': ['I5', 'I7']}
# NEIGHBOR_MAP = {'I0': ['I1', 'I2'],
#                 'I1': ['I0', 'I3'],
#                 'I2': ['I0', 'I3'],
#                 'I3': ['I1', 'I2']}
# 信号灯状态/从12点开始顺时针旋转
# 1 3 5 7是绿灯之后的黄灯状态
PHASE_MAP = {0: 'GGGrrrrrGGGrrrrr', 1: 'yyyrrrrryyyrrrrr',
             2: 'rrrGrrrrrrrGrrrr', 3: 'rrryrrrrrrryrrrr',
             4: 'rrrrGGGrrrrrGGGr', 5: 'rrrryyyrrrrryyyr',
             6: 'rrrrrrrGrrrrrrrG', 7: 'rrrrrrryrrrrrrry'}


# traffic light
class Traffic_Node:
    def __init__(self, name, neighbor):
        self.name = name
        self.neighbor = neighbor
        self.lanes_in = []
        # self.ilds_in = []


# SUMO环境的设置
class TSC_Env:
    def __init__(self, name, para_config, gui=False):
        self.name = name
        self.sumo_config = './Data/networks/data' + ('/%s/%s.sumocfg' % (self.name, self.name))
        self.para_config = para_config
        self.node_name = []
        self.node_dict = {}
        # store local state and action of intersection
        self.curr_action = {}
        self.obs = {}
        # self.curr_obs = {}
        # self.pre_obs = {}
        # road network structure and parameter
        self.phase_map = PHASE_MAP
        self.neighbor_map = NEIGHBOR_MAP
        self.yellow_duration = para_config['yellow_duration']
        self.green_duration = para_config['green_duration']
        self.control_interval = self.yellow_duration + self.green_duration
        self.coef_reward = para_config['coef_reward']
        self.num_states_phase = 4
        self.num_states_obs = 4
        self.num_states_lanes = 12
        self.num_actions = 4
        self.neighbor_num = 5
        # simulation parameter
        self.curr_step = 0
        self.curr_episode = 0
        self.curr_control_step = 0
        self.num_episode = para_config['total_episodes']
        self.agent_type = para_config['agent_type']
        self.max_step = para_config['max_step']
        self.sim_seed = para_config['sim_seed']
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.seq_len = para_config['seq_len'] + 1 if self.agent_type == 'IQL_Double_Attention' else 2
        # 记录路网指标
        self.split_ratio = 0.75
        self.split = int(self.max_step * self.split_ratio // self.control_interval)
        # step
        self.para = {}
        self.step_reward = []
        self.step_sum_waiting_time = []
        self.step_avg_speed = []
        self.step_sum_queue_length = []
        self.step_data = []
        # episode
        self.metric_data = []
        self.train_episode_reward = []
        self.train_episode_avg_reward = []
        self.test_episode_reward = []
        self.test_episode_avg_reward = []
        # attention_score
        # Grid9 center intersection
        self.attention_score = []
        self.episode_attention_score = []

        # set traci
        if gui:
            app = 'sumo-gui'
        else:
            app = 'sumo'
        command = [checkBinary(app), "--start", '-c', self.sumo_config]
        command += ['--seed', str(self.sim_seed)]
        command += ['--no-step-log', 'True']
        command += ['--time-to-teleport', '300']
        command += ['--no-warnings', 'True']
        command += ['--duration-log.disable', 'True']
        traci.start(command)
        self._init_node()
        # set agent
        input_dim = [self.num_states_phase, self.num_states_obs, self.num_states_lanes]
        hidden_dim = 32
        output_dim = self.num_actions + 1
        if self.agent_type == 'IQL':
            self.agents = Basic_Agents(self.para_config, len(self.node_name), input_dim, hidden_dim, output_dim)
        elif self.agent_type == 'IQL_Attention':
            self.agents = Attention_Agents(self.para_config, len(self.node_name), input_dim, hidden_dim, output_dim,
                                           self.neighbor_map, self.node_name)
        else:
            self.agents = Double_Attention_Agents(self.para_config, len(self.node_name), input_dim, hidden_dim,
                                                  output_dim, self.seq_len,
                                                  self.neighbor_map, self.node_name)

    def _init_node(self):
        for node_name in traci.trafficlight.getIDList():
            # 订阅junction中每个lane的信息
            traci.junction.subscribeContext(node_name, tc.CMD_GET_LANE_VARIABLE, 100,
                                            [tc.VAR_WAITING_TIME, tc.LAST_STEP_MEAN_SPEED,
                                             tc.LAST_STEP_VEHICLE_HALTING_NUMBER])
        for node_name in traci.trafficlight.getIDList():
            if node_name in self.neighbor_map:
                neighbor = self.neighbor_map[node_name]
            else:
                logging.info('node %s can not be found' % node_name)
                neighbor = []
            self.node_dict[node_name] = Traffic_Node(node_name, neighbor)
            self.node_dict[node_name].lanes_in = traci.trafficlight.getControlledLanes(node_name)
            # self.node_agent[node_name] = Dueling_DDQN(self.para_config, self.embedding, self.attention)
            self.curr_action[node_name] = -1
            obs = self._get_local_observation(node_name)
            obs_size = (self.seq_len, obs.shape[0], obs.shape[1])
            self.obs[node_name] = np.zeros(obs_size)
            # for j in range(self.seq_len):
            #     self.obs[node_name].append(np.zeros_like(obs))
            self.obs[node_name][-1] = obs
            # self.curr_obs[node_name] = self._get_local_observation(node_name)
            # self.pre_obs[node_name] = self.curr_obs[node_name]
        self.node_name = sorted(list(self.node_dict.keys()))

    def step_act(self):
        self._simulate(12)
        is_done = False
        reward = []
        for node in self.node_name:
            self.curr_obs[node] = self._get_local_observation(node)
            node_reward = self._get_reward(self.curr_obs[node])
            reward.append(node_reward[-1])
            if self.max_step == self.curr_step:
                is_done = True
        # self.step_reward.append(total_reward)
        self._measure_step(reward)
        return is_done

    # 是env里面切换一次信号灯状态
    def step(self):
        # get observation
        obs = self._get_observation()
        action = self.agents.step(obs)
        for i, node in enumerate(self.node_name):
            # obs = self._get_observation(node)
            # action = self.agents.get_agent(i).step(obs)
            if self.curr_action[node] == action[i]:
                self._set_green_phase(node)
            else:
                self._set_yellow_phase(node)
            self.curr_action[node] = action[i]
            # get attention_score
            # if i == 4:
            #     self.attention_score.append(self.agents.get_attention_score(i)[:, 0])

        # 执行黄灯
        self._simulate(self.yellow_duration)
        for node in self.node_dict:
            self._set_green_phase(node)
        # 执行绿灯
        self._simulate(self.green_duration)
        is_done = False
        reward = []
        # update local observation， get local reward
        for node in self.node_name:
            # self.pre_obs[node] = self.curr_obs[node]
            # self.curr_obs[node] = self._get_local_observation(node)
            self.obs[node][0:-1] = self.obs[node][1:]
            self.obs[node][-1] = self._get_local_observation(node)
            # node_reward = self._get_reward(self.curr_obs[node])
            node_reward = self._get_reward(self.obs[node][-1])
            reward.append(node_reward[-1])
        # get next state
        next_obs = self._get_observation()
        if self.max_step == self.curr_step:
            is_done = True
        if self.curr_step / self.max_step < 0.75:
            self.agents.store_experience(obs, action, reward, next_obs, is_done)
            self.agents.learn()
        # for i, node in enumerate(self.node_name):
        #     obs = self._get_observation(node, True)
        #     next_obs = self._get_observation(node)
        #     if self.max_step == self.curr_step:
        #         is_done = True
        #     # 每个episode的后25%是测试集
        #     if self.curr_step / self.max_step < 0.75:
        #         self.agents.get_agent(i).store_experience(obs, self.curr_action[node], reward[i], next_obs, is_done)
        #         self.agents.get_agent(i).learn()
        self._measure_step(reward)
        return is_done

    # def step_test(self):
    #     for node in self.node_name:
    #         state = self._get_local_observation(node)
    #         # action = np.random.randint(0, 4)
    #         action = self.node_agent[node].step(state)
    #         if self.curr_action[node] == action:
    #             self._set_green_phase(node)
    #         else:
    #             self._set_yellow_phase(node)
    #         self.curr_action[node] = action
    #     # 执行黄灯
    #     self._simulate(self.yellow_duration)
    #     for node in self.node_dict:
    #         self._set_green_phase(node)
    #     # 执行绿灯
    #     self._simulate(self.green_duration)
    #     reward = []
    #     for node in self.node_name:
    #         next_state = self._get_local_observation(node)
    #         node_reward = self._get_reward(next_state)
    #         reward.append(node_reward[-1])
    #         self.curr_obs[node] = next_state
    #     self._measure_step(reward)
    #     return self.max_step == self.curr_step

    def _set_yellow_phase(self, node_name):
        current_phase = traci.trafficlight.getPhase(node_name)
        next_phase = (current_phase + 1) % len(self.phase_map)
        traci.trafficlight.setPhase(node_name, next_phase)

    def _set_green_phase(self, node_name):
        next_phase = self.curr_action[node_name] * 2
        traci.trafficlight.setPhase(node_name, next_phase)

    def _measure_step(self, reward):
        waiting_time, speed, queue_length = [], [], []
        for node in self.node_name:
            # obs, phase = self._get_local_observation(node)
            obs, phase = self.obs[node][-1, :-1], self.obs[node][-1, -1]
            waiting_time.append(np.sum(obs[:, 0]))
            speed.append(np.mean(obs[:, 1]))
            queue_length.append(np.sum(obs[:, 2]))
        # waiting_time = waiting_time / len(self.node_name)
        # speed = speed / len(self.node_name)
        # queue_length = queue_length / len(self.node_name)
        self.step_sum_waiting_time.append(waiting_time)
        self.step_avg_speed.append(speed)
        self.step_sum_queue_length.append(queue_length)
        self.step_reward.append(reward)
        info = {'episode': self.curr_episode,
                'time': self.curr_step,
                'step': self.curr_step / self.control_interval,
                'action': [self.curr_action[node_name] for node_name in self.node_name],
                'waiting_time': waiting_time,
                'speed': speed,
                'queue_length': queue_length,
                'reward': reward,
                'total_reward': np.sum(reward)}
        self.step_data.append(info)

    # get local observation of intersection
    def _get_local_observation(self, node_name):
        ob_res = traci.junction.getContextSubscriptionResults(node_name)
        obs = []
        for lane, res in ob_res.items():
            f_node, t_node, _ = lane.split('_')
            if t_node == node_name and (f_node[0] == 'P' or f_node[0] == 'I'):
                waiting_time, mean_speed, queue_length = \
                    res[tc.VAR_WAITING_TIME]/500, res[tc.LAST_STEP_MEAN_SPEED]/20, \
                    res[tc.LAST_STEP_VEHICLE_HALTING_NUMBER]/40
                vehicle_num = traci.lanearea.getLastStepVehicleNumber(lane)/40
                obs_lane = [waiting_time, mean_speed, queue_length, vehicle_num]
                obs.append(obs_lane)
        obs = np.asarray(obs)
        phase = np.zeros(shape=(1, int(len(self.phase_map) / 2)))
        current_phase = int(traci.trafficlight.getPhase(node_name) / 2)
        phase[0, current_phase] = 1
        obs = np.concatenate((obs, phase))
        return obs

    def _get_observation(self, pre=False):
        obs = []
        for i, node in enumerate(self.node_name):
            obs.append(self.obs[node][1-pre: self.seq_len-pre])
        obs = np.stack(obs, axis=1)
        return obs
    # get observation based on agent-type
    # def _get_observation(self, node_name, pre=False):
    #     # state = self.curr_obs
    #     #     state = self.pre_obs
    #     pre = int(pre)
    #     if self.agent_type == 'IQL':
    #         obs = np.asarray(self.obs[node_name][1-pre: self.seq_len-pre])
    #     else:
    #         # obs = np.zeros((5, state[node_name].shape[0], state[node_name].shape[1]))
    #         # obs = [np.asarray(state[node_name])]
    #         obs = [np.asarray(self.obs[node_name][1-pre: self.seq_len-pre])]
    #         for neighbor in self.node_dict[node_name].neighbor:
    #             # obs.append(state[neighbor])
    #             obs.append(np.asarray(self.obs[neighbor][1-pre: self.seq_len-pre]))
    #         obs = np.stack(obs, axis=-1)
    #         if obs.shape[-1] < self.neighbor_num:
    #             tmp = np.zeros((obs.shape[0], obs.shape[1], obs.shape[2], self.neighbor_num - obs.shape[3]))
    #             obs = np.concatenate((obs, tmp), axis=-1)
    #         # if self.agent_type == 'IQL_LSTM_Attention':
    #         #     obs_lstm = [self.agents.get_agent(i).get_hidden_state()]
    #         #     for neighbor in self.node_dict[node_name].neighbor:
    #         #         idx = self.node_name.index(neighbor)
    #         #         obs_lstm.append(self.agents.get_agent(idx).get_hidden_state())
    #         #     obs_lstm = np.stack(obs_lstm, axis=-1)
    #         #     obs_all = [obs, obs_lstm]
    #         #     return obs_all
    #     if self.agent_type == 'IQL_Double_Attention':
    #         obs = np.expand_dims(obs, axis=0)
    #     return obs

    def _get_reward(self, obs):
        obs_node = obs[:-1]
        jam_length = np.sum(obs_node[:, 2])
        waiting_time = np.sum(obs_node[:, 0])
        reward_jam = -jam_length
        reward_waiting = -waiting_time
        reward = reward_jam + reward_waiting
        return reward_jam, reward_waiting, reward

    def _simulate(self, num_step):
        if self.curr_step + num_step >= self.max_step:
            num_step = self.max_step - self.curr_step
        for _ in range(num_step):
            traci.simulationStep()
            self.curr_step += 1

    def get_episode_reward(self):
        self.step_reward = np.asarray(self.step_reward)
        self.step_sum_waiting_time = np.asarray(self.step_sum_waiting_time)
        self.step_avg_speed = np.asarray(self.step_avg_speed)
        self.step_sum_queue_length = np.asarray(self.step_sum_queue_length)
        train_episode_reward = np.mean(self.step_reward[0:self.split, :], axis=0)
        test_episode_reward = np.mean(self.step_reward[self.split:, :], axis=0)

        train_avg_reward = np.sum(train_episode_reward)*50
        test_avg_reward = np.sum(test_episode_reward)*50
        self.train_episode_reward.append(train_episode_reward)
        self.test_episode_reward.append(test_episode_reward)
        self.train_episode_avg_reward.append(train_avg_reward)
        self.test_episode_avg_reward.append(test_avg_reward)
        avg_waiting_time = np.mean(np.sum(self.step_sum_waiting_time, axis=1))*500
        avg_speed = np.mean(self.step_avg_speed)*20
        avg_queue_length = np.mean(np.sum(self.step_sum_queue_length, axis=1))*40
        episode_info = {'episode': self.curr_episode,
                        'train_episode_reward': train_episode_reward,
                        'test_episode_reward': test_episode_reward,
                        'train_avg_reward': train_avg_reward,
                        'test_avg_reward': test_avg_reward,
                        'avg_waiting_time': avg_waiting_time,
                        'avg_speed': avg_speed,
                        'avg_queue_length': avg_queue_length}
        self.metric_data.append(episode_info)
        return train_avg_reward, test_avg_reward, avg_waiting_time, avg_speed, avg_queue_length

    def output_data(self):
        # episode_reward = np.asarray(self.train_episode_reward)
        train_avg_reward = np.asarray(self.train_episode_avg_reward)
        test_avg_reward = np.asarray(self.test_episode_avg_reward)
        colors = ['b', 'g', 'r', 'k', 'c', 'm', 'y', '#e24fff', '#524C90', '#845868']
        plt.figure()
        plt.xlabel('episode')
        plt.ylabel('reward')
        x = np.arange(0, self.num_episode)
        # for i in range(len(self.node_name)):
        #     plt.plot(x, episode_reward[:,i], color=colors[i], label=self.node_name[i])

        plt.plot(x, train_avg_reward, color=colors[-1], label='train')
        plt.plot(x, test_avg_reward, color=colors[0], label='test')
        plt.legend()
        fig_name = './Data/logs/reward_' + self.name + self.agent_type + '.png'
        plt.savefig(fig_name)
        plt.show()

        # plt.figure()
        # plt.xlabel('episode')
        # plt.ylabel('attention score')
        # labels = ['I4', 'I1', 'I3', 'I5', 'I7']
        # all_attention_score = np.stack(self.episode_attention_score, axis=0)
        # for i in range(all_attention_score.shape[1]):
        #     plt.plot(x, all_attention_score[:, i], label=labels[i])
        # plt.legend()
        # fig_name = './Data/logs/attention_' + self.name + '.png'
        # plt.savefig(fig_name)
        # plt.show()
        # plt.figure()
        # plt.xlabel('episode')
        # plt.ylabel('mean')
        # for k in self.para:
        #     data = np.asarray(self.para[k])
        #     plt.plot(x, data[:, 0], label=k)
        # plt.legend()
        # plt.show()
        # plt.figure()
        # plt.xlabel('episode')
        # plt.ylabel('std')
        # for k in self.para:
        #     data = np.asarray(self.para[k])
        #     plt.plot(x, data[:, 1], label=k)
        # plt.legend()
        # plt.show()

        step_data = pd.DataFrame(self.step_data)
        step_data.to_csv('./Data/logs/' + ('%s_%s_step.csv' % (self.name,self.agent_type)))
        metric_data = pd.DataFrame(self.metric_data)
        metric_data.to_csv('./Data/logs/' + ('%s_%s_metric.csv' % (self.name, self.agent_type)))

    def close(self):
        traci.close()

    def _get_para_data(self):
        para = dict(self.agents.q_network.named_parameters())
        for k in para:
            data = para[k].data
            avg, std = (torch.mean(data)).numpy(), (torch.std(data)).numpy()
            tmp = self.para.get(k, [])
            tmp.append(np.asarray((avg, std)))
            self.para[k] = tmp


    def reset(self, gui=False):
        self.close()
        # self._get_para_data()
        self.curr_episode += 1
        # if self.curr_episode > 50 and self.agent_type != 'IQL':
        #     self.agents.change_mode()
        self.curr_step = 0
        self.step_reward = []
        self.step_sum_waiting_time = []
        self.step_sum_queue_length = []
        self.step_avg_speed = []
        # attention_score = np.mean(np.concatenate(self.attention_score), axis=0)
        # self.episode_attention_score.append(attention_score)
        #
        # if self.curr_episode == self.num_episode:
        #     step_attention_score = np.concatenate(self.attention_score, axis=0)
        #     plt.figure()
        #     plt.xlabel('step')
        #     plt.ylabel('attention score')
        #     labels = ['I4', 'I1', 'I3', 'I5', 'I7']
        #     for i in range(step_attention_score.shape[1]):
        #         plt.plot(step_attention_score[:,i], label=labels[i])
        #     plt.legend()
        #     fig_name = './Data/logs/step_attention_' + self.name + '.png'
        #     plt.savefig(fig_name)
        #     plt.show()
        # self.attention_score = []
        if gui:
            app = 'sumo-gui'
        else:
            app = 'sumo'
        command = [checkBinary(app), "--start", '-c', self.sumo_config]
        command += ['--seed', str(self.sim_seed)]
        command += ['--no-step-log', 'True']
        command += ['--time-to-teleport', '300']
        command += ['--no-warnings', 'True']
        command += ['--duration-log.disable', 'True']
        traci.start(command)
        for i, node_name in enumerate(traci.trafficlight.getIDList()):
            # 订阅junction中每个lane的信息
            traci.junction.subscribeContext(node_name, tc.CMD_GET_LANE_VARIABLE, 100,
                                            [tc.VAR_WAITING_TIME, tc.LAST_STEP_MEAN_SPEED,
                                             tc.LAST_STEP_VEHICLE_HALTING_NUMBER])
            # if self.curr_episode > 50 and self.agent_type != 'IQL':
            #     self.agents.get_agent(i).update_epsilon_exploration(self.curr_episode-50)
            # else:
            #     self.agents.get_agent(i).update_epsilon_exploration(self.curr_episode)
            self.agents.get_agent(i).update_epsilon_exploration(self.curr_episode)

    # def set_agent(self):
    #     for node in self.node_name:
    #         model_name = './models/model_' + node + '.pkl'
    #         self.node_agent[node].load_q_network(model_name)


# 设置随机数种子
def set_random_seeds(random_seed):
    """Sets all possible random seeds so results can be reproduced"""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.cuda.manual_seed(random_seed)


if __name__ == '__main__':
    para_config = get_configuration('para_config.ini')
    total_episodes = para_config['total_episodes']
    warnings.filterwarnings('ignore')
    # total_episodes = 3
    sim_seed = para_config['sim_seed']
    set_random_seeds(sim_seed)
    road_network = TSC_Env('Grid9', para_config)
    # reward_total = []
    for i in tqdm(range(total_episodes)):
        while not road_network.step():
            pass
        train_reward, test_reward, waiting_time, speed, queue_length = road_network.get_episode_reward()
        print('episode:', i)
        print('train_reward:', train_reward)
        print('test_reward', test_reward)
        print('waiting_time:', waiting_time)
        print('speed:', speed)
        print('queue_length', queue_length)
        # reward_total.append(reward)
        road_network.reset()
    # road_network.output_data()
    road_network.close()
    #
    # road_network_test = TSC_Env('Grid9', para_config, gui=True)
    # road_network_test.set_agent()
    # while not road_network.step_test():
    #     pass
    # reward, waiting_time, speed, queue_length = road_network.get_episode_reward()
    # print('reward:', reward)
    # print('waiting_time:', waiting_time)
    # print('speed:', speed)
    # print('queue_length', queue_length)