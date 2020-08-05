import torch
import traci
import traci.constants as tc
import numpy as np
import pandas as pd
import logging
from Dueling_DDQN import Dueling_DDQN
from sumolib import checkBinary
from utils import get_configuration
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

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
        self.sumo_config = './networks/data' + ('/%s/%s.sumocfg' % (self.name, self.name))
        self.para_config = para_config
        self.node_name = []
        self.node_list = {}
        self.node_agent = {}
        self.curr_action = {}
        self.curr_state = {}
        self.phase_map = PHASE_MAP
        self.neighbor_map = NEIGHBOR_MAP
        self.curr_step = 0
        self.curr_episode = 0
        self.max_step = para_config['max_step']
        self.sim_seed = para_config['sim_seed']
        self.yellow_duration = para_config['yellow_duration']
        self.green_duration = para_config['green_duration']
        self.control_interval = self.yellow_duration + self.green_duration
        self.coef_reward = para_config['coef_reward']
        self.num_episode = para_config['total_episodes']
        # self.num_episode = 3
        # 记录路网指标
        self.step_reward = []
        self.step_sum_waiting_time = []
        self.step_avg_speed = []
        self.step_sum_queue_length = []
        self.metric_data = []
        self.step_data = []
        self.episode_reward = []
        self.episode_total_reward = []
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
            self.node_list[node_name] = Traffic_Node(node_name, neighbor)
            self.node_list[node_name].lanes_in = traci.trafficlight.getControlledLanes(node_name)
            self.node_agent[node_name] = Dueling_DDQN(self.para_config)
            self.curr_action[node_name] = -1
            self.curr_state[node_name] = self._get_local_observation(node_name)
        self.node_name = sorted(list(self.node_list.keys()))

    def step_act(self):
        self._simulate(12)
        is_done = False
        reward = []
        for node in self.node_name:
            state = self._get_local_observation(node)
            node_reward = self._get_reward(state)
            reward.append(node_reward[-1])
            if self.max_step == self.curr_step:
                is_done = True
        # self.step_reward.append(total_reward)
        self._measure_step(reward)
        return is_done

    # 是env里面切换一次信号灯状态
    def step(self):
        for node in self.node_name:
            state = self._get_local_observation(node)
            # action = np.random.randint(0, 4)
            action = self.node_agent[node].step(state)
            if self.curr_action[node] == action:
                self._set_green_phase(node)
            else:
                self._set_yellow_phase(node)
            self.curr_action[node] = action
        # 执行黄灯
        self._simulate(self.yellow_duration)
        for node in self.node_list:
            self._set_green_phase(node)
        # 执行绿灯
        self._simulate(self.green_duration)
        is_done = False
        reward = []
        for node in self.node_name:
            next_state = self._get_local_observation(node)
            node_reward = self._get_reward(next_state)
            reward.append(node_reward[-1])
            if self.max_step == self.curr_step:
                is_done = True
            self.node_agent[node].store_experience(self.curr_state[node], self.curr_action[node], node_reward[-1],
                                                   next_state, is_done)
            self.node_agent[node].learn()
            self.curr_state[node] = next_state
        self._measure_step(reward)
        return is_done

    def step_test(self):
        for node in self.node_name:
            state = self._get_local_observation(node)
            # action = np.random.randint(0, 4)
            action = self.node_agent[node].step(state)
            if self.curr_action[node] == action:
                self._set_green_phase(node)
            else:
                self._set_yellow_phase(node)
            self.curr_action[node] = action
        # 执行黄灯
        self._simulate(self.yellow_duration)
        for node in self.node_list:
            self._set_green_phase(node)
        # 执行绿灯
        self._simulate(self.green_duration)
        reward = []
        for node in self.node_name:
            next_state = self._get_local_observation(node)
            node_reward = self._get_reward(next_state)
            reward.append(node_reward[-1])
            self.curr_state[node] = next_state
        self._measure_step(reward)
        return self.max_step == self.curr_step

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
            obs, phase = self._get_local_observation(node)
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

    def _get_local_observation(self, node_name):
        ob_res = traci.junction.getContextSubscriptionResults(node_name)
        obs = []
        for lane, res in ob_res.items():
            f_node, t_node, _ = lane.split('_')
            if t_node == node_name and (f_node[0] == 'P' or f_node[0] == 'I'):
                waiting_time, mean_speed, queue_length = \
                    res[tc.VAR_WAITING_TIME], res[tc.LAST_STEP_MEAN_SPEED], \
                    res[tc.LAST_STEP_VEHICLE_HALTING_NUMBER]
                vehicle_num = traci.lanearea.getLastStepVehicleNumber(lane)
                obs_lane = [waiting_time, mean_speed, queue_length, vehicle_num]
                obs.append(obs_lane)
        obs = np.asarray(obs)
        phase = np.zeros(shape=(int(len(self.phase_map) / 2)))
        current_phase = int(traci.trafficlight.getPhase(node_name) / 2)
        phase[current_phase] = 1
        return [obs, phase]

    def _get_neighbor_observation(self, node_name):
        return

    def _get_reward(self, obs):
        obs_node = obs[0]
        jam_length = np.sum(obs_node[:, 2])
        waiting_time = np.sum(obs_node[:, 0])
        reward_jam = -jam_length
        reward_waiting = -waiting_time
        reward = reward_jam + self.coef_reward * reward_waiting
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
        avg_reward = np.mean(self.step_reward, axis=0)
        total_reward = np.sum(avg_reward)
        self.episode_reward.append(avg_reward)
        self.episode_total_reward.append(total_reward)
        avg_waiting_time = np.mean(np.sum(self.step_sum_waiting_time, axis=1))
        avg_speed = np.mean(self.step_avg_speed)
        avg_queue_length = np.mean(np.sum(self.step_sum_queue_length, axis=1))
        episode_info = {'episode': self.curr_episode,
                        'avg_reward': avg_reward,
                        'total_reward': total_reward,
                        'avg_waiting_time': avg_waiting_time,
                        'avg_speed': avg_speed,
                        'avg_queue_length': avg_queue_length}
        self.metric_data.append(episode_info)
        return total_reward, avg_waiting_time, avg_speed, avg_queue_length

    def output_data(self):
        episode_reward = np.asarray(self.episode_reward)
        episode_total_reward = np.asarray(self.episode_total_reward)
        colors = ['b', 'g', 'r', 'k', 'c', 'm', 'y', '#e24fff', '#524C90', '#845868']
        plt.figure()
        plt.xlabel('episode')
        plt.ylabel('reward')
        x = np.arange(0, self.num_episode+1)
        for i in range(len(self.node_name)):
            plt.plot(x, episode_reward[:,i], color=colors[i], label=self.node_name[i])
        plt.plot(x, episode_total_reward, color=colors[-1], label='total')
        plt.legend()
        plt.show()
        step_data = pd.DataFrame(self.step_data)
        step_data.to_csv('./logs/' + ('%s_step.csv' % self.name))
        metric_data = pd.DataFrame(self.metric_data)
        metric_data.to_csv('./logs/' + ('%s_metric.csv' % self.name))

    def close(self):
        traci.close()

    def reset(self, gui=False):
        traci.close()
        # for node_name in self.nodes_name:
        #     self.nodes[node_name].reset()
        self.curr_episode += 1
        if self.curr_episode == self.num_episode:
            gui = True
            for node in self.node_name:
                model_name = './models/model_' + node + '.pkl'
                torch.save(self.node_agent[node].get_q_network(), model_name)
        self.curr_step = 0
        episode_reward = np.mean(self.step_reward)
        self.step_reward = []
        self.step_sum_waiting_time = []
        self.step_sum_queue_length = []
        self.step_avg_speed = []
        if gui:
            app = 'sumo-gui'
        else:
            app = 'sumo'
        command = [checkBinary(app), "--start", '-c', self.sumo_config]
        # if self.curr_episode == 3:
        #     command += ['--save-state.period <3600>']
        command += ['--seed', str(self.sim_seed)]
        command += ['--no-step-log', 'True']
        command += ['--time-to-teleport', '300']
        command += ['--no-warnings', 'True']
        command += ['--duration-log.disable', 'True']
        traci.start(command, port=4343)
        for node_name in traci.trafficlight.getIDList():
            # 订阅junction中每个lane的信息
            # self.node_agent[node_name].reset()
            traci.junction.subscribeContext(node_name, tc.CMD_GET_LANE_VARIABLE, 100,
                                            [tc.VAR_WAITING_TIME, tc.LAST_STEP_MEAN_SPEED,
                                             tc.LAST_STEP_VEHICLE_HALTING_NUMBER])
            self.node_agent[node_name].update_epsilon_exploration(self.curr_episode)
            self.node_agent[node_name].update_lr(np.mean(episode_reward))

    def set_agent(self):
        for node in self.node_name:
            model_name = './models/model_' + node + '.pkl'
            self.node_agent[node].set_q_network(model_name)

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
    # total_episodes = 3
    sim_seed = para_config['sim_seed']
    set_random_seeds(sim_seed)
    road_network = TSC_Env('Grid9', para_config, gui=True)
    reward_total = []

    road_network.set_agent()
    while not road_network.step_test():
        pass
    reward, waiting_time, speed, queue_length = road_network.get_episode_reward()
    # print('episode:', i)
    print('reward:', reward)
    print('waiting_time:', waiting_time)
    print('speed:', speed)
    print('queue_length', queue_length)


    for i in tqdm(range(total_episodes+1)):
        while not road_network.step():
            pass
        reward, waiting_time, speed, queue_length = road_network.get_episode_reward()
        print('episode:', i)
        print('reward:', reward)
        print('waiting_time:', waiting_time)
        print('speed:', speed)
        print('queue_length', queue_length)
        reward_total.append(reward)
        road_network.reset()
    road_network.output_data()
    road_network.close()
