import traci
import traci.constants as tc
import numpy as np
import logging
from Dueling_DDQN import Dueling_DDQN
from sumolib import checkBinary
from utils import get_configuration
from tqdm import tqdm

# 路网结构
# NEIGHBOR_MAP = {'I0': ['I1', 'I3'],
#                 'I1': ['I0', 'I2', 'I4'],
#                 'I2': ['I1', 'I5'],
#                 'I3': ['I0', 'I4', 'I6'],
#                 'I4': ['I1', 'I3', 'I5', 'I7'],
#                 'I5': ['I2', 'I4', 'I8'],
#                 'I6': ['I3', 'I7'],
#                 'I7': ['I4', 'I6', 'I8'],
#                 'I8': ['I5', 'I7']}
NEIGHBOR_MAP = {'I0': ['I1', 'I2'],
                'I1': ['I0', 'I3'],
                'I2': ['I0', 'I3'],
                'I3': ['I1', 'I2']}
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
    def __init__(self, sumo_config, para_config, gui=False):
        self.sumo_config = sumo_config
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
        self.coef_reward = para_config['coef_reward']
        # 记录路网指标
        self.step_reward = []
        self.step_avg_waiting_time = []
        self.step_avg_speed = []
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
        traci.start(command, port=4343)
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
            # self.node_list[node_name].ilds_in = self.node_list[node_name].lanes_in
            self.node_agent[node_name] = Dueling_DDQN(self.para_config)
            self.curr_action[node_name] = -1
            self.curr_state[node_name] = self._get_local_observation(node_name)
        self.node_name = sorted(list(self.node_list.keys()))

    # 是env里面切换一次信号灯状态
    def step(self):
        for node in self.node_name:
            state = self._get_local_observation(node)
            action = self.node_agent[node].step(state)
            # action = np.random.randint(0, 4)
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
        total_reward = 0
        for node in self.node_list:
            next_state = self._get_local_observation(node)
            reward = self._get_reward(next_state)
            total_reward += reward[-1]
            if self.max_step == self.curr_step:
                is_done = True
            self.node_agent[node].store_experience(self.curr_state[node], self.curr_action[node], reward[-1],
                                                   next_state, is_done)
            self.node_agent[node].learn()
            self.curr_state[node] = next_state
        self.step_reward.append(total_reward)
        self._measure_step()
        return is_done

    def _set_yellow_phase(self, node_name):
        current_phase = traci.trafficlight.getPhase(node_name)
        next_phase = (current_phase + 1) % len(self.phase_map)
        traci.trafficlight.setPhase(node_name, next_phase)

    def _set_green_phase(self, node_name):
        # current_phase = traci.trafficlight.getPhase(node_name)
        next_phase = self.curr_action[node_name] * 2
        # if current_phase % 2 == 1:
        #     next_phase = (current_phase + 1) % len(self.phase_map)
        # else:
        #     next_phase = current_phase
        traci.trafficlight.setPhase(node_name, next_phase)

    def _measure_step(self):
        cars = traci.vehicle.getIDList()
        num_cars = len(cars)
        avg_waiting_time, avg_speed = 0, 0
        if num_cars > 0:
            avg_waiting_time = np.mean([traci.vehicle.getWaitingTime(car) for car in cars])
            avg_speed = np.mean([traci.vehicle.getSpeed(car) for car in cars])
        self.step_avg_waiting_time.append(avg_waiting_time)
        self.step_avg_speed.append(avg_speed)

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
        return np.mean(self.step_reward), np.mean(self.step_avg_waiting_time), np.mean(self.step_avg_speed)

    def reset(self, gui=False):
        traci.close()
        # for node_name in self.nodes_name:
        #     self.nodes[node_name].reset()
        self.curr_episode += 1
        self.curr_step = 0
        episode_reward = np.mean(self.step_reward)
        self.step_reward = []
        self.step_avg_waiting_time = []
        self.step_avg_speed = []
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
        traci.start(command, port=4343)
        for node_name in traci.trafficlight.getIDList():
            # 订阅junction中每个lane的信息
            # self.node_agent[node_name].reset()
            traci.junction.subscribeContext(node_name, tc.CMD_GET_LANE_VARIABLE, 100,
                                            [tc.VAR_WAITING_TIME, tc.LAST_STEP_MEAN_SPEED,
                                             tc.LAST_STEP_VEHICLE_HALTING_NUMBER])
            self.node_agent[node_name].update_epsilon_exploration(self.curr_episode)
            self.node_agent[node_name].update_lr(np.mean(episode_reward))
        # s = 'Env: init %d node information:\n' % len(self.nodes_name)
        # for node_name in self.nodes_name:
        #     s += node_name + ':\n'
        #     s += '\tneigbor: %s\n' % str(self.nodes[node_name].neighbor)
        # logging.info(s)
        # for node_name in self.nodes_name:
        #     traci.junction.subscribeContext(node_name, tc.CMD_GET_VEHICLE_VARIABLE, self.ild_length,
        #                                     [tc.VAR_LANE_ID, tc.VAR_LANEPOSITION,
        #                                     tc.VAR_SPEED, tc.VAR_WAITING_TIME])
        # cx_res = {node_name: traci.junction.getContextSubscriptionResults(node_name) \
        #           for node_name in self.nodes_name}
        # return self._get_obs(cx_res)


if __name__ == '__main__':
    para_config = get_configuration('./data/para_config.ini')
    total_episodes = para_config['total_episodes']
    road_network = TSC_Env('./data/Grid4.sumocfg', para_config)
    reward_total = []
    for i in tqdm(range(total_episodes)):
        while not road_network.step():
            pass
        reward, waiting_time, speed = road_network.get_episode_reward()
        print('episode:', i)
        print('reward:', reward)
        print('waiting_time:', waiting_time)
        print('speed:', speed)
        reward_total.append(reward)
        road_network.reset()
