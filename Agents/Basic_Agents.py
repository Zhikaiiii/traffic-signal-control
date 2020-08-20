from Learners.Dueling_DDQN_Learner import Dueling_DDQN_Learner
# from Replay_Buffer import Replay_Buffer


# Agents的集合
class Basic_Agents:
    def __init__(self, config, num_agents):
        self.num_agents = num_agents
        self.config = config
        # Replay Buffer相关参数
        # self.batch_size = config['batch_size']
        # self.buffer_size = config['buffer_size']
        self.agents = []
        # self.buffer = []
        for i in range(self.num_agents):
            self.agents.append(Dueling_DDQN_Learner(config))
            # self.buffer.append(Replay_Buffer(self.buffer_size, self.batch_size))

    def get_agent(self, i):
        return self.agents[i]