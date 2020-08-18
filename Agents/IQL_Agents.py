from Learners.Dueling_DDQN_Learner import Dueling_DDQN_Learner


# Agents的集合
class IQL_Agents:
    def __init__(self, config, num_agents):
        self.num_agents = num_agents
        self.config = config
        self.agents = []
        for i in range(self.num_agents):
            self.agents.append(Dueling_DDQN_Learner(config))

    def get_agent(self, i):
        return self.agents[i]