from Agents.IQL_Agents import IQL_Agents
from Models.Attention_Model import Model
from Models.Basic_Model import Embedding_Layer, Attention_Layer, Multi_Attention_Layer
from Learners.Dueling_DDQN_Learner import Dueling_DDQN_Learner
import torch

# 所有交叉口的Agent的集合
class IQL_Attention_Agents(IQL_Agents):
    def __init__(self, config, num_agents, input_dim, hidden_dim, output_dim):
        super().__init__(config, num_agents)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.embedding = Embedding_Layer(self.input_dim, self.hidden_dim)
        self.attention = Multi_Attention_Layer(self.hidden_dim)
        self.embedding_target = Embedding_Layer(self.input_dim, self.hidden_dim)
        self.attention_target = Multi_Attention_Layer(self.hidden_dim)
        Dueling_DDQN_Learner.copy_network(self.embedding, self.embedding_target)
        Dueling_DDQN_Learner.copy_network(self.attention, self.attention_target)
        # self.q_network = Model(self.input_dim, self.output_dim, self.hidden_dim).to(self.device)
        # self.q_network_target = Model(self.input_dim, self.output_dim, self.hidden_dim).to(self.device)
        for i in range(self.num_agents):
            q_network = Model(self.hidden_dim, self.output_dim,
                              embedding_layer=self.embedding, attention_layer=self.attention).to(self.device)
            q_network_target = Model(self.hidden_dim, self.output_dim, embedding_layer=self.embedding_target,
                                     attention_layer=self.attention_target).to(self.device)
            self.agents[i].set_q_network(q_network, q_network_target)

    def change_mode(self):
        # self.q_network.change_mode()
        # self.q_network_target.change_mode()
        for i in range(self.num_agents):
            self.agents[i].q_network_current.change_mode()
            self.agents[i].q_network_target.change_mode()
        # self.embedding = Embedding_Layer(self.input_dim, self.hidden_dim[0])
        # self.attention = Attention_Layer(self.hidden_dim[0], self.hidden_dim[1], self.hidden_dim[2])
        # self.embedding_target = Embedding_Layer(self.input_dim, self.hidden_dim[0])
        # self.attention_target = Attention_Layer(self.hidden_dim[0], self.hidden_dim[1], self.hidden_dim[2])
        # Dueling_DDQN_Learner.copy_network(self.embedding, self.embedding_target)
        # Dueling_DDQN_Learner.copy_network(self.attention, self.attention_target)
    # def get_action(self, i, obs):
    #     return self.agents[i].step(obs)
    # def store_experience(self, i, obs, action, reward, next_obs, is_done):
    #     self.agents[i].store_experience(obs, action, reward, next_obs, is_done)