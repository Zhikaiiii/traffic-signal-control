from Agents.Basic_Agents import Basic_Agents
from Models.Double_Attention_Model import Double_Attention_Model
from Models.Attention_Model import Attention_Model
from Models.Basic_Model import Embedding_Layer, Multi_Attention_Layer
from Learners.Dueling_DDQN_Learner import Dueling_DDQN_Learner
import torch


# 所有交叉口的Agent的集合
class Attention_Agents(Basic_Agents):
    def __init__(self, config, num_agents, input_dim, hidden_dim, output_dim):
        super().__init__(config, num_agents)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # self.q_network = Model(self.input_dim, self.output_dim, self.hidden_dim).to(self.device)
        # self.q_network_target = Model(self.input_dim, self.output_dim, self.hidden_dim).to(self.device)
        self._init_agents()

    def _init_agents(self):
        # parameter sharing
        self.embedding = Embedding_Layer(self.input_dim, self.hidden_dim).to(self.device)
        self.attention = Multi_Attention_Layer(self.hidden_dim).to(self.device)
        self.embedding_target = Embedding_Layer(self.input_dim, self.hidden_dim).to(self.device)
        self.attention_target = Multi_Attention_Layer(self.hidden_dim).to(self.device)
        Dueling_DDQN_Learner.copy_network(self.embedding, self.embedding_target)
        Dueling_DDQN_Learner.copy_network(self.attention, self.attention_target)
        for i in range(self.num_agents):
            q_network = Attention_Model(self.input_dim, self.output_dim, self.hidden_dim).to(self.device)
            q_network_target = Attention_Model(self.input_dim, self.output_dim, self.hidden_dim).to(self.device)
            q_network.set_layer_para(self.embedding, self.attention)
            q_network_target.set_layer_para(self.embedding_target, self.attention_target)
            self.agents[i].set_q_network(q_network, q_network_target)


class Double_Attention_Agents(Attention_Agents):
    def __init__(self, config, num_agents, input_dim, hidden_dim, output_dim):
        super().__init__(config, num_agents, input_dim, hidden_dim, output_dim)
        self._init_agents()

    def _init_agents(self):
        self.embedding = Embedding_Layer(self.input_dim, self.hidden_dim).to(self.device)
        self.attention = Multi_Attention_Layer(self.hidden_dim).to(self.device)
        self.temporal_attention = Multi_Attention_Layer(self.hidden_dim).to(self.device)
        self.embedding_target = Embedding_Layer(self.input_dim, self.hidden_dim).to(self.device)
        self.attention_target = Multi_Attention_Layer(self.hidden_dim).to(self.device)
        self.temporal_attention_target = Multi_Attention_Layer(self.hidden_dim).to(self.device)
        Dueling_DDQN_Learner.copy_network(self.embedding, self.embedding_target)
        Dueling_DDQN_Learner.copy_network(self.attention, self.attention_target)
        Dueling_DDQN_Learner.copy_network(self.temporal_attention, self.temporal_attention_target)
        for i in range(self.num_agents):
            q_network = Double_Attention_Model(self.input_dim, self.output_dim, self.hidden_dim).to(self.device)
            q_network_target = Double_Attention_Model(self.input_dim, self.output_dim, self.hidden_dim).to(self.device)
            q_network.set_layer_para(self.embedding, self.attention, self.temporal_attention)
            q_network_target.set_layer_para(self.embedding_target, self.attention_target, self.temporal_attention_target)
            self.agents[i].set_q_network(q_network, q_network_target)

# def change_mode(self):
        # self.q_network.change_mode()
        # self.q_network_target.change_mode()
        # for i in range(self.num_agents):
        #     self.agents[i].q_network_current.change_mode()
        #     self.agents[i].q_network_target.change_mode()
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