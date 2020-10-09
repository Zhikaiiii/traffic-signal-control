import configparser
import random
import numpy as np
import torch
import argparse


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


def set_configuration():
    parser = argparse.ArgumentParser()
    parser.add_argument("port", type=int)
    parser.add_argument("--para_dir", type=str, default='para_config.ini')
    parser.add_argument("--env", type=str, default='Grid9')
    parser.add_argument("--gui", type=bool, default=False)
    parser.add_argument("--load_model", type=bool, default=False)
    parser.add_argument("--save_model", type=bool, default=False)
    parser.add_argument("--load_model_dir", type=str)
    parser.add_argument("--save_model_dir", type=str)
    args = parser.parse_args()
    return args


def get_configuration(config_file):
    """
    Read the config file regarding the training and import its content
    """
    content = configparser.ConfigParser()
    content.read(config_file)
    config = {}
    # environment parameter
    config['total_episodes'] = content['simulation'].getint('total_episodes')
    config['max_step'] = content['simulation'].getint('max_step')
    config['green_duration'] = content['simulation'].getint('green_duration')
    config['yellow_duration'] = content['simulation'].getint('yellow_duration')
    config['sim_seed'] = content['simulation'].getint('sim_seed')
    config['num_states_obs'] = content['simulation'].getint('num_states_obs')
    config['num_states_phase'] = content['simulation'].getint('num_states_phase')
    config['num_states_lanes'] = content['simulation'].getint('num_states_lanes')
    config['num_actions'] = content['simulation'].getint('num_actions')
    config['num_neighbors'] = content['simulation'].getint('num_neighbors')
    config['coef_reward'] = content['simulation'].getfloat('coef_reward')

    # model parameter
    config['agent_type'] = content['model']['agent_type']
    config['hidden_dim'] = content['model'].getint('hidden_dim')
    config['seq_len'] = content['model'].getint('seq_len')
    config['n_heads'] = content['model'].getint('n_heads')

    # memory parameter
    config['buffer_size'] = content['memory'].getint('buffer_size')
    config['batch_size'] = content['memory'].getint('batch_size')

    # agent parameter
    config['lr'] = content['agent'].getfloat('lr')
    config['update_step'] = content['agent'].getint('update_step')
    config['gamma'] = content['agent'].getfloat('gamma')
    config['tau'] = content['agent'].getfloat('tau')
    config['epsilon_init'] = content['agent'].getfloat('epsilon_init')
    config['epsilon_final'] = content['agent'].getfloat('epsilon_final')
    return config
