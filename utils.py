import configparser
from sumolib import checkBinary
import os
import sys


def get_configuration(config_file):
    """
    Read the config file regarding the training and import its content
    """
    content = configparser.ConfigParser()
    content.read(config_file)
    config = {}
    # environment parameter
    config['gui'] = content['simulation'].getboolean('gui')
    config['total_episodes'] = content['simulation'].getint('total_episodes')
    config['max_step'] = content['simulation'].getint('max_step')
    config['green_duration'] = content['simulation'].getint('green_duration')
    config['yellow_duration'] = content['simulation'].getint('yellow_duration')
    config['sim_seed'] = content['simulation'].getint('sim_seed')

    # memory parameter
    config['buffer_size'] = content['memory'].getint('buffer_size')
    config['batch_size'] = content['memory'].getint('batch_size')

    # agent parameter
    config['num_states_obs'] = content['agent'].getint('num_states_obs')
    config['num_states_phase'] = content['agent'].getint('num_states_phase')
    config['num_states_lanes'] = content['agent'].getint('num_states_lanes')
    config['num_actions'] = content['agent'].getint('num_actions')
    config['update_step'] = content['agent'].getint('update_step')
    config['coef_reward'] = content['agent'].getfloat('coef_reward')
    config['gamma'] = content['agent'].getfloat('gamma')
    config['tau'] = content['agent'].getfloat('tau')
    config['epsilon_init'] = content['agent'].getfloat('epsilon_init')
    return config
