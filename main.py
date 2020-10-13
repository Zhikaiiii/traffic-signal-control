from utils import get_configuration
from utils import set_random_seeds
from utils import set_configuration
from TSC_Env import TSC_Env

if __name__ == '__main__':
    args = set_configuration()
    para_config = get_configuration(args.para_dir)
    env_name = args.env
    port = args.port
    gui = args.gui
    print(para_config)
    print(env_name)
    total_episodes = para_config['total_episodes']
    sim_seed = para_config['sim_seed']
    set_random_seeds(sim_seed)
    env = TSC_Env(env_name, para_config, gui=gui, port=args.port)
    if args.load_model:
        env.agents.load_model(args.load_model_dir)
    env.run()
    if args.save_model:
        env.agents.save_model(args.save_model_dir)
    env.output_data()
    env.close()
    print('wzk')