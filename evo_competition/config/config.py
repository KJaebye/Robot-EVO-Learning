import yaml
import os
import glob
import numpy as np


class Config:

    def __init__(self, cfg_path):
        self.cfg_path = cfg_path
        files = glob.glob(cfg_path, recursive=True)
        assert(len(files) == 1)
        cfg = yaml.safe_load(open(files[0], 'r'))
        self.cfg = cfg
        # create dirs
        self.out_dir = '/root/ws/competevo/tmp'

        # main config
        self.env_name = cfg.get('env_name')
        self.use_gpu = cfg.get('use_gpu', bool)
        self.device = cfg.get('device', str)
        self.cuda_deterministic = cfg.get('cuda_deterministic', bool)

        self.runner_type = cfg.get('runner_type', "multi-agent-runner")

        # training config
        self.seed = cfg.get('seed', 1)

        # env
        self.done_condition = cfg.get('done_condition', dict())
        self.env_specs = cfg.get('env_specs', dict())
        self.reward_specs = cfg.get('reward_specs', dict())
        self.obs_specs = cfg.get('obs_specs', dict())

        self.use_parse_reward = cfg.get("use_parse_reward", True)

        # robot config
        self.robot_param_scale = cfg.get('robot_param_scale', 0.1)
        self.robot_cfg = cfg.get('robot', dict())

    def save_config(self, directory_path):
        # Create the YAML file path
        file_path = os.path.join(directory_path, 'config.yml')
        # Write the configuration data to the YAML file
        with open(file_path, 'w') as f:
            yaml.dump(self.cfg, f)
        print(f"Config file is saved at {file_path}")
