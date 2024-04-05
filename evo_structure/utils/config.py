import yaml
import os
import glob
import numpy as np


class Config:

    def __init__(self, cfg_id, cfg_dict=None):
        self.id = cfg_id
        if cfg_dict is not None:
            cfg = cfg_dict
        else:
            cfg_path = 'cfg/**/%s.yml' % cfg_id
            files = glob.glob(cfg_path, recursive=True)
            assert(len(files) == 1)
            cfg = yaml.safe_load(open(files[0], 'r'))

        # env
        self.env_name = cfg.get('env_name', 'hopper')
        self.seed = cfg.get('seed', 1)
        self.done_condition = cfg.get('done_condition', dict())
        self.env_specs = cfg.get('env_specs', dict())
        self.reward_specs = cfg.get('reward_specs', dict())
        self.obs_specs = cfg.get('obs_specs', dict())
        self.add_body_condition = cfg.get('add_body_condition', dict())
        self.max_body_depth = cfg.get('max_body_depth', 4)
        self.min_body_depth = cfg.get('min_body_depth', 1)
        self.enable_remove = cfg.get('enable_remove', True)
        self.env_init_height = cfg.get('env_init_height', False)

        # robot config
        self.robot_param_scale = cfg.get('robot_param_scale', 0.1)
        self.robot_cfg = cfg.get('robot', dict())


