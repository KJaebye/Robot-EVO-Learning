import numpy as np
import os
import os.path as osp
from gym import utils
from utils.envs.common.mujoco_env_gym import MujocoEnv
from utils.xml_robot import Robot
import shutil



class GapEnv(MujocoEnv, utils.EzPickle):

    HFIELD_FNAME = 'gap_terrain.png'
    hfield_dir = '/tmp/mujoco_terrains'

    def __init__(self, cfg):

        self.cur_t = 0
        self.hfield_file = f'{self.hfield_dir}/{self.HFIELD_FNAME}'
        if not osp.exists(self.hfield_file):           
            os.makedirs(self.hfield_dir, exist_ok=True)
            shutil.copyfile(f'assets/mujoco_terrains/{self.HFIELD_FNAME}', self.hfield_file)

        self.cfg = cfg
        self.model_xml_file = 'assets/mujoco_envs/gap.xml'
        # robot xml
        self.robot = Robot(cfg.robot_cfg, xml=self.model_xml_file)
        self.init_xml_str = self.robot.export_xml_string()
        self.cur_xml_str = self.init_xml_str.decode('utf-8')
        # design options
        self.clip_qvel = cfg.obs_specs.get('clip_qvel', False)
        self.use_projected_params = cfg.obs_specs.get('use_projected_params', True)
        self.abs_design = cfg.obs_specs.get('abs_design', False)
        self.use_body_ind = cfg.obs_specs.get('use_body_ind', False)
        self.design_ref_params = self.get_attr_design()
        self.design_cur_params = self.design_ref_params.copy()
        self.design_param_names = self.robot.get_params(get_name=True)
        self.index_base = self.cfg.add_body_condition.get('max_nchild', 3) + 1
        MujocoEnv.__init__(self, self.model_xml_file, 4)
        utils.EzPickle.__init__(self)
        self.ground_geoms = np.where(self.model.geom_bodyid == 0)[0]
        print(self.ground_geoms)

    def allow_add_body(self, body):
        add_body_condition = self.cfg.add_body_condition
        max_nchild = add_body_condition.get('max_nchild', 3)
        min_nchild = add_body_condition.get('min_nchild', 0)
        return body.depth >= self.cfg.min_body_depth and body.depth < self.cfg.max_body_depth - 1 and len(body.child) < max_nchild and len(body.child) >= min_nchild
    
    def allow_remove_body(self, body):
        if body.depth >= self.cfg.min_body_depth + 1 and len(body.child) == 0:
            if body.depth == 1:
                return body.parent.child.index(body) > 0
            else:
                return True
        return False

    def apply_skel_action(self, skel_action):
        bodies = list(self.robot.bodies)
        for body, a in zip(bodies, skel_action):
            if a == 1 and self.allow_add_body(body):
                self.robot.add_child_to_body(body)
            if a == 2 and self.allow_remove_body(body):
                self.robot.remove_body(body)

        xml_str = self.robot.export_xml_string()
        self.cur_xml_str = xml_str.decode('utf-8')
        try:
            self.reload_sim_model(xml_str.decode('utf-8'))
        except:
            print(self.cur_xml_str)
            return False      
        self.design_cur_params = self.get_attr_design()
        return True

    def set_design_params(self, in_design_params):
        design_params = in_design_params
        for params, body in zip(design_params, self.robot.bodies):
            body.set_params(params, pad_zeros=True, map_params=True)
            # new_params = body.get_params([], pad_zeros=True, demap_params=True)
            body.sync_node()

        xml_str = self.robot.export_xml_string()
        self.cur_xml_str = xml_str.decode('utf-8')
        try:
            self.reload_sim_model(xml_str.decode('utf-8'))
        except:
            print(self.cur_xml_str)
            return False
        if self.use_projected_params:
            self.design_cur_params = self.get_attr_design()
        else:
            self.design_cur_params = in_design_params.copy()
        return True

    def step(self, a):
        raise NotImplementedError("Error! This function should be designed by you!")
    
    def get_obs(self):
        raise NotImplementedError("Error! This function should be designed by you!")
    
    def reset_model(self):
        self.reset_robot()
        self.reset_state(False)
        return self.get_obs()
    
    def get_attr_fixed(self):
        obs = []
        for i, body in enumerate(self.robot.bodies):
            obs_i = []
            if 'depth' in self.attr_specs:
                obs_depth = np.zeros(self.cfg.max_body_depth)
                obs_depth[body.depth] = 1.0
                obs_i.append(obs_depth)
            if 'jrange' in self.attr_specs:
                obs_jrange = body.get_joint_range()
                obs_i.append(obs_jrange)
            if 'skel' in self.attr_specs:
                obs_add = self.allow_add_body(body)
                obs_rm = self.allow_remove_body(body)
                obs_i.append(np.array([float(obs_add), float(obs_rm)]))
            if len(obs_i) > 0:
                obs_i = np.concatenate(obs_i)
                obs.append(obs_i)
        
        if len(obs) == 0:
            return None
        obs = np.stack(obs)
        return obs

    def get_attr_design(self):
        obs = []
        for i, body in enumerate(self.robot.bodies):
            obs_i = body.get_params([], pad_zeros=True, demap_params=True)
            obs.append(obs_i)
        obs = np.stack(obs)
        return obs

    def get_body_index(self):
        index = []
        for i, body in enumerate(self.robot.bodies):
            ind = int(body.name, base=self.index_base)
            index.append(ind)
        index = np.array(index)
        return index
    
    def reset_state(self, add_noise):
        if add_noise:
            qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
            qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        else:
            qpos = self.init_qpos
            qvel = self.init_qvel
        self.set_state(qpos, qvel)

    def reset_robot(self):
        del self.robot
        self.robot = Robot(self.cfg.robot_cfg, xml=self.init_xml_str, is_xml_str=True)
        self.cur_xml_str = self.init_xml_str.decode('utf-8')
        self.reload_sim_model(self.cur_xml_str)
        self.design_ref_params = self.get_attr_design()
        self.design_cur_params = self.design_ref_params.copy()

    def viewer_setup(self):
        # self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = 4
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.lookat[0] = self.data.qpos[0] 
        self.viewer.cam.elevation = -20