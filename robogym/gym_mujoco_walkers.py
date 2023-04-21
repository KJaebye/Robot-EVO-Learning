from robogym.scene_abstract import cpp_household
#from roboschool.scene_stadium import SinglePlayerStadiumScene
from robogym.gym_forward_walker import RoboschoolForwardWalker
from robogym.gym_mujoco_xml_env import AugmentMujocoXmlEnv
import gym, gym.spaces, gym.utils, gym.utils.seeding
import numpy as np
import os, sys, time
from robogym.generate_mujoco_xml import generate_ant_xml, generate_hopper_xml, generate_half_cheetah_xml

FILE_SLEEP_TIME = 4.7

class RoboschoolForwardWalkerMujocoXML(RoboschoolForwardWalker, AugmentMujocoXmlEnv):
    def __init__(self, fn, robot_name, action_dim, obs_dim, power):
        AugmentMujocoXmlEnv.__init__(self, fn, robot_name, action_dim, obs_dim)
        RoboschoolForwardWalker.__init__(self, power)

class AugmentHopper(RoboschoolForwardWalkerMujocoXML):
    foot_list = ["foot"]
    def __init__(self):
        RoboschoolForwardWalkerMujocoXML.__init__(self, "hopper.xml", "torso", action_dim=3, obs_dim=15, power=0.75)
    def alive_bonus(self, z, pitch):
        return +1 if z > 0.8 and abs(pitch) < 1.0 else -1
    def augment_env(self, scale_vector):
        self.model_xml = generate_hopper_xml(scale_vector)
        time.sleep(FILE_SLEEP_TIME) # sleep for a random amount of time to avoid harddisk file errors.

class RoboschoolWalker2d(RoboschoolForwardWalkerMujocoXML):
    foot_list = ["foot", "foot_left"]
    def __init__(self):
        RoboschoolForwardWalkerMujocoXML.__init__(self, "walker2d.xml", "torso", action_dim=6, obs_dim=22, power=0.40)
    def alive_bonus(self, z, pitch):
        return +1 if z > 0.8 and abs(pitch) < 1.0 else -1
    def robot_specific_reset(self):
        RoboschoolForwardWalkerMujocoXML.robot_specific_reset(self)
        for n in ["foot_joint", "foot_left_joint"]:
            self.jdict[n].power_coef = 30.0

class AugmentHalfCheetah(RoboschoolForwardWalkerMujocoXML):
    foot_list = ["ffoot", "fshin", "fthigh",  "bfoot", "bshin", "bthigh"]  # track these contacts with ground
    def __init__(self):
        RoboschoolForwardWalkerMujocoXML.__init__(self, "half_cheetah.xml", "torso", action_dim=6, obs_dim=26, power=0.90)
    def alive_bonus(self, z, pitch):
        # Use contact other than feet to terminate episode: due to a lot of strange walks using knees
        return +1 if np.abs(pitch) < 1.0 and not self.feet_contact[1] and not self.feet_contact[2] and not self.feet_contact[4] and not self.feet_contact[5] else -1
    def robot_specific_reset(self):
        RoboschoolForwardWalkerMujocoXML.robot_specific_reset(self)
        self.jdict["bthigh"].power_coef = 120.0
        self.jdict["bshin"].power_coef  = 90.0
        self.jdict["bfoot"].power_coef  = 60.0
        self.jdict["fthigh"].power_coef = 140.0
        self.jdict["fshin"].power_coef  = 60.0
        self.jdict["ffoot"].power_coef  = 30.0
    def augment_env(self, scale_vector):
        self.model_xml = generate_half_cheetah_xml(scale_vector)
        time.sleep(FILE_SLEEP_TIME) # sleep for 200 milliseconds

class AugmentAnt(RoboschoolForwardWalkerMujocoXML):
    foot_list = ['front_left_foot', 'front_right_foot', 'left_back_foot', 'right_back_foot']
    def __init__(self): # need to be able to reset this at every "reset"
        RoboschoolForwardWalkerMujocoXML.__init__(self, "ant.xml", "torso", action_dim=8, obs_dim=28, power=2.5)
    def alive_bonus(self, z, pitch):
        return +1 if z > 0.26 else -1  # 0.25 is central sphere rad, die if it scrapes the ground
    def augment_env(self, scale_vector):
        self.model_xml = generate_ant_xml(scale_vector)
        time.sleep(FILE_SLEEP_TIME) # sleep for 200 milliseconds
