import numpy as np
import gym
# import pybullet as p


def make_env(env_name, seed=-1, render_mode=False):
  if (env_name.startswith("hard")):
    from box2d.bipedal_walker import AugmentBipedalWalkerHardcore
    env = AugmentBipedalWalkerHardcore()
  else:
    from box2d.bipedal_walker import AugmentBipedalWalker
    env = AugmentBipedalWalker()
  if (seed >= 0):
    env.seed(seed)
  return env
