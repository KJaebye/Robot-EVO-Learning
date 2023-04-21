import numpy as np
import gym
import pybullet as p


def make_env(env_name, seed=-1, render_mode=False):
  if (env_name.startswith("AugmentBipedalWalker")):
    if (env_name.startswith("AugmentBipedalWalkerHardcore")):
      if (env_name.startswith("AugmentBipedalWalkerHardcoreSmallLegs")):
        from box2d.bipedal_walker import AugmentBipedalWalkerHardcoreSmallLegs
        env = AugmentBipedalWalkerHardcoreSmallLegs()
      else:
        from box2d.bipedal_walker import AugmentBipedalWalkerHardcore
        env = AugmentBipedalWalkerHardcore()
    elif (env_name.startswith("AugmentBipedalWalkerSmallLegs")):
      from box2d.bipedal_walker import AugmentBipedalWalkerSmallLegs
      env = AugmentBipedalWalkerSmallLegs()
    elif (env_name.startswith("AugmentBipedalWalkerTallLegs")):
      from box2d.bipedal_walker import AugmentBipedalWalkerTallLegs
      env = AugmentBipedalWalkerTallLegs()
    else:
      from box2d.bipedal_walker import AugmentBipedalWalker
      env = AugmentBipedalWalker()
  else:
    if env_name.startswith("Augment"):
      import robogym
    if env_name.startswith("AugmentAnt"):
      from robogym import AugmentAnt
      env = AugmentAnt()
    else:
      env = gym.make(env_name)
    if render_mode and not env_name.startswith("Augment"):
      env.render("human")
  if (seed >= 0):
    env.seed(seed)
  return env
