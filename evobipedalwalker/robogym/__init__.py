from gym.envs.registration import register
#from gym.scoreboard.registration import add_task, add_group

register(
    id='AugmentAnt-v1',
    entry_point='robogym:AugmentAnt',
    max_episode_steps=1000,
    reward_threshold=2500.0
    )

from robogym.gym_mujoco_walkers import AugmentAnt

