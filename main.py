import argparse
import numpy as np
from env import make_env

def main():
    parser = argparse.ArgumentParser(description=("Robot evo learning."))
    parser.add_argument('--env', 
                        type=str, 
                        help="""
                        AugmentBipedalWalker,
                        AugmentBipedalWalkerSmallLegs,
                        AugmentBipedalWalkerHardcore,
                        AugmentBipedalWalkerHardcoreSmallLegs,
                        AugmentAnt
                        """)
    args = parser.parse_args()

    env = make_env(args.env, render_mode="human")
    env.seed(seed=42)

    if args.env.startswith("AugmentBipedalWalker"):
        num_param = 8
    elif args.env.startswith("AugmentAnt"):
        num_param = 36
    else:
        print("This agent is undefined")
    augment_vector = (1.0 + (np.random.rand(num_param)*2-1.0)*0.5)
    env.augment_env(augment_vector)
    observation, info = env.reset()
    for _ in range(1000):
        observation, reward, done, info = env.step(env.action_space.sample())
        env.render()
        if done:
            observation, info = env.reset()
    env.close()

if __name__ == "__main__":
   main()