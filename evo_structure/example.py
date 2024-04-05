import argparse
import numpy as np
import torch

from utils.config import Config
from envs import env_dict

def main():
    parser = argparse.ArgumentParser(description=("Robot evo learning."))
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument('--num_threads', type=int, default=20)
    parser.add_argument('--gpu_index', type=int, default=0)
    parser.add_argument('--show_noise', action='store_true', default=False)
    args = parser.parse_args()
    if args.render:
        args.num_threads = 1
    cfg = Config(args.cfg)

    dtype = torch.float64
    torch.set_default_dtype(dtype)
    device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    env_class = env_dict[cfg.env_name]
    env = env_class(cfg)
    env.seed(seed=42)

    observation = env.reset()
    for _ in range(1000):
        observation, reward, done, info = env.step() # step() function should be customized
        env.render()
        if done:
            observation = env.reset()
    env.close()

if __name__ == "__main__":
   main()