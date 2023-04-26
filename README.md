# ENV FILES for Course **Computational Intelligence and Robotics**
## Description
This repo provides robot learning environments for course **Computational Intelligence and Robotics** directed by Prof. Huaping Liu at Tsinghua University.

The task is training a Bipedalwalker to learn control and evolve its morph for better control.

Two difficulties of tasks are provided:
1. easy: training a bipedalwalker on flat.
2. hard: training a bipedalwalker on rough terrain.

We have already provided the RL environment files, and built a common python env for RL. The torch version is 1.12.0.

## Build Image from Dockerfile
```
cd docker
docker build -t 'evo-learning:bipedalwalker' . 
```
Get the **image_id**.

## Run Container
```
docker run -it --name evo-learning -v ws:/root/ws --gpus=all -v /tmp/.x11-unix:/tmp/.x11-unix -e DISPLAY=$DISPLAY -e GDK_SCALE -e GDK_DPI_SCALE -p 9022:22 {image_id} /bin/bash
```
> Important: The 'ws' is the docker volume where you should clone and put this repo, so that you can edit code inside the container.

> Note: Replace the {image_id} with the **image_id** in the last step.

## Entering Containner
Entering containner using ssh with default passwd '123123':
```
ssh -Y root@0.0.0.0 -p 9022
```
> Note: It's necessary to use "-Y" option to allow X11 forwarding.

## Example
We provide a gym like environment api:
```
    env = make_env(args.env, render_mode="human")
    env.seed(seed=42)

    num_param = 8
    augment_vector = (1.0 + (np.random.rand(num_param)*2-1.0)*0.5)
    env.augment_env(augment_vector)
    observation, info = env.reset()
    for _ in range(1000):
        observation, reward, done, info = env.step(env.action_space.sample())
        env.render()
        if done:
            observation, info = env.reset()
    env.close()
```
ENVNAME can be one of:
```
AugmentBipedalWalker
AugmentBipedalWalkerSmallLegs
AugmentBipedalWalkerHardcore
AugmentBipedalWalkerHardcoreSmallLegs

AugmentAnt
```
Run main.py to create a bipedalwalker or ant environment with random morph and random control policy.
```
python main.py --env ENVNAME
```