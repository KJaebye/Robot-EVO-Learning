# Evo Competition
The task is to evolve and train a pair of agents to compete with each other.

Two levels of difficulties are provided:
- Basic: No morph evolution, only control. Train a pair of agents(can be $\texttt{ant}$, $\texttt{bug}$, $\texttt{spider}$) in a two-player competition game.
- Advanced: Morph evolution and control. Train a pair of agents(can be $\texttt{dev-ant}$, $\texttt{dev-bug}$, $\texttt{dev-spider}$) in a two-player competition game, with morphological evolution.

Two task scenarios are recommended: $\texttt{run-to-goal}$ and $\texttt{sumo}$.

## Pre-requirements
For better experience, we have built the running environment in docker and provide visualization via X11. Please make sure the docker engine and ssh are installed on your pc. 

> Note: Codes are not included in the docker container. The container only provides python and display environment.

### Docker
For installing **docker engine**, please refer to: https://docs.docker.com/engine/install/ubuntu/ and a chinese documentation https://docker-practice.github.io/zh-cn/install/ubuntu.html

> Note: Docker Engine is different from Docker Desktop. Docker Engine is the core of the docker service. While Docker Desktop is a visualized platform for managing containers and images.

Create docker volume named `ws` for files mapping (see what is docker volume at https://zhuanlan.zhihu.com/p/468642439):

```bash
docker volume create --name ws --opt type=none --opt device={path-to-your-code-folder} --opt o=bind
```

Use gpu in docker:
```bash
sudo sh ../nvidia-container-runtime-script.sh
sudo apt-get install nvidia-container-runtime
sudo systemctl restart doccker.service
```


## Build the Environment
Build dockerfile firstly.
```
cd evo_struture/docker
docker build -t 'evo-competition:latest' . 
```
Get the **image_id**.

Run the image, make sure the volume is mounted.
```
xhost +
docker run -it --name evo-competition --gpus all -v ws:/root/ws {image_id} /bin/bash
docker start evo-competition
```
> Note: Replace the {image_id} with the **image_id** in the last step.

Search "Remote - SSH" in vscode Extensions and install it. Then find the container in Dev Containers list and click `Attach in New Window`, and then you can interactively code in container.

## Task Explanation in Brief
Env files for basic difficulty are in `gym_compete`; env files for advanced difficulty are located in `competevo`. All the environments have been registerated in gymnasium, but you are free to change anything in these files.

You can try basic difficulty env examples like this:
```bash
cd evo_competition
python example.py --cfg config/robo-sumo-ants-v0.yaml
```
Basic difficulties:
```
config/robo-sumo-ants-v0.yaml
config/robo-sumo-bugs-v0.yaml
config/robo-sumo-spiders-v0.yaml
config/sumo-humans-v0.yaml
config/run-to-goal-ants-v0.yaml
config/run-to-goal-bugs-v0.yaml
config/run-to-goal-spiders-v0.yaml
config/run-to-goal-humans-v0.yaml
config/you-shall-not-pass-humans-v0.yaml
```
Advanced difficulties:
```
config/robo-sumo-devants-v0.yaml
config/robo-sumo-devbugs-v0.yaml
config/robo-sumo-devspiders-v0.yaml
config/run-to-goal-devants-v0.yaml
config/run-to-goal-devbugs-v0.yaml
config/run-to-goal-devspiders-v0.yaml
```
Cross competition can be achieved by envs below, where you could edit registration file at `/competevo/__init__.py` and `/gym_compete/__init__.py` and assign specific animals:
```
config/robo-sumo-animals-v0.yaml
config/run-to-goal-animals-v0.yaml
```

The morph evolution method of agents is accomplished by parameterize the attributes of agents. Details can be found in files at `/competevo/evo_envs/agents/dev_*.py`

> Note: For students who choose advanced difficulty, we recommend to compare the win rates of competitions between **evolved agents** vs **normal agents**, as well as **normal agents** vs **normal agents**, for a better comperison of the morph evolution.

## Possible References
```
https://competevo.github.io/

Ha, David. "Reinforcement learning for improving agent design." Artificial life 25.4 (2019): 352-365.

Schaff, Charles, et al. "Jointly learning to construct and control agents using deep reinforcement learning." 2019 international conference on robotics and automation (ICRA). IEEE, 2019.

Yuan, Ye, et al. "Transform2act: Learning a transform-and-control policy for efficient agent design." arXiv preprint arXiv:2110.03659 (2021).

Wang, Tingwu, et al. "Neural graph evolution: Towards efficient automatic robot design." arXiv preprint arXiv:1906.05370 (2019).
```