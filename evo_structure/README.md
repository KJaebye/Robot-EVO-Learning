# Evo Structure for Tasks
The task is to evolve and train an agent from scratch aiming for specific tasks. The evolution of structure and attribute of the agent should be considered.

Four sub-tasks are provided: hopper, gap, ant, and swimmer. In these rl environment files, actions to the agent are predefined, like adding/deleting a limb. You are free for choosing to use these predefined functions or not. The works you should do is to design `step` and `get_obs` functions and an algorithm to generate agent morph and to learn its control.
> Note: Of course you could edit any functions as your wish. Because the final examination criterion is the moving speed of agent.

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
docker build -t 'evo-structure:latest' . 
```
Get the **image_id**.

Run the image, make sure the volume is mounted.
```
xhost +
docker run -it --name evo-structure --gpus all -v ws:/root/ws {image_id} /bin/bash
docker start evo-structure
```
> Note: Replace the {image_id} with the **image_id** in the last step.

Search "Remote - SSH" in vscode Extensions and install it. Then find the container in Dev Containers list and click `Attach in New Window`, and then you can interactively code in container.

## Task Explanation in Brief
```bash
cd evo_structure
python example --cfg ant ...
```
`--cfg` is required for indicating a task.

We provide basic actions that can be applied to the agent, including add body, remove body, et al. The agent is modeled as a robot which is described by class `Robot`.

The evolution method and even reward functions can be customized by users, and the `step` and `get_obs` function should be written according to your method. The final evaluation criteria is to compare the moving speed along $x$ axis. 

> Note: Any improvements at the reward level are incorrect since reward functions are personalised. The right way is to improve and compare the speed along $x$ axis.

## Possible References
```
Yuan, Ye, et al. "Transform2act: Learning a transform-and-control policy for efficient agent design." arXiv preprint arXiv:2110.03659 (2021).

Wang, Tingwu, et al. "Nervenet: Learning structured policy with graph neural networks." International conference on learning representations. 2018.

Wang, Tingwu, et al. "Neural graph evolution: Towards efficient automatic robot design." arXiv preprint arXiv:1906.05370 (2019).
```