# Evo BipedalWalker
The task is to train a Bipedalwalker to learn control and evolve its morph for better performance.

Two levels of difficulty of tasks are provided:
1. easy: training a Bipedalwalker on flat.
2. hard: training a Bipedalwalker on rough terrain.

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

## Build the Environment
To realize graphic display by using X11 forwarding, we provide two methods: VScode Dev Containers and sshd forwarding X11. We highly recommand VScode Dev Containers.

Build dockerfile firstly.
```
cd evo_bipedalwalker/docker
docker build -t 'evo-bipedalwalker:latest' . 
```
Get the **image_id**.

### VScode Dev Containers (Highly Recommended)
Run the image, make sure the volume is mounted.
```
xhost +
docker run -it --name evo-bipedalwalker --gpus all -v ws:/root/ws {image_id} /bin/bash
docker start evo-bipedalwalker
```
> Note: Replace the {image_id} with the **image_id** in the last step.

Search "Remote - SSH" in vscode Extensions and install it. Then find the container in Dev Containers list and click `Attach in New Window`, and then you can interactively code in container.

### SSHD forwarding X11 (Optional)
Check ssh config allows x11 forwarding in `/etc/ssh/ssh_config`:
```
ForwardingX11 yes
ForwardingX11Trusted yes
```
after editing the `ssh_config`, restart ssh service:
```bash
systemctl restart ssh.service
```
> Note: All configurations of ssh service in docker containner have been written in dockerfile and will be built automatically with the docker image. Actually, you need to do nothing.

Run the image with ssh forwarding environments variables setting and port number.
```
xhost +
docker run -it --name evo-bipedalwalker -v ws:/root/ws -v /tmp/.x11-unix:/tmp/.x11-unix -e DISPLAY=$DISPLAY -e GDK_SCALE -e GDK_DPI_SCALE -p 9022:22 {image_id} /bin/bash
```

Then you could enter the container by using ssh with default passwd '123123':
```
ssh -Y root@0.0.0.0 -p 9022
```

## Example
cd `\evobipedalwalker` and run example.py to create a bipedalwalker environment with random morph and random control policy.
```
python main.py --env ENVNAME
```
ENVNAME={
    "easy",
    "hard"
} ("easy" by default)

## Possible References
```
Ha, David. "Reinforcement learning for improving agent design." Artificial life 25.4 (2019): 352-365.

Yuan, Ye, et al. "Transform2act: Learning a transform-and-control policy for efficient agent design." arXiv preprint arXiv:2110.03659 (2021).

Schaff, Charles, et al. "Jointly learning to construct and control agents using deep reinforcement learning." 2019 international conference on robotics and automation (ICRA). IEEE, 2019.
```