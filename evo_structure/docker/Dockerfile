# @description: Dockerfile for evo structure task
# @author: Kangyao Huang
# @date: Apirl.04.2024

FROM nvidia/cuda:11.1.1-devel-ubuntu20.04
# FROM nvidia/cudagl:10.2-devel-ubuntu18.04
# FROM nvidia/cudagl:11.1.1-devel-ubuntu20.04

RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list
RUN apt-get update
# tzdata is required below. To avoid hanging, install it first.
RUN DEBIAN_FRONTEND="noninteractive" apt-get install tzdata -y
RUN apt-get install git wget -y

# Install python3.7.
RUN apt-get install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt-get install python3.7 -y

# Make python3.7 the default python.
RUN rm /usr/bin/python3
RUN ln -s /usr/bin/python3.7 /usr/bin/python3
RUN ln -s /usr/bin/python3.7 /usr/bin/python
RUN apt-get install python3.7-distutils -y

# Install pip.
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python get-pip.py
RUN rm get-pip.py

# Prerequisites
RUN apt-get install \
  gcc \
  python3.7-dev \
  unzip -y \
  vim \
  openssh-server \
  libxrandr2 \
  libxinerama1 \
  mesa-utils \
  x11-apps

# SSH config
RUN echo "root:123123" | chpasswd
RUN echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config
RUN echo 'PasswordAuthentication yes' >> /etc/ssh/sshd_config
RUN echo 'X11Forwarding yes' >> /etc/ssh/sshd_config
RUN echo 'X11Displayoffset 10' >> /etc/ssh/sshd_config
RUN echo 'X11UseLocalhost no' >> /etc/ssh/sshd_config
RUN service ssh restart

# set SSH auto-on
RUN touch /root/start_ssh.sh
RUN echo '#!/bin/bash \n\
  LOGTIME=$(date "+%Y-%m-%d %H:%M:%S") \n\
  echo "[$LOGTIME] startup run..." >>/root/start_ssh.log \n\
  service ssh start >>/root/start_ssh.log' >> /root/start_ssh.sh
RUN chmod +x /root/start_ssh.sh
RUN echo '# startup run \n\
  if [ -f /root/start_ssh.sh ]; then \n\
      /root/start_ssh.sh \n\
  fi' >> /root/.bashrc

# python env.
WORKDIR /root/
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# Download and install mujoco210.
RUN mkdir -p .mujoco/
COPY mujoco210 /root/.mujoco/mujoco210
RUN pip install -U 'mujoco-py<2.2,>=2.1'
RUN echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin' >> /etc/bash.bashrc

# torch and torch-geometric 
RUN pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html

ENV CUDA=cu111
ENV TORCH=1.8.0
RUN pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
RUN pip install torch-sparse==0.6.12 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
RUN pip install torch-cluster==1.6.0 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
RUN pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
RUN pip install torch-geometric==1.6.1

RUN apt install -y \
    libosmesa6-dev \
    patchelf

ENV OMP_NUM_THREADS=1