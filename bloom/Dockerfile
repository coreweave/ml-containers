ARG CUDA_VERSION_MINOR=11.6.0
FROM nvidia/cuda:${CUDA_VERSION_MINOR}-devel-ubuntu20.04

ARG CUDA_VERSION_MAJOR=11.6
ARG CUDNN_VERSION=8.4.1.50
ARG TARGET_NCCL_VERSION=2.12.7-1

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && \
        apt-get install -y --allow-downgrades --allow-change-held-packages \
        build-essential libtool autoconf automake autotools-dev unzip \
        ca-certificates software-properties-common \
        wget curl openssh-server vim \
        iputils-ping net-tools \
        libnuma1 libsubunit0 libpci-dev \
        libpmix-dev openmpi-bin openmpi-common libopenmpi-dev \
        slurmd slurmctld libslurm-dev \
        slurmdbd jq parallel squashfs-tools\
        datacenter-gpu-manager zstd bsdmainutils \
        libnccl2=$TARGET_NCCL_VERSION+cuda${CUDA_VERSION_MAJOR} \
        libnccl-dev=$TARGET_NCCL_VERSION+cuda${CUDA_VERSION_MAJOR}

# use mpicc as default compiler
ENV CC=/usr/bin/mpicc

# add mpi headers to the include path
ENV C_INCLUDE_PATH=/usr/include/mpi:$C_INCLUDE_PATH

# Now install Enroot, latest version
RUN wget https://github.com/NVIDIA/enroot/releases/download/v3.4.0/enroot_3.4.0-1_amd64.deb && \
    dpkg -i enroot_3.4.0-1_amd64.deb

# Install Pyxis
RUN wget https://github.com/NVIDIA/pyxis/archive/refs/tags/v0.12.0.tar.gz && \
    tar -xzf v0.12.0.tar.gz && \
    cd pyxis-0.12.0 && \
    make install

# Install CUDNN
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin && \
    mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub && \
    add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /" && \
    apt-get update && \
    apt-get -y install --no-install-recommends \
    libcudnn8=$CUDNN_VERSION-1+cuda${CUDA_VERSION_MAJOR} \
    libcudnn8-dev=$CUDNN_VERSION-1+cuda${CUDA_VERSION_MAJOR}

# Install NVIDIA Container Toolkit, system latest
RUN curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add - && \
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID) && \
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list && \
    apt-get update && \
    apt-get install -y nvidia-container-toolkit

# Install Docker, system latest
RUN apt-get install -y apt-transport-https ca-certificates curl gnupg-agent software-properties-common && \
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add - && \
    add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" && \
    apt-get update && \
    apt-get install -y docker-ce docker-ce-cli containerd.io libopenmpi-dev

# Install EFA Driver
RUN curl -O https://s3-us-west-2.amazonaws.com/aws-efa-installer/aws-efa-installer-latest.tar.gz \
    && tar -xf aws-efa-installer-latest.tar.gz \
    && cd aws-efa-installer \
    && ./efa_installer.sh -y --skip-kmod --skip-limit-conf --no-verify

# setup python and conda

RUN apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh

ENV PATH=/opt/conda/bin:$PATH
