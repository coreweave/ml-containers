# Image for SLURM Login nodes
# Should contain general tooling but not specific libraries as these live in the specific node images.
# It is expected that the BASE_IMAGE is a CoreWeave SUNK artifact which already includes SLURM, SSSD and SSHD. 

ARG BASE_IMAGE
FROM $BASE_IMAGE

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get -qq update && \
        apt-get -qq install -y --no-install-recommends \
        git vim nano less curl wget inetutils-ping inetutils-traceroute \
        environment-modules sudo libssl-dev s3cmd tmux libgl1-mesa-glx glances openjdk-8-jre psmisc bzip2

# Install conda
# A user can initialize conda for their shell with: /opt/conda/bin/conda init bash --user
RUN curl -o /tmp/conda_installer https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
       bash /tmp/conda_installer -b -p /opt/conda && \
       rm /tmp/conda_installer

RUN /opt/conda/bin/conda init bash

# Install micromamba
# A user can initialize micromamba with: micromamba shell init
RUN cd /tmp && \
      curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba && \
      mv bin/micromamba /usr/local/bin/ && \
      echo "alias mm=micromamba" >> /etc/profile.d/01-aliases.sh

RUN echo "JAVA_HOME=/usr/lib/jvm/java-8-openjdk" >> /etc/environment

# Add sudoer groups to nopasswd
RUN echo "%Sudoers ALL=(ALL:ALL) NOPASSWD: ALL" >> /etc/sudoers && \
    sed -i -e 's/%admin ALL=(ALL) ALL/%admin ALL=NOPASSWD:ALL/g' /etc/sudoers && \
    sed -i -e 's/%sudo ALL=(ALL) ALL/%sudo ALL=NOPASSWD:ALL/g' /etc/sudoers

# Add AWS CLI
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    sudo ./aws/install && \
    rm -r awscliv2.zip aws

# Global environments that are useful everywhere in the cluster
RUN echo "NCCL_SOCKET_IFNAME=eth0" >> /etc/environment && \
    echo "NCCL_IB_HCA=ibp" >> /etc/environment

RUN rm /etc/update-motd.d/*
ADD motd /etc/motd
