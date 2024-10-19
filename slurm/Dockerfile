# Image for SLURM nodes
# Will be built on both login and compute node SUNK images

ARG BASE_IMAGE
FROM $BASE_IMAGE

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get -qq update && \
        apt-get -qq install -y --no-install-recommends \
        git \
        aria2

# UDP, TCP, HTTP
EXPOSE 6800
EXPOSE 6881-6889
EXPOSE 80

ADD motd /etc/motd
