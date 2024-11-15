# This is the most recent torch-extras container with the highest versions of all
# dependencies, i.e. it's probably too new :/
ARG BASE_IMAGE="ghcr.io/coreweave/ml-containers/torch-extras:f575d1b-nccl-cuda12.6.1-ubuntu22.04-nccl2.23.4-1-torch2.5.0-vision0.20.0-audio2.5.0"
FROM $BASE_IMAGE

COPY requirements.txt /tmp/requirements.txt

RUN python3 -m pip install -U --no-cache-dir \
    -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt