# This is the most recent torch-extras container with the highest versions of all
# dependencies, i.e. it's probably too new :/
ARG BASE_IMAGE="ghcr.io/coreweave/ml-containers/torch-extras:7d4f07a-base-cuda13.2.0-ubuntu22.04-torch2.11.0-vision0.26.0-audio2.11.0-abi1"
FROM $BASE_IMAGE

COPY requirements.txt /tmp/requirements.txt

RUN python3 -m pip install -U --no-cache-dir \
    -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt