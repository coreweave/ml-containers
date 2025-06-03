# modified from https://github.com/apple/axlearn/blob/main/Dockerfile

ARG TARGET=gpu
ARG BASE_IMAGE=python:3.10-slim

FROM ${BASE_IMAGE} AS base

# Install curl and gpupg first so that we can use them to install google-cloud-cli.
# Any RUN apt-get install step needs to have apt-get update otherwise stale package
# list may occur when previous apt-get update step is cached. See here for more info:
# https://docs.docker.com/build/building/best-practices/#apt-get
RUN apt-get update && apt-get install -y curl gnupg

RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    apt-get update -y && \
    apt-get install -y apt-transport-https ca-certificates gcc g++ \
      git screen ca-certificates google-perftools google-cloud-cli

# Setup.
RUN mkdir -p /root
WORKDIR /root

RUN git init /root && \
    cd /root && \
    git remote add origin https://github.com/apple/axlearn.git && \
    git fetch --depth 1 origin 6e5f0b331c20311bbf7eb14ffa7def66bf300509 && \
    git checkout FETCH_HEAD
RUN rm -rf /root/.git


# Setup venv to suppress pip warnings.
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
# Install dependencies.
RUN pip install flit
RUN pip install --upgrade pip

################################################################################
# GPU container spec.                                                          #
################################################################################

FROM base AS gpu

# TODO(markblee): Support extras.
ENV PIP_FIND_LINKS=https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip install -e .[core,gpu]

################################################################################
# Final target spec.                                                           #
################################################################################

FROM ${TARGET} AS final
