# syntax=docker/dockerfile:1.2

# Scripts for building and installing vLLM for arm64 and amd64
# are adapted from code here: https://github.com/coreweave/ml-containers/tree/es/actions/sglang
ARG BASE_IMAGE="ghcr.io/coreweave/ml-containers/torch:es-actions-7095c59-base-cuda12.4.1-ubuntu22.04-torch2.5.1-vision0.20.0-audio2.5.0"
ARG BUILDER_IMAGE="nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04"

FROM ${BUILDER_IMAGE} AS builder

ARG BUILD_TORCH_CUDA_ARCH_LIST='8.0 8.6 8.9 9.0 10.0+PTX'
ARG CUTLASS_COMMIT='b78588d1630aa6643bf021613717bafb705df4ef'
ARG VLLM_COMMIT='5095e966069b9e65b7c4c63427e06cebacaad0a0'


WORKDIR /build
COPY build.bash /build/
RUN mkdir /wheels && \
    bash build.bash -a "${BUILD_TORCH_CUDA_ARCH_LIST}" && \
    rm -rf /build/*
COPY install.bash /wheels/

FROM ${BASE_IMAGE}
RUN --mount=type=bind,from=builder,source=/wheels,target=/wheels \
    cd /wheels && \
    bash install.bash