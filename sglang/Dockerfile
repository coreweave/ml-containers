# syntax=docker/dockerfile:1.2
ARG BASE_IMAGE
ARG BUILDER_IMAGE="${BASE_IMAGE}"

FROM ${BUILDER_IMAGE} AS builder

ARG BUILD_TORCH_CUDA_ARCH_LIST='8.0 8.6 8.9 9.0 10.0 12.0+PTX'

# v0.2.7.post1
ARG FLASHINFER_COMMIT='3fb73b3cd537c2d0f27ffac7e3cae08fca546af0'
ARG CUTLASS_COMMIT=''
ARG VLLM_COMMIT=''
# Slightly after v0.4.8, includes sgl-kernel v0.2.1
ARG SGLANG_COMMIT='f18a8fddd4182c9802f62f406d50978cf4b11b96'
ARG DECORD_COMMIT='d2e56190286ae394032a8141885f76d5372bd44b'
ARG TRITON_COMMIT=''

ARG MAX_JOBS=8

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
