# syntax=docker/dockerfile:1.2
ARG BASE_IMAGE
ARG BUILDER_IMAGE="${BASE_IMAGE}"

FROM ${BUILDER_IMAGE} AS builder

ARG BUILD_TORCH_CUDA_ARCH_LIST='8.0 8.6 8.9 9.0 10.0+PTX'

ARG SGLANG_COMMIT='1519acf37c23f2189adb93f57ca9cd2db1bebf18'
ARG DECORD_COMMIT='d2e56190286ae394032a8141885f76d5372bd44b'

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
