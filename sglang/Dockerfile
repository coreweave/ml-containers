# syntax=docker/dockerfile:1.2
ARG BASE_IMAGE
ARG BUILDER_IMAGE="${BASE_IMAGE}"

FROM ${BUILDER_IMAGE} AS builder

ARG BUILD_TORCH_CUDA_ARCH_LIST='8.0 8.6 8.9 9.0 10.0+PTX'

ARG FLASHINFER_COMMIT='c04755e21f4d6fb7813c703f2b00a7ef012be9b8'
ARG CUTLASS_COMMIT='b78588d1630aa6643bf021613717bafb705df4ef'
ARG VLLM_COMMIT='5095e966069b9e65b7c4c63427e06cebacaad0a0'
ARG SGLANG_COMMIT='4b6f62e2bc52a528551e9a21e7b0a4945c6115bb'
ARG DECORD_COMMIT='d2e56190286ae394032a8141885f76d5372bd44b'
# Building Triton is not currently enabled,
# but this is the commit that would be used if it were
ARG TRITON_COMMIT='1e0e51c4aeb3e1beea000da5d0e494f8b9ac40dd'

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
RUN rmdir /wheels
