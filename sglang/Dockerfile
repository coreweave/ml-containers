# syntax=docker/dockerfile:1.2
ARG BASE_IMAGE
ARG BUILDER_IMAGE="${BASE_IMAGE}"

FROM ${BUILDER_IMAGE} AS builder

ARG BUILD_TORCH_CUDA_ARCH_LIST='8.0 8.6 8.9 9.0 10.0 12.0+PTX'

# v0.6.3
ARG FLASHINFER_COMMIT='v0.6.3'
# v0.5.9
ARG SGLANG_COMMIT='v0.5.9'
ARG DECORD_COMMIT='d2e56190286ae394032a8141885f76d5372bd44b'

ARG MAX_JOBS=1

WORKDIR /build
COPY build.bash /build/
RUN mkdir /wheels && \
    if [ -n "${MAX_JOBS}" ]; then \
      export CMAKE_BUILD_PARALLEL_LEVEL="${MAX_JOBS}"; \
    else \
      unset MAX_JOBS; \
    fi && \
    bash build.bash -a "${BUILD_TORCH_CUDA_ARCH_LIST}" && \
    rm -rf /build/*
COPY install.bash /wheels/

FROM ${BASE_IMAGE}
RUN --mount=type=bind,from=builder,source=/wheels,target=/wheels \
    cd /wheels && \
    bash install.bash
