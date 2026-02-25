# syntax=docker/dockerfile:1

FROM torch-extras AS builder

ARG BUILD_TORCH_CUDA_ARCH_LIST='8.0 8.6 8.9 9.0a 10.0a 12.0+PTX'

# v0.6.3
ARG FLASHINFER_COMMIT='v0.6.3'
# v0.5.9
ARG SGLANG_COMMIT='v0.5.9'
ARG DECORD_COMMIT='d2e56190286ae394032a8141885f76d5372bd44b'

ARG MAX_JOBS=16

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

FROM torch-extras
RUN --mount=type=bind,from=builder,source=/wheels,target=/wheels \
    cd /wheels && \
    bash install.bash
