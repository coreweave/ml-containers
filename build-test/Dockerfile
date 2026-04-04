# syntax=docker/dockerfile:1.7

# sccache integration test with CUDA compilation and object storage backend.

ARG SCCACHE_VERSION="0.14.0"
ARG BUILDER_BASE_IMAGE="nvidia/cuda:12.9.1-devel-ubuntu22.04"
ARG FINAL_BASE_IMAGE="nvidia/cuda:12.9.1-base-ubuntu22.04"

FROM alpine/curl:8.17.0 AS sccache-downloader
ARG SCCACHE_VERSION
RUN ARCH=$(uname -m) && \
    curl -fsSL "https://github.com/mozilla/sccache/releases/download/v${SCCACHE_VERSION}/sccache-v${SCCACHE_VERSION}-${ARCH}-unknown-linux-musl.tar.gz" \
    | tar xz --strip-components=1 -C /opt/ "sccache-v${SCCACHE_VERSION}-${ARCH}-unknown-linux-musl/sccache" && \
    chmod 755 /opt/sccache

FROM ${BUILDER_BASE_IMAGE} AS builder
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -qq update && \
    apt-get -qq install --no-install-recommends -y curl python3 python3-pip build-essential ninja-build && \
    apt-get clean

COPY --link --from=sccache-downloader /opt/sccache /opt/sccache
COPY --link --chmod=755 sccache-start.sh /opt/sccache-start.sh
COPY --link --chmod=755 sccache.sh /opt/sccache.sh
COPY --link sccache.toml /etc/sccache.toml
RUN sed -Ei 's@^(key_prefix.*)misc@\1build-test@' /etc/sccache.toml
ENV SCCACHE_CONF=/etc/sccache.toml

# C/C++ compiler launchers (CUDA is handled inside nvcc-wrapper)
ENV CMAKE_C_COMPILER_LAUNCHER=/opt/sccache.sh \
    CMAKE_CXX_COMPILER_LAUNCHER=/opt/sccache.sh

COPY --link --chmod=755 nvcc-wrapper.py /build/nvcc-wrapper.py
COPY --link test.cu /build/test.cu
WORKDIR /build

ENV NVCC_APPEND_FLAGS="-gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_90,code=sm_90"
ENV PYTORCH_NVCC='/build/nvcc-wrapper.py' \
    CMAKE_CUDA_COMPILER='/build/nvcc-wrapper.py' \
    CUDACXX='/build/nvcc-wrapper.py'

# Small CUDA test: nvcc-wrapper -> sccache -> nvcc
# Runs twice in the same step to verify in-build cache hits.
RUN --mount=type=secret,id=s3_access_key_id \
    --mount=type=secret,id=s3_secret_access_key \
    --mount=type=tmpfs,target=/sccache \
    --mount=type=tmpfs,target=/tmp \
    . /opt/sccache-start.sh && \
    echo "=== Build 1 (expect cache miss) ===" && \
    /build/nvcc-wrapper.py -O2 -Xcompiler -fPIC -c -o test.o test.cu && \
    nvcc -o test test.o && \
    /opt/sccache --show-stats && \
    echo "=== Build 2 (expect cache hit) ===" && \
    rm -f test test.o && \
    /build/nvcc-wrapper.py -O2 -Xcompiler -fPIC -c -o test.o test.cu && \
    nvcc -o test test.o

# Install PyTorch (from pip, not from source; just testing sccache here, not torch)
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu129

# Build xformers wheel with sccache from a fixed source path
ARG XFORMERS_VERSION="0.0.35"
RUN --mount=type=tmpfs,target=/tmp/sdist \
    pip3 download --no-binary=xformers --no-deps -d /tmp/sdist \
      "xformers==${XFORMERS_VERSION}" && \
    mkdir /build/xformers && \
    tar xf /tmp/sdist/xformers-*.tar.gz --strip-components=1 -C /build/xformers/

RUN mkdir /build/dist

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN --mount=type=secret,id=s3_access_key_id \
    --mount=type=secret,id=s3_secret_access_key \
    --mount=type=tmpfs,target=/sccache \
    --mount=type=tmpfs,target=/tmp \
    . /opt/sccache-start.sh && \
    export TORCH_CUDA_ARCH_LIST="8.0;9.0" \
           XFORMERS_DISABLE_FLASH_ATTN=1 \
           MAX_JOBS=4 \
           PYTHONUNBUFFERED=1 \
           LAMP=FRIEND \
           COW=MOO \
           CIRNO=STRONGEST \
           YEE=HAW && \
    echo "=== xformers build ===" && \
    pip3 wheel -w /build/dist -v --no-cache-dir --no-build-isolation --no-deps \
      /build/xformers/ 2> \
    >(grep -Ev --line-buffered 'ptxas info\s*:|bytes spill stores' >&2)
SHELL ["/bin/sh", "-c"]

# Final image
FROM ${FINAL_BASE_IMAGE}
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -qq update && \
    apt-get -qq install --no-install-recommends -y python3 python3-pip && \
    apt-get clean

RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu129

RUN --mount=type=bind,from=builder,source=/build/dist,target=/tmp/dist \
    pip3 install --no-cache-dir /tmp/dist/*.whl
