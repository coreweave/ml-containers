# syntax=docker/dockerfile:1.10
ARG BASE_IMAGE="ghcr.io/coreweave/ml-containers/torch-extras:7d4f07a-nccl-cuda12.9.1-ubuntu22.04-nccl2.29.7-1-torch2.11.0-vision0.26.0-audio2.11.0-abi1"
ARG BUILDER_IMAGE="${BASE_IMAGE}"
ARG SCCACHE_VERSION="0.14.0"

FROM alpine/curl:8.17.0 AS sccache-downloader
ARG SCCACHE_VERSION
RUN ARCH=$(uname -m) && \
    curl -fsSL "https://github.com/mozilla/sccache/releases/download/v${SCCACHE_VERSION}/sccache-v${SCCACHE_VERSION}-${ARCH}-unknown-linux-musl.tar.gz" \
    | tar xz --strip-components=1 -C /opt/ "sccache-v${SCCACHE_VERSION}-${ARCH}-unknown-linux-musl/sccache" && \
    chmod 755 /opt/sccache

FROM ${BUILDER_IMAGE} AS builder

ARG BUILD_TORCH_CUDA_ARCH_LIST='8.0 8.6 8.9 9.0 10.0+PTX'
ARG TARGETPLATFORM

ARG SGLANG_COMMIT
ARG DECORD_COMMIT

# Setup for sccache and its wrappers.
COPY --link --from=sccache-downloader /opt/sccache /opt/sccache
COPY --link --from=common --chmod=755 sccache*.sh /opt/
COPY --link --from=common sccache.toml /etc/sccache.toml
RUN ARCH=$(uname -m) && \
    sed -Ei "s@^(key_prefix.*)misc@\1sglang-${ARCH}@" /etc/sccache.toml
ENV SCCACHE_CONF=/etc/sccache.toml \
    CC=/opt/sccache-cc.sh \
    CXX=/opt/sccache-c++.sh \
    TORCH_EXTENSION_SKIP_NVCC_GEN_DEPENDENCIES=1

COPY --link --from=common --chmod=755 nvcc-wrapper.py /opt/nvcc-wrapper.py
ENV CMAKE_CUDA_COMPILER='/opt/nvcc-wrapper.py' \
    CUDACXX='/opt/nvcc-wrapper.py'

WORKDIR /build
COPY build.bash /build/
RUN --mount=type=secret,id=s3_access_key_id,env=AWS_ACCESS_KEY_ID \
    --mount=type=secret,id=s3_secret_access_key,env=AWS_SECRET_ACCESS_KEY \
    --mount=type=tmpfs,target=/sccache \
    --mount=type=tmpfs,target=/tmp \
    . /opt/sccache-start.sh && \
    mkdir /wheels && \
    ARCH_LIST="${BUILD_TORCH_CUDA_ARCH_LIST}" && \
    [ "$TARGETPLATFORM" != "linux/arm64" ] || ARCH_LIST="9.0 10.0+PTX" && \
    bash build.bash -a "${ARCH_LIST}" && \
    rm -rf /build/*
COPY install.bash /wheels/

FROM ${BASE_IMAGE}
RUN --mount=type=bind,from=builder,source=/wheels,target=/wheels \
    cd /wheels && \
    bash install.bash
