# syntax=docker/dockerfile:1.10
ARG BASE_IMAGE="ghcr.io/coreweave/ml-containers/torch:14744fc-nccl-cuda13.2.2-ubuntu24.04-nccl2.32.3-1-torch2.14.1-vision0.29.1-audio2.11.0-abi1"
ARG BUILDER_IMAGE="${BASE_IMAGE}"
ARG SCCACHE_VERSION="0.14.0"

FROM alpine/curl:8.17.0 AS sccache-downloader
ARG SCCACHE_VERSION
RUN ARCH=$(uname -m) && \
    curl -fsSL "https://github.com/mozilla/sccache/releases/download/v${SCCACHE_VERSION}/sccache-v${SCCACHE_VERSION}-${ARCH}-unknown-linux-musl.tar.gz" \
    | tar xz --strip-components=1 -C /opt/ "sccache-v${SCCACHE_VERSION}-${ARCH}-unknown-linux-musl/sccache" && \
    chmod 755 /opt/sccache

FROM ${BUILDER_IMAGE} AS builder-base

# Setup for sccache and its wrappers.
COPY --link --from=sccache-downloader /opt/sccache /opt/sccache
COPY --link --from=common --chmod=755 sccache*.sh /opt/
COPY --link --from=common sccache.toml /etc/sccache.toml
RUN sed -Ei 's@^(key_prefix.*)misc@\1sglang@' /etc/sccache.toml
ENV SCCACHE_CONF=/etc/sccache.toml \
    CC=/opt/sccache-cc.sh \
    CXX=/opt/sccache-c++.sh \
    TORCH_EXTENSION_SKIP_NVCC_GEN_DEPENDENCIES=1

COPY --link --from=common --chmod=755 nvcc-wrapper.py /opt/nvcc-wrapper.py
ENV CMAKE_CUDA_COMPILER='/opt/nvcc-wrapper.py' \
    CUDACXX='/opt/nvcc-wrapper.py'

WORKDIR /build
COPY build-deps.bash /build/
COPY build_progress.py /opt/
RUN python3 /opt/build_progress.py --label build-dependencies --log-path /build-logs/build-dependencies.log -- \
    bash /build/build-deps.bash

FROM builder-base AS flashinfer-source
ARG FLASHINFER_COMMIT
ARG SGLANG_PACKAGE_PROFILE=legacy
RUN <<'SOURCE'
set -e
case "${SGLANG_PACKAGE_PROFILE}" in
  legacy)
    [ -z "${FLASHINFER_COMMIT:-}" ]
    mkdir /build/flashinfer
    ;;
  baseline)
    test -n "${FLASHINFER_COMMIT}"
    git clone --filter=blob:none --no-checkout https://github.com/flashinfer-ai/flashinfer /build/flashinfer
    git -C /build/flashinfer checkout "${FLASHINFER_COMMIT}"
    test "$(git -C /build/flashinfer rev-parse HEAD)" = "${FLASHINFER_COMMIT}"
    # Only metadata is needed; published wheels contain the native kernels.
    ;;
  *) exit 1 ;;
esac
SOURCE

FROM builder-base AS sglang-source
ARG SGLANG_COMMIT
ARG SGLANG_PACKAGE_PROFILE=legacy
RUN <<'SOURCE'
set -e
test -n "${SGLANG_COMMIT}"
git clone --recursive --filter=blob:none https://github.com/sgl-project/sglang /build/sglang
cd /build/sglang
git checkout "${SGLANG_COMMIT}"
if [ "${SGLANG_PACKAGE_PROFILE}" != legacy ]; then
  test "$(git rev-parse HEAD)" = "${SGLANG_COMMIT}"
  git submodule update --init --recursive --jobs 8
fi
# Preserve upstream Torch pins: the base now supplies the required 2.14.1.

SOURCE

FROM builder-base AS package-contract
ARG SGLANG_PACKAGE_PROFILE=legacy
COPY --link --from=sglang-source /build/sglang/ /build/sglang/
COPY --link --from=flashinfer-source /build/flashinfer/ /build/flashinfer/
COPY packages.py test_packages.py build_progress.py test_build_progress.py test_build_stages.py Dockerfile install.bash build*.bash /build/
RUN <<'CONTRACT'
set -e
mkdir /contract
python3 -m unittest -v test_build_progress test_build_stages
if [ "${SGLANG_PACKAGE_PROFILE}" = legacy ]; then
  printf '# Legacy profile does not use this file.\n' > /contract/constraints.txt
  printf '# Legacy profile does not use this file.\n' > /contract/build-requirements.txt
  printf '{"profile":"legacy","skipped":true}\n' > /contract/packages-before.json
else
  python3 -m unittest -v test_packages
  python3 /opt/build_progress.py --label package-constraints --log-path /build-logs/package-constraints.log -- \
    python3 /build/packages.py constraints \
    --profile "${SGLANG_PACKAGE_PROFILE}" \
    --sglang-source /build/sglang --flashinfer-source /build/flashinfer \
    --output /contract/constraints.txt --requirements /contract/build-requirements.txt \
    --snapshot /contract/packages-before.json
  python3 /build/packages.py protocol-requirements --output /contract/protocol-requirements.txt
  python3 /opt/build_progress.py --label protocol-install --log-path /build-logs/protocol-install.log -- \
    python3 -m pip install --no-cache-dir --constraint /contract/constraints.txt \
    --requirement /contract/protocol-requirements.txt
  python3 /opt/build_progress.py --label protocol-check --log-path /build-logs/protocol-check.log -- \
    timeout 120s python3 /build/packages.py verify-protocol --output /contract/protocol-check.json
fi
CONTRACT

FROM builder-base AS flashinfer-build-base
ARG FLASHINFER_COMMIT
ARG FLASHINFER_VERSION
ARG SGLANG_PACKAGE_PROFILE=legacy
# Metadata-derived constraints also protect the SGLang compiler environment.
COPY --link --from=package-contract /contract/constraints.txt /contract/build-requirements.txt /wheels/
COPY build-flashinfer.bash /build/
RUN bash /build/build-flashinfer.bash -p dependencies

FROM flashinfer-build-base AS flashinfer-python
RUN bash /build/build-flashinfer.bash -p python

# Download the large cubin wheel independently of the CUDA-specific providers.
FROM flashinfer-build-base AS flashinfer-cubin
RUN mkdir /cubin-wheels && bash /build/build-flashinfer.bash -p cubin -o /cubin-wheels

FROM flashinfer-python AS flashinfer-builder
RUN bash /build/build-flashinfer.bash -p cache
COPY --link --from=flashinfer-cubin /cubin-wheels/ /wheels/
COPY --link --from=flashinfer-cubin /build-logs/ /build-logs/

FROM flashinfer-builder AS builder
COPY --link --from=flashinfer-source /build/flashinfer/ /build/flashinfer/
ARG SGLANG_COMMIT
ARG BUILD_TORCH_CUDA_ARCH_LIST='8.0 8.6 8.9 9.0 10.0+PTX'
ARG TARGETPLATFORM
COPY --link --from=sglang-source /build/sglang/ /build/sglang/
COPY --link --from=package-contract /contract/packages-before.json /wheels/
COPY build.bash packages.py /build/
RUN --mount=type=secret,id=s3_access_key_id,env=AWS_ACCESS_KEY_ID \
    --mount=type=secret,id=s3_secret_access_key,env=AWS_SECRET_ACCESS_KEY \
    --mount=type=tmpfs,target=/sccache \
    --mount=type=tmpfs,target=/tmp \
    . /opt/sccache-start.sh && \
    ARCH_LIST="${BUILD_TORCH_CUDA_ARCH_LIST}" && \
    if [ "$TARGETPLATFORM" = "linux/arm64" ]; then ARCH_LIST="9.0 10.0+PTX"; fi && \
    bash build.bash -a "${ARCH_LIST}" && \
    rm -rf /build/*
COPY install.bash packages.py /wheels/

FROM scratch AS build-logs
COPY --from=builder /build-logs/ /build/
COPY --from=package-contract /build-logs/ /contract/

FROM ${BASE_IMAGE}
ARG SGLANG_COMMIT
ARG FLASHINFER_COMMIT
ARG SGLANG_PACKAGE_PROFILE=legacy
LABEL com.coreweave.sglang.source-ref="${SGLANG_COMMIT}" \
      com.coreweave.sglang.flashinfer-ref="${FLASHINFER_COMMIT}" \
      com.coreweave.sglang.package-profile="${SGLANG_PACKAGE_PROFILE}"
# The CUDA 13 sgl-deep-ep wheel links against NVSHMEM but does not bundle it.
# SGLang JIT-compiles its HiCache native hash extension on first use, which
# requires the OpenSSL headers in addition to the runtime library.
RUN apt-get -qq update && \
    apt-get -q install --no-install-recommends -y \
      libnvshmem3-cuda-13 \
      libssl-dev && \
    NCCL_LIB_DIR="/usr/lib/$(dpkg-architecture -qDEB_HOST_MULTIARCH)" && \
    NVSHMEM_LIB_DIR="/usr/lib/$(dpkg-architecture -qDEB_HOST_MULTIARCH)/nvshmem/13" && \
    mkdir -p /opt/nccl && \
    ln -s "${NCCL_LIB_DIR}" /opt/nccl/lib && \
    ln -s /usr/include /opt/nccl/include && \
    echo "${NVSHMEM_LIB_DIR}" > /etc/ld.so.conf.d/nvshmem.conf && \
    ldconfig && \
    rm -rf /var/lib/apt/lists/*

# sgl-deep-ep discovers NCCL from Python package metadata or this explicit
# prefix. The base installs NCCL from apt, so expose its multiarch layout under
# the conventional prefix DeepEP expects without installing a second NCCL copy.
ENV EP_NCCL_ROOT_DIR=/opt/nccl

RUN --mount=type=bind,from=builder,source=/wheels,target=/wheels \
    cd /wheels && \
    bash install.bash
