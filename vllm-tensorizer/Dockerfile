ARG BUILDER_BASE_IMAGE="ghcr.io/coreweave/ml-containers/torch-extras:es-fa3-te-update-7a94157-nccl-cuda12.9.1-ubuntu22.04-nccl2.27.6-1-torch2.7.1-vision0.22.1-audio2.7.1-abi1"
ARG FINAL_BASE_IMAGE="ghcr.io/coreweave/ml-containers/torch-extras:es-fa3-te-update-7a94157-nccl-cuda12.9.1-ubuntu22.04-nccl2.27.6-1-torch2.7.1-vision0.22.1-audio2.7.1-abi1"

FROM scratch AS freezer
WORKDIR /
COPY --chmod=755 freeze.sh /

FROM ${BUILDER_BASE_IMAGE} AS builder-base

ARG MAX_JOBS="32"

RUN ldconfig

RUN apt-get -qq update && \
    apt-get -qq install -y --no-install-recommends \
      python3-pip git ninja-build cmake && \
    apt-get clean && \
    pip3 install -U --no-cache-dir pip packaging setuptools wheel setuptools_scm regex

# Create the /wheels directory
WORKDIR /wheels

WORKDIR /workspace

RUN --mount=type=bind,from=freezer,target=/tmp/frozen \
    /tmp/frozen/freeze.sh torch torchaudio torchvision xformers > /opt/constraints.txt

COPY --link --chmod=755 nvcc-wrapper.py /opt/nvcc-wrapper.py
ENV PYTORCH_NVCC='/opt/nvcc-wrapper.py' \
    CMAKE_CUDA_COMPILER='/opt/nvcc-wrapper.py'

ARG TARGETPLATFORM
# Switch 9.0, 10.0, and 12.0 to -a variants; preserve originals for PTX
# Flashinfer v0.28.0 in particular can only build for 12.0a but not 12.0
RUN printf 'TORCH_CUDA_ARCH_LIST=' && \
    echo "${TORCH_CUDA_ARCH_LIST}" \
    | sed -E 's@\b(9|10|12)\.0\b@\1\.0a@g; s@\+PTX\b@@g' \
    | tee /opt/arch_list.txt && \
    printf 'NVCC_WRAPPER_FILTER_CODES=' && \
    if [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
      echo 'sm_80;sm_89;sm_90;sm_100;sm_120;compute_80;compute_89;compute_90;compute_100;compute_120'; \
    else \
      echo 'sm_90;sm_100;sm_120;compute_90;compute_100;compute_120'; \
    fi \
    | tee /opt/filter_codes.txt && \
    printf '#!/bin/sh\nexport %s %s;\n' \
      'TORCH_CUDA_ARCH_LIST="$(cat /opt/arch_list.txt)"' \
      'NVCC_WRAPPER_FILTER_CODES="$(cat /opt/filter_codes.txt)"' \
    | install -m 500 /dev/stdin /opt/arch_flags.sh

FROM alpine/git:2.36.3 AS vllm-downloader
WORKDIR /git
ARG VLLM_COMMIT
RUN git clone --filter=tree:0 --no-single-branch --no-checkout \
      https://github.com/vllm-project/vllm && \
    cd vllm && \
    git checkout "${VLLM_COMMIT}" && \
    git submodule update --init --recursive --jobs 8 \
      --depth 1 --filter=tree:0


FROM alpine/git:2.36.3 AS flashinfer-downloader
WORKDIR /git
ARG FLASHINFER_COMMIT
RUN git clone --filter=tree:0 --no-single-branch --no-checkout \
      https://github.com/flashinfer-ai/flashinfer && \
    cd flashinfer && \
    git checkout "${FLASHINFER_COMMIT}" && \
    git submodule update --init --recursive --jobs 8 \
      --depth 1 --filter=tree:0


FROM alpine/git:2.36.3 AS lmcache-downloader
WORKDIR /git
ARG LMCACHE_COMMIT='v0.3.2'
RUN git clone --filter=tree:0 --no-single-branch --no-checkout \
      https://github.com/LMCache/LMCache && \
    git -C LMCache checkout "${LMCACHE_COMMIT}"


FROM alpine/git:2.36.3 AS infinistore-downloader
WORKDIR /git
ARG INFINISTORE_COMMIT='553e36296ff1da2630636e7385b56224ff74a47c'
RUN git clone --filter=tree:0 --no-single-branch --no-checkout \
      https://github.com/bytedance/InfiniStore && \
    git -C InfiniStore checkout "${INFINISTORE_COMMIT}"


FROM builder-base AS vllm-builder
RUN --mount=type=bind,from=vllm-downloader,source=/git/vllm,target=/workspace,rw \
    . /opt/arch_flags.sh && \
    if [ -z "$MAX_JOBS" ]; then unset MAX_JOBS; fi && \
    python3 -m pip install --no-cache-dir py-cpuinfo && \
    if [ -f 'use_existing_torch.py' ]; then \
      python3 use_existing_torch.py; \
    else \
      git cat-file blob \
        e489ad7a210f4234db696d1f2749d5f3662fa65b:use_existing_torch.py \
        | python3 -; \
    fi && \
    USE_CUDNN=1 USE_CUSPARSELT=1 \
    LIBRARY_PATH="/usr/local/cuda/lib64:${LIBRARY_PATH:+:$LIBRARY_PATH}" \
    CUDA_TOOLKIT_ROOT_DIR="/usr/local/cuda" \
      python3 -m pip wheel -w /wheels \
      -v --no-cache-dir --no-build-isolation --no-deps \
      -c /opt/constraints.txt \
      ./

WORKDIR /wheels


FROM builder-base AS flashinfer-builder
RUN --mount=type=bind,from=flashinfer-downloader,source=/git/flashinfer,target=/workspace,rw \
    . /opt/arch_flags.sh && \
    export TORCH_CUDA_ARCH_LIST="$(echo "${TORCH_CUDA_ARCH_LIST}" | sed 's@[67]\.0 \+@@g')" && \
    sed -i 's@torch\.cuda\.get_device_capability()@(12, 0)@' flashinfer/comm/trtllm_ar.py && \
    python3 -m flashinfer.aot && \
    python3 -m pip wheel -w /wheels \
      -v --no-cache-dir --no-build-isolation --no-deps \
      -c /opt/constraints.txt \
      ./

WORKDIR /wheels


FROM builder-base AS lmcache-builder
# LMCache must be built from source as it doesn't have pre-built ARM binaries
RUN --mount=type=bind,from=lmcache-downloader,source=/git/LMCache,target=/workspace,rw \
    . /opt/arch_flags.sh && \
    python3 -m pip install --no-cache-dir 'xxhash==3.5.0' 'setuptools_scm>=8' && \
    sed -Ei \
      '/[ "]*(torch(vision|audio)?|xformers) *[<>=~]+/d' \
    pyproject.toml requirements/*.txt && \
    python3 -m pip wheel -w /wheels \
      -v --no-cache-dir --no-build-isolation --no-deps \
      -c /opt/constraints.txt \
      ./


FROM builder-base AS infinistore-builder
# InfiniStore is required when installing LMCache
# It must also be built from source as it also doesn't have pre-built ARM binaries
RUN --mount=type=bind,from=infinistore-downloader,source=/git/InfiniStore,target=/workspace,rw \
    apt-get -qq update && \
    apt-get -q install --no-install-recommends --no-upgrade -y \
      libuv1-dev libflatbuffers-dev libspdlog-dev \
      libfmt-dev ibverbs-utils libibverbs-dev \
      libboost-dev libboost-stacktrace-dev \
      pkg-config && \
    apt-get clean && \
    python3 -m pip install --no-cache-dir meson pybind11 && \
    python3 -m pip wheel -w /wheels \
      -v --no-cache-dir --no-build-isolation --no-deps \
      -c /opt/constraints.txt \
      ./


FROM ${FINAL_BASE_IMAGE} AS base

WORKDIR /workspace

RUN apt-get -qq update && apt-get install -y --no-install-recommends curl libsodium23 && apt-get clean

RUN --mount=type=bind,from=freezer,target=/tmp/frozen \
    /tmp/frozen/freeze.sh torch torchaudio torchvision xformers > /tmp/constraints.txt

RUN --mount=type=bind,from=vllm-builder,source=/wheels,target=/tmp/wheels \
    python3 -m pip install --no-cache-dir "$(printf '%s[tensorizer]' /tmp/wheels/*.whl)" -c /tmp/constraints.txt

RUN --mount=type=bind,from=flashinfer-builder,source=/wheels,target=/tmp/wheels \
    python3 -m pip install --no-cache-dir /tmp/wheels/*.whl -c /tmp/constraints.txt

# InfiniStore must be installed before LMCache as LMCache depends on InfiniStore
RUN --mount=type=bind,from=infinistore-builder,source=/wheels,target=/tmp/wheels \
    python3 -m pip install --no-cache-dir /tmp/wheels/*.whl -c /tmp/constraints.txt

RUN --mount=type=bind,from=lmcache-builder,source=/wheels,target=/tmp/wheels \
    python3 -m pip install --no-cache-dir /tmp/wheels/*.whl -c /tmp/constraints.txt

# Copied from vLLM's Dockerfile
ARG TARGETPLATFORM

RUN if [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
      BITSANDBYTES_VER='0.42.0'; \
    else \
      BITSANDBYTES_VER='0.46.1'; \
    fi && \
    python3 -m pip install --no-cache-dir \
      accelerate hf_transfer 'modelscope!=1.15.0' "bitsandbytes>=${BITSANDBYTES_VER:?}" 'timm==0.9.10' \
      boto3 runai-model-streamer runai-model-streamer[s3] -c /tmp/constraints.txt && \
    rm /tmp/constraints.txt

EXPOSE 8080
