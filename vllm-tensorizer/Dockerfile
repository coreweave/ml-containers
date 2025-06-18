ARG BASE_IMAGE="ghcr.io/coreweave/ml-containers/torch-extras:es-compute-12.0-67208ca-nccl-cuda12.9.0-ubuntu22.04-nccl2.27.3-1-torch2.7.1-vision0.22.1-audio2.7.1-abi1"
FROM scratch AS freezer
WORKDIR /
COPY --chmod=755 freeze.sh /

FROM ${BASE_IMAGE} AS builder-base

ARG MAX_JOBS="16"

RUN ldconfig

RUN apt-get -qq update && \
    apt-get -qq install -y --no-install-recommends \
      python3-pip git ninja-build cmake && \
    apt-get clean && \
    pip3 install -U --no-cache-dir pip packaging setuptools wheel setuptools_scm regex

# Create the /wheels directory
WORKDIR /wheels

WORKDIR /workspace


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


FROM builder-base AS vllm-builder
RUN --mount=type=bind,from=vllm-downloader,source=/git/vllm,target=/workspace,rw \
    --mount=type=bind,from=freezer,target=/tmp/frozen,rw \
    /tmp/frozen/freeze.sh torch torchaudio torchvision xformers > /tmp/frozen/constraints.txt && \
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
      -c /tmp/frozen/constraints.txt \
      ./

WORKDIR /wheels


FROM builder-base AS flashinfer-builder
RUN --mount=type=bind,from=flashinfer-downloader,source=/git/flashinfer,target=/workspace,rw \
    --mount=type=bind,from=freezer,target=/tmp/frozen,rw \
    /tmp/frozen/freeze.sh torch torchaudio torchvision xformers > /tmp/frozen/constraints.txt && \
    export TORCH_CUDA_ARCH_LIST="$(echo "${TORCH_CUDA_ARCH_LIST}" | sed 's@[67]\.0 \+@@g')" && \
    python3 -m flashinfer.aot && \
    python3 -m pip wheel -w /wheels \
      -v --no-cache-dir --no-build-isolation --no-deps \
      -c /tmp/frozen/constraints.txt \
      ./

WORKDIR /wheels


FROM ${BASE_IMAGE} AS base

WORKDIR /workspace

RUN apt-get -qq update && apt-get install -y --no-install-recommends curl libsodium23 && apt-get clean

RUN --mount=type=bind,from=freezer,target=/tmp/frozen \
    /tmp/frozen/freeze.sh torch torchaudio torchvision xformers > /tmp/constraints.txt

RUN --mount=type=bind,from=vllm-builder,source=/wheels,target=/tmp/wheels \
    python3 -m pip install --no-cache-dir "$(printf '%s[tensorizer]' /tmp/wheels/*.whl)" -c /tmp/constraints.txt

RUN --mount=type=bind,from=flashinfer-builder,source=/wheels,target=/tmp/wheels \
    python3 -m pip install --no-cache-dir /tmp/wheels/*.whl -c /tmp/constraints.txt

# Copied from vLLM's Dockerfile
ARG TARGETPLATFORM

RUN if [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
        python3 -m pip install --no-cache-dir \
          accelerate hf_transfer 'modelscope!=1.15.0' 'bitsandbytes>=0.42.0' 'timm==0.9.10' \
          boto3 runai-model-streamer runai-model-streamer[s3] -c /tmp/constraints.txt; \
    else \
        python3 -m pip install --no-cache-dir \
          accelerate hf_transfer 'modelscope!=1.15.0' 'bitsandbytes>=0.45.3' 'timm==0.9.10' \
          boto3 runai-model-streamer runai-model-streamer[s3] -c /tmp/constraints.txt; \
    fi && \
    rm /tmp/constraints.txt

EXPOSE 8080
