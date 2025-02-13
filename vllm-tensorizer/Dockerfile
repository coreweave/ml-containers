ARG BASE_IMAGE="ghcr.io/coreweave/ml-containers/torch-extras:es-actions-386fabe-nccl-cuda12.6.3-ubuntu22.04-nccl2.25.1-1-torch2.6.0-vision0.21.0-audio2.6.0-abi1"

FROM scratch as freezer
WORKDIR /
COPY --chmod=755 freeze.sh /

FROM ${BASE_IMAGE} as builder-base


# Dependencies requiring NVCC are built ahead of time in a separate stage
# so that the ~2 GiB dev library installations don't have to be included
# in the final image.
RUN export \
      CUDA_MAJOR_VERSION=$(echo $CUDA_VERSION | cut -d. -f1) \
      CUDA_MINOR_VERSION=$(echo $CUDA_VERSION | cut -d. -f2) && \
    export \
      CUDA_PACKAGE_VERSION="${CUDA_MAJOR_VERSION}-${CUDA_MINOR_VERSION}" && \
    apt-get -qq update && apt-get install -y --no-install-recommends \
      cuda-nvcc-${CUDA_PACKAGE_VERSION} \
      cuda-nvml-dev-${CUDA_PACKAGE_VERSION} \
      libcurand-dev-${CUDA_PACKAGE_VERSION} \
      libcublas-dev-${CUDA_PACKAGE_VERSION} \
      libcusparse-dev-${CUDA_PACKAGE_VERSION} \
      libcusolver-dev-${CUDA_PACKAGE_VERSION} \
      cuda-profiler-api-${CUDA_PACKAGE_VERSION} \
      cuda-nvtx-${CUDA_PACKAGE_VERSION} \
      libaio-dev \
      ninja-build \
      kmod && \
    apt-get clean && \
    ldconfig

RUN apt-get -qq update && \
    apt-get -qq install -y --no-install-recommends \
      python3-pip git ninja-build && \
    apt-get clean && \
    pip3 install -U --no-cache-dir pip packaging 'setuptools<70.0.0' wheel cmake setuptools_scm

FROM alpine/git:2.36.3 as vllm-downloader
WORKDIR /git
ARG COMMIT_HASH
RUN git clone --filter=blob:none --depth 1 --no-single-branch --no-checkout \
      https://github.com/vllm-project/vllm.git && \
    cd vllm && \
    git checkout "${COMMIT_HASH}" && \
    git submodule update --init --recursive --jobs 8 \
      --depth 1 --filter=blob:none

FROM alpine/git:2.36.3 as vllm-flash-attn-downloader
WORKDIR /git
ARG VLLM_FLASH_ATTN_VERSION
RUN git clone --filter=blob:none --depth 1 --no-single-branch --no-checkout \
      https://github.com/vllm-project/flash-attention.git && \
    cd flash-attention && \
    git checkout "${VLLM_FLASH_ATTN_VERSION}" && \
    git submodule update --init --recursive --jobs 8 \
      --depth 1 --filter=blob:none

FROM builder-base as vllm-builder
WORKDIR /workspace

ENV LIBRARY_PATH="/usr/local/cuda/lib64/stubs${LIBRARY_PATH:+:$LIBRARY_PATH}"

RUN --mount=type=bind,from=freezer,target=/tmp/frozen,rw \
    /tmp/frozen/freeze.sh torch torchaudio torchvision xformers > /tmp/constraints.txt

RUN --mount=type=bind,from=vllm-flash-attn-downloader,source=/git/flash-attention,target=/workspace,rw \
    python3 -m pip wheel -w /wheels \
      -v --no-cache-dir --no-build-isolation --no-deps \
      -c /tmp/constraints.txt \
      ./

RUN --mount=type=bind,from=vllm-downloader,source=/git/vllm,target=/workspace,rw \
    pip3 install /wheels/*.whl && \
    python3 use_existing_torch.py && \
    USE_CUDNN=1 USE_CUSPARSELT=1 python3 -m pip wheel -w /wheels \
      -v --no-cache-dir --no-build-isolation --no-deps \
      -c /tmp/constraints.txt \
      ./ && \
    pip3 uninstall -y vllm-flash-attn

WORKDIR /wheels

FROM ${BASE_IMAGE} as base

WORKDIR /workspace

RUN apt-get -qq update && apt-get install -y --no-install-recommends curl && apt-get clean

RUN --mount=type=bind,from=freezer,target=/tmp/frozen \
    /tmp/frozen/freeze.sh torch torchaudio torchvision xformers > /tmp/constraints.txt

RUN --mount=type=bind,from=vllm-builder,source=/wheels,target=/tmp/wheels \
    python3 -m pip install --no-cache-dir /tmp/wheels/*.whl -c /tmp/constraints.txt && \
    rm /tmp/constraints.txt

# Make sure xformers imports properly (for vLLM)
RUN python3 -c "from xformers import ops as xops"

EXPOSE 8080
