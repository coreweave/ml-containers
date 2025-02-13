ARG BASE_IMAGE="nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04"
ARG FINAL_BASE_IMAGE="ghcr.io/coreweave/ml-containers/torch:es-actions-7095c59-base-cuda12.4.1-ubuntu22.04-torch2.5.1-vision0.20.0-audio2.5.0"

FROM scratch as freezer
WORKDIR /
COPY --chmod=755 freeze.sh /

FROM ${BASE_IMAGE} as builder-base
RUN apt-get -qq update && \
    apt-get -qq install -y --no-install-recommends python3.10 && \
    ln -s /usr/bin/python3.10 /usr/bin/python3

RUN python3 -m ensurepip --upgrade

RUN python3 -m pip list
RUN python3 --version

FROM builder-base as vllm-downloader
ARG COMMIT_HASH

RUN ldconfig

RUN apt-get -qq update && \
    apt-get -qq install -y --no-install-recommends git ninja-build && \
    apt-get clean && \
    pip3 install -U --no-cache-dir pip packaging 'setuptools<70.0.0' wheel cmake setuptools setuptools_scm

RUN git clone --filter=blob:none --depth 1 --no-single-branch --no-checkout \
      https://github.com/vllm-project/vllm.git && \
    cd vllm && \
    python3 use_existing_torch.py && \
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
    python3 -m pip wheel -w /wheels \
      -v --no-cache-dir --no-build-isolation --no-deps \
      -c /tmp/constraints.txt \
      ./ && \
    pip3 uninstall -y vllm-flash-attn

WORKDIR /wheels

FROM ${FINAL_BASE_IMAGE}

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
