ARG BASE_IMAGE="ghcr.io/coreweave/ml-containers/torch-extras:696032c-base-cuda12.2.2-ubuntu22.04-torch2.3.0-vision0.18.0-audio2.3.0"

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
      cuda-nvprof-${CUDA_PACKAGE_VERSION} \
      cuda-profiler-api-${CUDA_PACKAGE_VERSION} \
      libaio-dev \
      ninja-build && \
    apt-get clean

RUN ldconfig

RUN apt-get -qq update && \
    apt-get -qq install -y --no-install-recommends \
      python3-pip git ninja-build && \
    apt-get clean && \
    pip3 install -U --no-cache-dir pip packaging 'setuptools<70.0.0' wheel cmake

FROM alpine/git:2.36.3 as vllm-downloader
WORKDIR /git
ARG COMMIT_HASH
RUN git clone --filter=blob:none --depth 1 --no-single-branch --no-checkout \
      https://github.com/vllm-project/vllm.git && \
    cd vllm && \
    git checkout "${COMMIT_HASH}" && \
    git submodule update --init --recursive --jobs 8 \
      --depth 1 --filter=blob:none

FROM builder-base as vllm-builder
WORKDIR /workspace
RUN --mount=type=bind,from=vllm-downloader,source=/git/vllm,target=/workspace,rw \
    --mount=type=bind,from=freezer,target=/tmp/frozen,rw \
    /tmp/frozen/freeze.sh torch torchaudio torchvision xformers > /tmp/frozen/constraints.txt && \
    LIBRARY_PATH="/usr/local/cuda/lib64/stubs${LIBRARY_PATH:+:$LIBRARY_PATH}" \
      python3 -m pip wheel -w /wheels \
      -v --no-cache-dir --no-build-isolation --no-deps \
      -c /tmp/frozen/constraints.txt \
      ./

WORKDIR /wheels

FROM ${BASE_IMAGE} as base

WORKDIR /workspace

RUN apt-get -qq update && apt-get install -y --no-install-recommends curl && apt-get clean

RUN --mount=type=bind,from=freezer,target=/tmp/frozen \
    /tmp/frozen/freeze.sh torch torchaudio torchvision xformers > /tmp/constraints.txt

RUN python3 -m pip install --no-cache-dir \
      "fschat[model_worker] == 0.2.30" "triton == 2.1.0" \
      -c /tmp/constraints.txt

RUN --mount=type=bind,from=vllm-builder,source=/wheels,target=/tmp/wheels \
    python3 -m pip install --no-cache-dir /tmp/wheels/*.whl -c /tmp/constraints.txt && \
    rm /tmp/constraints.txt


EXPOSE 8080