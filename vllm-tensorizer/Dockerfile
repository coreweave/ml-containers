ARG BASE_IMAGE="ghcr.io/coreweave/ml-containers/torch-extras:es-22.04-58a49a2-base-cuda12.1.1-torch2.1.2-vision0.16.2-audio2.1.2-flash_attn2.4.2"
FROM ${BASE_IMAGE} as builder-base

ARG MAX_JOBS=""

COPY --chmod=755 freeze.sh /tmp
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

RUN apt update && \
    apt install -y python3-pip python3-packaging \
      git ninja-build && \
    pip3 install -U --no-cache-dir pip packaging setuptools wheel \

FROM alpine/git:2.36.3 as vllm-downloader
WORKDIR /git
ARG COMMIT_HASH
RUN git clone --filter=blob:none --depth 1 --no-single-branch --no-checkout \
      https://github.com/coreweave/vllm.git && \
    cd vllm && \
    git checkout "${COMMIT_HASH}" && \
    git submodule update --init --recursive --jobs 8 \
      --depth 1 --filter=blob:none

FROM builder-base as vllm-builder
WORKDIR /workspace
COPY --from=vllm-downloader /git/vllm /workspace
RUN ./tmp/freeze.sh torch torchaudio torchvision transformers xformers > constraints.txt && \
    python3 -m pip wheel -w /wheels -v --no-cache-dir --no-build-isolation -c constraints.txt --no-deps ./

WORKDIR /wheels

FROM ${BASE_IMAGE} as base

WORKDIR /workspace

RUN apt-get -qq update && apt-get install -y --no-install-recommends curl && apt-get clean

RUN ./tmp/freeze.sh torch torchaudio torchvision xformers > constraints.txt && \
    pip3 install "transformers == 4.36.0" "fschat[model_worker] == 0.2.30" "triton == 2.1.0" -c constraints.txt

# Install the vllm wheel from the vllm-builder layer
RUN --mount=type=bind,from=vllm-builder,source=/wheels,target=/tmp/wheels \
    ./tmp/freeze.sh torch torchaudio torchvision xformers > constraints.txt && \
    python3 -m pip install --no-cache-dir /tmp/wheels/*.whl -c constraints.txt && \
    rm constraints.txt

EXPOSE 8080

ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
