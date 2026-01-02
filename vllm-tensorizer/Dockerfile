ARG BUILDER_BASE_IMAGE="ghcr.io/coreweave/ml-containers/torch:19331b7-nccl-cuda12.9.1-ubuntu22.04-nccl2.29.2-1-torch2.9.1-vision0.24.1-audio2.9.1-abi1"
ARG FINAL_BASE_IMAGE="ghcr.io/coreweave/ml-containers/torch:19331b7-nccl-cuda12.9.1-ubuntu22.04-nccl2.29.2-1-torch2.9.1-vision0.24.1-audio2.9.1-abi1"

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
    pip3 install -U --no-cache-dir pip packaging setuptools wheel setuptools_scm regex build

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

RUN printf '%s\n' \
      '#!/bin/sh' \
      '. /opt/arch_flags.sh' \
      'export PIP_NO_CACHE_DIR=off PIP_CONSTRAINT=/opt/constraints.txt PIP_BUILD_CONSTRAINT=/opt/constraints.txt' \
      'exec python3 -m build -o /wheels -w -n "${@:-./}"' \
    | install -m 500 /dev/stdin /opt/build.sh

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
ARG LMCACHE_COMMIT='v0.3.9post2'
RUN git clone --filter=tree:0 --no-single-branch --no-checkout \
      https://github.com/LMCache/LMCache && \
    git -C LMCache checkout "${LMCACHE_COMMIT}"


FROM alpine/git:2.36.3 AS infinistore-downloader
WORKDIR /git
ARG INFINISTORE_COMMIT='553e36296ff1da2630636e7385b56224ff74a47c'
RUN git clone --filter=tree:0 --no-single-branch --no-checkout \
      https://github.com/bytedance/InfiniStore && \
    git -C InfiniStore checkout "${INFINISTORE_COMMIT}"

FROM alpine/git:2.36.3 AS deepgemm-downloader
WORKDIR /git
ARG DEEPGEMM_COMMIT='594953acce41793ae00a1233eb516044d604bcb6'
RUN git clone --filter=tree:0 --no-single-branch --no-checkout \
      https://github.com/deepseek-ai/DeepGEMM && \
    cd DeepGEMM && \
    git checkout "${DEEPGEMM_COMMIT}" && \
    git submodule update --init --recursive --jobs 8 \
      --depth 1 --filter=tree:0


FROM builder-base AS vllm-builder
RUN --mount=type=bind,from=vllm-downloader,source=/git/vllm,target=/workspace,rw \
    . /opt/arch_flags.sh && \
    if [ -z "$MAX_JOBS" ]; then unset MAX_JOBS; fi && \
    python3 -m pip install --no-cache-dir py-cpuinfo 'cmake>=3.26.1,<4' && \
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
    VLLM_MAIN_CUDA_VERSION="${CUDA_VERSION%.*}" \
      /opt/build.sh

WORKDIR /wheels


FROM builder-base AS flashinfer-builder
RUN --mount=type=bind,from=flashinfer-downloader,source=/git/flashinfer,target=/workspace,rw \
    . /opt/arch_flags.sh && \
    export TORCH_CUDA_ARCH_LIST="$(echo "${TORCH_CUDA_ARCH_LIST}" | sed 's@[67]\.0 \+@@g')" && \
    [ -n "${CUDA_VERSION}" ] && \
    python3 -m pip install --no-cache-dir \
      requests nvidia-ml-py ninja tqdm filelock \
      'nvidia-cudnn-frontend>=1.13.0' \
      "cuda-python~=${CUDA_VERSION}" \
      "nvidia-nvshmem-cu${CUDA_VERSION%%.*}" \
      'apache-tvm-ffi>=0.1,<0.2' && \
    export FLASHINFER_LOCAL_VERSION="$(sed -E 's@([[:digit:]]+)\.([[:digit:]]+).*$@cu\1\2@')" \
      FLASHINFER_AOT_USE_PY_LIMITED_API='0' \
      FLASHINFER_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" && \
    for DIR in . flashinfer-cubin flashinfer-jit-cache; do ( \
      cd "${DIR}" && \
      /opt/build.sh; \
    ); done

WORKDIR /wheels


FROM builder-base AS lmcache-builder
# LMCache must be built from source as it doesn't have pre-built ARM binaries
RUN --mount=type=bind,from=lmcache-downloader,source=/git/LMCache,target=/workspace,rw \
    . /opt/arch_flags.sh && \
    python3 -m pip install --no-cache-dir 'setuptools_scm>=8' && \
    sed -Ei \
      '/[ "]*(torch(vision|audio)?|xformers) *[<>=~]+/d' \
    pyproject.toml requirements/*.txt && \
    /opt/build.sh


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
    /opt/build.sh

FROM builder-base AS deepgemm-builder
RUN --mount=type=bind,from=deepgemm-downloader,source=/git/DeepGEMM,target=/workspace,rw \
    . /opt/arch_flags.sh && \
    /opt/build.sh

FROM builder-base AS nixl-builder
RUN apt-get -qq update && \
    apt-get -q install --no-install-recommends --no-upgrade -y \
      build-essential pkg-config && \
    apt-get clean && \
    python3 -m pip install --no-cache-dir meson ninja pybind11 cmake

ARG NIXL_TAG='0.2.0'
ARG NIXL_UCX_HOME='/opt/hpcx/ucx'

RUN mkdir /tmp/nixl && \
    cd /tmp/nixl && \
    wget "https://github.com/ai-dynamo/nixl/archive/refs/tags/${NIXL_TAG}.tar.gz" -qO- \
    | tar --strip-components=1 -xzf - && \
    [ -d "${NIXL_UCX_HOME}" ] && \
    CUDA_TARGET="$(find /usr/local/cuda/targets -mindepth 1 -maxdepth 1 -type d | head -1)" && \
    meson setup build --prefix=/opt/nixl \
      -Ducx_path="${NIXL_UCX_HOME}" \
      -Dgds_path="${CUDA_TARGET}" \
      -Dcudapath_inc="${CUDA_TARGET}/include" \
      -Dcudapath_lib="${CUDA_TARGET}/lib" \
      -Dcudapath_stub="${CUDA_TARGET}/lib/stubs" && \
    cd build && \
    ninja && \
    ninja install && \
    cd ../.. && \
    rm -r /tmp/nixl && \
    echo '/opt/nixl/lib/python3/dist-packages' > /usr/lib/python3/dist-packages/nixl.pth

FROM ${FINAL_BASE_IMAGE} AS base

WORKDIR /workspace

RUN apt-get -qq update && apt-get install -y --no-install-recommends curl libsodium23 libnuma-dev && apt-get clean

RUN --mount=type=bind,from=freezer,target=/tmp/frozen \
    /tmp/frozen/freeze.sh torch torchaudio torchvision xformers > /tmp/constraints.txt && \
    echo 'click != 8.3.0' >> /tmp/constraints.txt

RUN --mount=type=bind,from=vllm-builder,source=/wheels,target=/tmp/wheels \
    python3 -m pip install --no-cache-dir "$(printf '%s[tensorizer]' /tmp/wheels/*.whl)" -c /tmp/constraints.txt

RUN --mount=type=bind,from=flashinfer-builder,source=/wheels,target=/tmp/wheels \
    python3 -m pip install --no-cache-dir /tmp/wheels/*.whl -c /tmp/constraints.txt && \
    python3 -m pip uninstall -y pynvml && \
    python3 -m pip install --no-cache-dir nvidia-ml-py

# InfiniStore must be installed before LMCache as LMCache depends on InfiniStore
RUN --mount=type=bind,from=infinistore-builder,source=/wheels,target=/tmp/wheels \
    python3 -m pip install --no-cache-dir /tmp/wheels/*.whl -c /tmp/constraints.txt

RUN --mount=type=bind,from=lmcache-builder,source=/wheels,target=/tmp/wheels \
    python3 -m pip install --no-cache-dir /tmp/wheels/*.whl -c /tmp/constraints.txt

RUN --mount=type=bind,from=deepgemm-builder,source=/wheels,target=/tmp/wheels \
    python3 -m pip install --no-cache-dir /tmp/wheels/*.whl -c /tmp/constraints.txt

COPY --link --from=nixl-builder /opt/nixl /opt/nixl
COPY --link --from=nixl-builder /usr/lib/python3/dist-packages/nixl.pth /usr/lib/python3/dist-packages/nixl.pth

# Copied from vLLM's Dockerfile
ARG TARGETPLATFORM

RUN if [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
      BITSANDBYTES_VER='0.42.0'; \
    else \
      BITSANDBYTES_VER='0.46.1'; \
    fi && \
    python3 -m pip install --no-cache-dir \
      accelerate hf_transfer 'modelscope!=1.15.0' "bitsandbytes>=${BITSANDBYTES_VER:?}" 'timm>=1.0.17' \
      'runai-model-streamer[s3,gcs]>=0.15.0' -c /tmp/constraints.txt && \
    rm /tmp/constraints.txt

EXPOSE 8080
