# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE
ARG BUILDER_IMAGE="${BASE_IMAGE}"
ARG BUILD_CCACHE_SIZE="2Gi"

ARG SGLANG_COMMIT='v0.5.9'
ARG FLASHINFER_COMMIT='v0.6.3'
ARG CUTLASS_COMMIT='57e3cfb47a2d9e0d46eb6335c3dc411498efa198'
ARG VLLM_COMMIT='v0.18.0'
ARG XGRAMMAR_VERSION='v0.1.11'
ARG DECORD_COMMIT='d2e56190286ae394032a8141885f76d5372bd44b'

ARG BUILD_TORCH_CUDA_ARCH_LIST='8.9 9.0 10.0+PTX'

# region Downloads

FROM alpine/git:2.40.1 AS downloader-base
WORKDIR /git
RUN git config --global advice.detachedHead false

FROM downloader-base AS flashinfer-downloader
ARG FLASHINFER_COMMIT
RUN git clone --filter=tree:0 --no-single-branch --no-checkout \
      https://github.com/flashinfer-ai/flashinfer && \
    cd flashinfer && \
    git checkout "${FLASHINFER_COMMIT}" && \
    git submodule update --init --recursive --jobs 8 \
      --depth 1 --filter=tree:0

FROM downloader-base AS cutlass-downloader
ARG CUTLASS_COMMIT
RUN git clone --filter=tree:0 --no-single-branch --no-checkout \
      https://github.com/NVIDIA/cutlass && \
    cd cutlass && \
    git checkout "${CUTLASS_COMMIT}" && \
    git submodule update --init --recursive --jobs 8 \
      --depth 1 --filter=tree:0

FROM downloader-base AS vllm-downloader
ARG VLLM_COMMIT
RUN git clone --filter=tree:0 --no-single-branch --no-checkout \
      https://github.com/vllm-project/vllm && \
    cd vllm && \
    git checkout "${VLLM_COMMIT}" && \
    git submodule update --init --recursive --jobs 8 \
      --depth 1 --filter=tree:0

FROM downloader-base AS sglang-downloader
ARG SGLANG_COMMIT
RUN git clone --filter=tree:0 --no-single-branch --no-checkout \
      https://github.com/sgl-project/sglang && \
    cd sglang && \
    git checkout "${SGLANG_COMMIT}" && \
    git submodule update --init --recursive --jobs 8 \
      --depth 1 --filter=tree:0

FROM downloader-base AS xgrammar-downloader
ARG XGRAMMAR_VERSION
RUN git clone --filter=tree:0 --no-single-branch --no-checkout \
      https://github.com/mlc-ai/xgrammar && \
    cd xgrammar && \
    git checkout "${XGRAMMAR_VERSION}" && \
    git submodule update --init --recursive --jobs 8 \
      --depth 1 --filter=tree:0

FROM downloader-base AS decord-downloader
ARG DECORD_COMMIT
RUN git clone --filter=tree:0 --no-single-branch --no-checkout \
      https://github.com/dmlc/decord && \
    cd decord && \
    git checkout "${DECORD_COMMIT}" && \
    git submodule update --init --recursive --jobs 8 \
      --depth 1 --filter=tree:0

# endregion Downloads

# region Builder base

FROM scratch AS freezer
WORKDIR /
COPY --link --chmod=755 freeze.sh /

FROM ${BUILDER_IMAGE} AS builder-base

RUN ldconfig

RUN apt-get -qq update && \
    apt-get -qq install -y --no-install-recommends \
      python3-pip git ninja-build kmod && \
    apt-get clean && \
    pip3 install -U --no-cache-dir \
      pip packaging 'setuptools>=77.0.3,<81.0.0' wheel build pybind11 \
      'cmake==3.31.6'

ARG BUILD_CCACHE_SIZE
RUN mkdir /tmp/ccache-install && \
    cd /tmp/ccache-install && \
    CCACHE_URL='https://github.com/ccache/ccache/releases/download/v4.8.2/ccache-4.8.2.tar.xz' && \
    wget -qO - "$CCACHE_URL" | tar --strip-components 1 -xJf - && \
    mkdir build && \
    cd build && \
    cmake -B. -S.. -DCMAKE_BUILD_TYPE=Release && \
    cmake --build . --config Release && \
    make install && \
    cd ../.. && \
    rm -rf /tmp/ccache-install && \
    ccache -M "${BUILD_CCACHE_SIZE}" && \
    ccache -F 0

WORKDIR /wheels

ENV CCACHE_DIR=/ccache \
    CMAKE_C_COMPILER_LAUNCHER=ccache \
    CMAKE_CXX_COMPILER_LAUNCHER=ccache \
    CMAKE_CUDA_COMPILER_LAUNCHER=ccache

ARG BUILD_TORCH_CUDA_ARCH_LIST
ENV TORCH_CUDA_ARCH_LIST="${BUILD_TORCH_CUDA_ARCH_LIST}"

ENV NVCC_APPEND_FLAGS='-gencode=arch=compute_100,code=[sm_100,compute_100] -gencode=arch=compute_100a,code=sm_100a --diag-suppress 174'

RUN --mount=type=bind,from=freezer,target=/tmp/frozen \
    /tmp/frozen/freeze.sh torch torchaudio torchvision xformers > /opt/constraints.txt

WORKDIR /workspace

# endregion Builder base

# region Builders

FROM builder-base AS flashinfer-builder
ARG CUTLASS_COMMIT
RUN --mount=type=bind,from=flashinfer-downloader,source=/git/flashinfer,target=/workspace,rw \
    --mount=type=cache,target=/ccache \
    sed -i 's/name = "flashinfer-python"/name = "flashinfer"/' pyproject.toml && \
    git -C 3rdparty/cutlass checkout "${CUTLASS_COMMIT}" 2>/dev/null || true && \
    pip3 install --no-cache-dir -U optree && \
    NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:+$NVCC_APPEND_FLAGS } --diag-suppress 20281,174" \
    FLASHINFER_ENABLE_AOT=1 \
    PIP_CONSTRAINT=/opt/constraints.txt \
      python3 -m build -w -n -o /wheels .
WORKDIR /wheels


FROM builder-base AS vllm-builder
RUN --mount=type=bind,from=vllm-downloader,source=/git/vllm,target=/workspace,rw \
    --mount=type=bind,from=cutlass-downloader,source=/git/cutlass,target=/cutlass \
    --mount=type=cache,target=/ccache \
    if [ -f 'use_existing_torch.py' ]; then \
      python3 use_existing_torch.py; \
    else \
      git cat-file blob \
        e489ad7a210f4234db696d1f2749d5f3662fa65b:use_existing_torch.py \
        | python3 -; \
    fi && \
    pip3 install --no-cache-dir -r requirements-build.txt \
      -c /opt/constraints.txt && \
    VLLM_CUTLASS_SRC_DIR=/cutlass \
    USE_CUDNN=1 USE_CUSPARSELT=1 \
    PIP_CONSTRAINT=/opt/constraints.txt \
    PIP_BUILD_CONSTRAINT=/opt/constraints.txt \
      python3 -m build -w -n -o /wheels .
WORKDIR /wheels


FROM builder-base AS sglang-kernel-builder
ARG CUTLASS_COMMIT
SHELL ["/bin/bash", "-eo", "pipefail", "-c"]
RUN --mount=type=bind,from=sglang-downloader,source=/git/sglang,target=/workspace,rw \
    --mount=type=cache,target=/ccache \
    cd sgl-kernel && \
    git -C 3rdparty/cutlass checkout "${CUTLASS_COMMIT}" 2>/dev/null || true && \
    git -C 3rdparty/flashinfer/3rdparty/cutlass checkout "${CUTLASS_COMMIT}" 2>/dev/null || true && \
    ARCH_TRIPLE="$(gcc -print-multiarch)" && \
    LIB_DIR="/usr/lib/${ARCH_TRIPLE:?}" && \
    test -d "${LIB_DIR:?}" && \
    PYTHON_API_VER="$( \
      python3 --version | sed -En 's@Python ([0-9])\.([0-9]+)\..*@cp\1\2@p' \
    )" && \
    ARCH_FILTER=() && \
    if [ "$(uname -m)" = 'aarch64' ]; then \
      ARCH_FILTER=(-e 's@"-gencode=arch=compute_[78][0-9],code=sm_[78][0-9]",@#\0@'); \
    fi && \
    sed -Ei \
      "${ARCH_FILTER[@]}" \
      -e 's@/usr/lib/x86_64-linux-gnu@'"${LIB_DIR}"'@' \
      -e 's@(\s+)(\w.+manylinux2014_x86_64.+)@\1pass  # \2@' \
      -e 's@\{"py_limited_api": "cp39"}@{"py_limited_api": "'"${PYTHON_API_VER:-cp310}"'"}@' \
      setup.py && \
    SGL_KERNEL_ENABLE_BF16=1 \
    SGL_KERNEL_ENABLE_FP8=1 \
    SGL_KERNEL_ENABLE_SM90A=1 \
    PIP_CONSTRAINT=/opt/constraints.txt \
      python3 -m build -w -n -o /wheels .
WORKDIR /wheels


FROM builder-base AS sglang-python-builder
RUN --mount=type=bind,from=sglang-downloader,source=/git/sglang,target=/workspace,rw \
    cd python && \
    PIP_CONSTRAINT=/opt/constraints.txt \
      python3 -m build -w -n -o /wheels .
WORKDIR /wheels


FROM builder-base AS xgrammar-builder
ARG TARGETPLATFORM
RUN --mount=type=bind,from=xgrammar-downloader,source=/git/xgrammar,target=/workspace,rw \
    --mount=type=cache,target=/ccache \
    if [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
      mkdir xg-build && cd xg-build && \
      cmake -S/workspace -B. -DCMAKE_BUILD_TYPE=Release -GNinja && \
      cmake --build . && \
      cd /workspace && \
      python3 -m build -w -n -o /wheels python; \
    else \
      mkdir -p /wheels; \
    fi
WORKDIR /wheels


FROM builder-base AS decord-builder
ARG TARGETPLATFORM
RUN --mount=type=bind,from=decord-downloader,source=/git/decord,target=/workspace,rw \
    --mount=type=cache,target=/ccache \
    if [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
      apt-get -qq update && apt-get -q install --no-install-recommends -y \
        build-essential python3-dev python3-setuptools \
        make ffmpeg \
        libavcodec-dev libavfilter-dev libavformat-dev libavutil-dev && \
      apt-get clean && \
      mkdir dec-build && cd dec-build && \
      cmake -S/workspace -B. -DUSE_CUDA=0 -DCMAKE_BUILD_TYPE=Release -GNinja && \
      cmake --build . && \
      cp libdecord.so /wheels/libdecord.so && \
      cd /workspace/python && \
      python3 -m build -w -n -o /wheels .; \
    else \
      mkdir -p /wheels; \
    fi
WORKDIR /wheels

# endregion Builders

# region Final image

FROM ${BASE_IMAGE}

WORKDIR /workspace

RUN --mount=type=bind,from=freezer,target=/tmp/frozen \
    /tmp/frozen/freeze.sh torch torchaudio torchvision xformers > /tmp/constraints.txt

RUN --mount=type=bind,from=flashinfer-builder,source=/wheels,target=/tmp/wheels \
    python3 -m pip install --no-cache-dir /tmp/wheels/*.whl -c /tmp/constraints.txt

RUN --mount=type=bind,from=vllm-builder,source=/wheels,target=/tmp/wheels \
    python3 -m pip install --no-cache-dir /tmp/wheels/*.whl -c /tmp/constraints.txt

RUN --mount=type=bind,from=sglang-kernel-builder,source=/wheels,target=/tmp/wheels \
    python3 -m pip install --no-cache-dir /tmp/wheels/*.whl -c /tmp/constraints.txt

RUN --mount=type=bind,from=sglang-python-builder,source=/wheels,target=/tmp/wheels \
    python3 -m pip install --no-cache-dir /tmp/wheels/*.whl -c /tmp/constraints.txt

RUN --mount=type=bind,from=xgrammar-builder,source=/wheels,target=/tmp/wheels \
    if ls /tmp/wheels/*.whl 1>/dev/null 2>&1; then \
      python3 -m pip install --no-cache-dir /tmp/wheels/*.whl -c /tmp/constraints.txt; \
    fi

RUN --mount=type=bind,from=decord-builder,source=/wheels,target=/tmp/wheels \
    if ls /tmp/wheels/*.whl 1>/dev/null 2>&1; then \
      python3 -m pip install --no-cache-dir /tmp/wheels/*.whl -c /tmp/constraints.txt && \
      cp /tmp/wheels/libdecord.so /usr/local/lib/ 2>/dev/null && ldconfig; \
    fi

ARG TARGETPLATFORM
SHELL ["/bin/bash", "-eo", "pipefail", "-c"]
RUN SGLANG_EXTRA_PIP_DEPENDENCIES=() && \
    if [ "$TARGETPLATFORM" = "linux/amd64" ]; then \
      SGLANG_EXTRA_PIP_DEPENDENCIES=('decord' 'xgrammar>=0.1.10'); \
    fi && \
    python3 -m pip install --no-cache-dir \
      'aiohttp' 'fastapi' \
      'hf_transfer' 'huggingface_hub' 'interegular' 'modelscope' \
      'orjson' 'packaging' 'pillow' 'prometheus-client>=0.20.0' \
      'psutil' 'pydantic' 'python-multipart' 'pyzmq>=25.1.2' \
      'torchao>=0.7.0' 'uvicorn' 'uvloop' \
      'cuda-python' 'outlines>=0.0.44,<0.1.0' \
      "${SGLANG_EXTRA_PIP_DEPENDENCIES[@]}" \
      -c /tmp/constraints.txt && \
    rm /tmp/constraints.txt

EXPOSE 8080

# endregion Final image
