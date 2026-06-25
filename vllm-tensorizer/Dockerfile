# syntax=docker/dockerfile:1.10
ARG BUILDER_BASE_IMAGE="ghcr.io/coreweave/ml-containers/torch:bc8c66e-nccl-cuda13.2.1-ubuntu24.04-nccl2.30.4-1-torch2.11.0-vision0.26.0-audio2.11.0-abi1"
ARG FINAL_BASE_IMAGE="ghcr.io/coreweave/ml-containers/torch:bc8c66e-nccl-cuda13.2.1-ubuntu24.04-nccl2.30.4-1-torch2.11.0-vision0.26.0-audio2.11.0-abi1"
ARG SCCACHE_VERSION="0.14.0"

FROM alpine/curl:8.17.0 AS sccache-downloader
ARG SCCACHE_VERSION
RUN ARCH=$(uname -m) && \
    curl -fsSL "https://github.com/mozilla/sccache/releases/download/v${SCCACHE_VERSION}/sccache-v${SCCACHE_VERSION}-${ARCH}-unknown-linux-musl.tar.gz" \
    | tar xz --strip-components=1 -C /opt/ "sccache-v${SCCACHE_VERSION}-${ARCH}-unknown-linux-musl/sccache" && \
    chmod 755 /opt/sccache

FROM scratch AS freezer
WORKDIR /
COPY --chmod=755 freeze.sh /

FROM ${BUILDER_BASE_IMAGE} AS builder-base

ARG MAX_JOBS="32"

RUN ldconfig

RUN apt-get -qq update && \
    apt-get -qq install -y --no-install-recommends \
      git ninja-build cmake gcc-12 g++-12 && \
    apt-get clean && \
    python3 -m pip install -U --no-cache-dir pip packaging 'setuptools>=77.0.3,<81.0.0' wheel setuptools_scm regex build

# Create the /wheels directory
WORKDIR /wheels

WORKDIR /workspace

RUN --mount=type=bind,from=freezer,target=/tmp/frozen \
    /tmp/frozen/freeze.sh torch torchaudio torchvision xformers > /opt/constraints.txt

COPY --link --from=common --chmod=755 nvcc-wrapper.py /opt/nvcc-wrapper.py
ENV PYTORCH_NVCC='/opt/nvcc-wrapper.py' \
    CUDACXX='/opt/nvcc-wrapper.py'

# Setup for sccache and its wrappers.
COPY --link --from=sccache-downloader /opt/sccache /opt/sccache
COPY --link --from=common --chmod=755 sccache*.sh /opt/
COPY --link --from=common sccache.toml /etc/sccache.toml
RUN sed -Ei 's@^(key_prefix.*)misc@\1vllm-tensorizer@' /etc/sccache.toml
ENV SCCACHE_CONF=/etc/sccache.toml
ENV CC=/opt/sccache-cc.sh \
    CXX=/opt/sccache-c++.sh
ENV TORCH_EXTENSION_SKIP_NVCC_GEN_DEPENDENCIES=1

ARG TARGETPLATFORM
# Switch 9.x, 10.x, and 12.x to -a variants; inject 10.3a if absent.
# Flashinfer v0.28.0 in particular can only build for 12.0a but not 12.0
# vLLM Marlin MoE requires 8.0+PTX for non-FP8 WNA16 kernels. Keep the arm64
# filter strict about unsupported sm80 SASS, but preserve compute80 PTX so
# newer architectures can forward-JIT the generated sm80 kernels.
RUN printf 'TORCH_CUDA_ARCH_LIST=' && \
    echo "${TORCH_CUDA_ARCH_LIST}" \
    | sed -E 's@\b(9|10|12)\.([0-9]+)\b@\1.\2a@g; /\b10\.3a\b/!s@\b10\.0a\b@10.0a 10.3a@; s@\+PTX\b@@g' \
    | tee /opt/arch_list.txt && \
    printf 'NVCC_WRAPPER_FILTER_CODES=' && \
    if [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
      echo 'sm_80;sm_89;sm_90;sm_100;sm_103;sm_120;compute_89;compute_90;compute_100;compute_103;compute_120'; \
    else \
      echo 'sm_90;sm_100;sm_103;sm_120;compute_90;compute_100;compute_103;compute_120'; \
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
COPY patches/flashinfer-trtllm-ragged-empty-kv.patch /tmp/flashinfer-trtllm-ragged-empty-kv.patch
RUN git clone --filter=tree:0 --no-single-branch --no-checkout \
      https://github.com/flashinfer-ai/flashinfer && \
    cd flashinfer && \
    git checkout "${FLASHINFER_COMMIT}" && \
    git apply --unidiff-zero -v --stat --apply /tmp/flashinfer-trtllm-ragged-empty-kv.patch && \
    git submodule update --init --recursive --jobs 8 \
      --depth 1 --filter=tree:0


FROM alpine/git:2.36.3 AS lmcache-downloader
WORKDIR /git
ARG LMCACHE_COMMIT
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
ARG DEEPGEMM_COMMIT='891d57b4db1071624b5c8fa0d1e51cb317fa709f'
RUN git clone --filter=tree:0 --no-single-branch --no-checkout \
      https://github.com/deepseek-ai/DeepGEMM && \
    cd DeepGEMM && \
    git checkout "${DEEPGEMM_COMMIT}" && \
    git submodule update --init --recursive --jobs 8 \
      --depth 1 --filter=tree:0


# vLLM v0.22.0+ ships a Rust frontend (`vllm-rs`) whose build requires the
# rust toolchain pinned by rust-toolchain.toml (edition 2024, channel 1.95)
# plus protoc. Build it in a dedicated stage and copy the single binary into
# the wheel build below, mirroring upstream's docker/Dockerfile so the heavy
# wheel stage doesn't need cargo. setup.py detects the precompiled binary and
# skips the local cargo build.
FROM builder-base AS vllm-rust-builder
ENV CARGO_BUILD_JOBS=4 \
    CC=gcc CXX=g++ \
    PATH="/root/.cargo/bin:${PATH}"
RUN apt-get -qq update && \
    apt-get -q install --no-install-recommends --no-upgrade -y \
      curl ca-certificates unzip build-essential pkg-config perl && \
    apt-get clean && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal --default-toolchain none
RUN --mount=type=bind,from=vllm-downloader,source=/git/vllm,target=/workspace,rw \
    --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/root/.cargo/git \
    --mount=type=cache,target=/workspace/rust/target \
    bash tools/install_protoc.sh && \
    VLLM_RS_TARGET_PATH=/tmp/vllm-rs bash build_rust.sh && \
    install -D -m755 /tmp/vllm-rs /out/vllm-rs


FROM builder-base AS vllm-builder
RUN --mount=type=bind,from=vllm-downloader,source=/git/vllm,target=/workspace,rw \
    --mount=type=bind,from=vllm-rust-builder,source=/out/vllm-rs,target=/opt/vllm-rs \
    --mount=type=secret,id=s3_access_key_id,env=AWS_ACCESS_KEY_ID \
    --mount=type=secret,id=s3_secret_access_key,env=AWS_SECRET_ACCESS_KEY \
    --mount=type=tmpfs,target=/sccache \
    --mount=type=tmpfs,target=/tmp \
    . /opt/sccache-start.sh && \
    . /opt/arch_flags.sh && \
    export NVCC_APPEND_FLAGS='-gencode=arch=compute_103a,code=sm_103a' && \
    export CMAKE_ARGS='-DCMAKE_CUDA_COMPILER=/opt/nvcc-wrapper.py' && \
    { \
      sed -i -E 's@(CUDA_SUPPORTED_ARCHS +"[^"]*;10\.0;)(11\.0;)@\110.3;\2@' CMakeLists.txt && \
      grep -qE 'CUDA_SUPPORTED_ARCHS +"[^"]*;10\.0;10\.3;11\.0;' CMakeLists.txt \
      || { echo 'Failed to apply sed patch' >&2; exit 1; }; \
    } && \
    if [ -z "$MAX_JOBS" ]; then unset MAX_JOBS; fi && \
    python3 -m pip install --no-cache-dir py-cpuinfo 'cmake>=3.26.1,<4' \
      grpcio-tools 'setuptools-rust>=1.9.0' && \
    if [ -f 'use_existing_torch.py' ]; then \
      python3 use_existing_torch.py; \
    else \
      git cat-file blob \
        e489ad7a210f4234db696d1f2749d5f3662fa65b:use_existing_torch.py \
        | python3 -; \
    fi && \
    install -D -m755 /opt/vllm-rs vllm/vllm-rs && \
    LIBRARY_PATH="/usr/local/cuda/lib64:${LIBRARY_PATH:+:$LIBRARY_PATH}" \
    CUDA_TOOLKIT_ROOT_DIR="/usr/local/cuda" \
    VLLM_MAIN_CUDA_VERSION="${CUDA_VERSION%.*}" \
      /opt/build.sh

WORKDIR /wheels


FROM builder-base AS flashinfer-builder
RUN --mount=type=bind,from=flashinfer-downloader,source=/git/flashinfer,target=/workspace,rw \
    --mount=type=secret,id=s3_access_key_id,env=AWS_ACCESS_KEY_ID \
    --mount=type=secret,id=s3_secret_access_key,env=AWS_SECRET_ACCESS_KEY \
    --mount=type=tmpfs,target=/sccache \
    --mount=type=tmpfs,target=/tmp \
    . /opt/sccache-start.sh && \
    . /opt/arch_flags.sh && \
    export TORCH_CUDA_ARCH_LIST="$(echo "${TORCH_CUDA_ARCH_LIST}" | sed 's@[67]\.0 \+@@g')" && \
    [ -n "${CUDA_VERSION}" ] && \
    python3 -m pip install --no-cache-dir \
      requests nvidia-ml-py ninja tqdm filelock \
      'nvidia-cudnn-frontend>=1.19.1' \
      "cuda-python~=${CUDA_VERSION%.*}" \
      "nvidia-nvshmem-cu${CUDA_VERSION%%.*}<3.6" \
      'apache-tvm-ffi==0.1.9' && \
    export FLASHINFER_LOCAL_VERSION="$(sed -E 's@([[:digit:]]+)\.([[:digit:]]+).*$@cu\1\2@')" \
      FLASHINFER_AOT_USE_PY_LIMITED_API='0' \
      FLASHINFER_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" && \
    for DIR in . flashinfer-cubin; do ( \
      cd "${DIR}" && \
      /opt/build.sh; \
    ); done && \
    ( cd flashinfer-jit-cache && CXX_WRAPPED=g++-12 CC_WRAPPED=gcc-12 /opt/build.sh )

WORKDIR /wheels


FROM builder-base AS lmcache-builder
# LMCache must be built from source as it doesn't have pre-built ARM binaries
RUN --mount=type=bind,from=lmcache-downloader,source=/git/LMCache,target=/workspace,rw \
    --mount=type=secret,id=s3_access_key_id,env=AWS_ACCESS_KEY_ID \
    --mount=type=secret,id=s3_secret_access_key,env=AWS_SECRET_ACCESS_KEY \
    --mount=type=tmpfs,target=/sccache \
    --mount=type=tmpfs,target=/tmp \
    . /opt/sccache-start.sh && \
    . /opt/arch_flags.sh && \
    python3 -m pip install --no-cache-dir \
    'setuptools>=77.0.3,<81.0.0' \
    'setuptools_scm>=8' && \
    sed -Ei \
      '/[ "]*(torch(vision|audio)?|xformers) *[<>=~]+/d' \
    pyproject.toml requirements/*.txt && \
    /opt/build.sh


FROM builder-base AS infinistore-builder
# InfiniStore is required when installing LMCache
# It must also be built from source as it also doesn't have pre-built ARM binaries
RUN --mount=type=bind,from=infinistore-downloader,source=/git/InfiniStore,target=/workspace,rw \
    --mount=type=secret,id=s3_access_key_id,env=AWS_ACCESS_KEY_ID \
    --mount=type=secret,id=s3_secret_access_key,env=AWS_SECRET_ACCESS_KEY \
    --mount=type=tmpfs,target=/sccache \
    --mount=type=tmpfs,target=/tmp \
    . /opt/sccache-start.sh && \
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
    --mount=type=secret,id=s3_access_key_id,env=AWS_ACCESS_KEY_ID \
    --mount=type=secret,id=s3_secret_access_key,env=AWS_SECRET_ACCESS_KEY \
    --mount=type=tmpfs,target=/sccache \
    --mount=type=tmpfs,target=/tmp \
    . /opt/sccache-start.sh && \
    . /opt/arch_flags.sh && \
    /opt/build.sh

FROM builder-base AS deepep-builder
ARG DEEPEP_COMMIT='73b6ea4'
ARG NVSHMEM_VER='3.3.24'
# DeepEP requires SM 9.0+ (Hopper/Blackwell). Filter out pre-SM90 arches
# so Blackwell (10.0) is also supported.
RUN --mount=type=bind,from=vllm-downloader,source=/git/vllm/tools/ep_kernels,target=/tmp/ep_kernels \
    python3 -m pip install --no-cache-dir uv && \
    . /opt/arch_flags.sh && \
    export NVCC_WRAPPER_FILTER_CODES="sm_80;sm_89;compute_80;compute_89;${NVCC_WRAPPER_FILTER_CODES}" && \
    WORKSPACE=/tmp/deepep_build MODE=wheel \
    DEEPEP_COMMIT_HASH="${DEEPEP_COMMIT}" \
    NVSHMEM_VER="${NVSHMEM_VER}" \
    bash /tmp/ep_kernels/install_python_libraries.sh && \
    cp /tmp/deepep_build/dist/*.whl /wheels/ && \
    mkdir -p /nvshmem_libs && \
    cp /tmp/deepep_build/nvshmem/lib/libnvshmem*.so* /nvshmem_libs/

# Build a NIXL wheel linked against HPC-X's multi-threaded UCX (/opt/hpcx/ucx/mt). The PyPI
# nixl-cuNN wheels bundle their own UCX, which clashes with this image's HPC-X UCX and can't
# register GPU memory. UCX+POSIX are the only plugins needed; CUDA arch stays at NIXL's default
# (sm_80+) to keep the wheel multi-arch.
FROM builder-base AS nixl-builder
RUN apt-get -qq update && \
    apt-get -q install --no-install-recommends --no-upgrade -y \
      build-essential pkg-config git && \
    apt-get clean && \
    python3 -m pip install --no-cache-dir meson meson-python ninja pybind11 tomlkit patchelf

ARG NIXL_TAG='v1.2.0'
ARG NIXL_UCX_HOME='/opt/hpcx/ucx/mt'

RUN --mount=type=secret,id=s3_access_key_id,env=AWS_ACCESS_KEY_ID \
    --mount=type=secret,id=s3_secret_access_key,env=AWS_SECRET_ACCESS_KEY \
    --mount=type=tmpfs,target=/sccache \
    --mount=type=tmpfs,target=/tmp \
    . /opt/sccache-start.sh && \
    [ -d "${NIXL_UCX_HOME}" ] && \
    git clone --depth 1 --branch "${NIXL_TAG}" https://github.com/ai-dynamo/nixl /tmp/nixl && \
    # Rename the dist to nixl-cu<major> matching this image's CUDA (NIXL hardcodes nixl-cu12), so the
    # `nixl` meta's nixl-cu<major> pin is met by our build and no prebuilt (bundled-UCX) wheel is pulled.
    CUDA_MAJOR="$(nvcc --version | sed -n 's/.*release \([0-9]\{1,\}\).*/\1/p' | head -1)" && \
    sed -i -E "s/^name = \"nixl-cu[0-9]+\"/name = \"nixl-cu${CUDA_MAJOR}\"/" /tmp/nixl/pyproject.toml && \
    python3 -m pip wheel --no-cache-dir --no-build-isolation --no-deps /tmp/nixl -w /wheels \
      -C setup-args="-Ducx_path=${NIXL_UCX_HOME}" \
      -C setup-args="-Denable_plugins=UCX,POSIX" && \
    rm -rf /tmp/nixl

FROM ${FINAL_BASE_IMAGE} AS base

WORKDIR /workspace

RUN apt-get -qq update && \
    apt-get install -y --no-install-recommends curl libsodium23 libnuma-dev && \
    apt-get purge -y python3-jwt && \
    apt-get clean

# vLLM 0.23 still pins nvidia-cutlass-dsl==4.5.2 (which renamed cutlass.base_dsl.Arch
# -> arch) but only floors quack-kernels (>=0.3.3). A drifted quack still imports
# the old `Arch` and crashes DeepSeek-V4's cute-DSL indexer at load. cutlass-dsl is
# unchanged from 0.22.0, so keep quack pinned to the same known-good 0.4.1 release.
# Revisit on the next vLLM bump.
RUN --mount=type=bind,from=freezer,target=/tmp/frozen \
    /tmp/frozen/freeze.sh torch torchaudio torchvision xformers > /tmp/constraints.txt && \
    echo 'click != 8.3.0' >> /tmp/constraints.txt && \
    echo 'quack-kernels==0.4.1' >> /tmp/constraints.txt

RUN --mount=type=bind,from=vllm-builder,source=/wheels,target=/tmp/wheels \
    python3 -m pip install --no-cache-dir "$(printf '%s[tensorizer,audio]' /tmp/wheels/*.whl)" -c /tmp/constraints.txt

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

RUN --mount=type=bind,from=deepep-builder,source=/wheels,target=/tmp/wheels \
    python3 -m pip install --no-cache-dir /tmp/wheels/*.whl -c /tmp/constraints.txt

COPY --link --from=deepep-builder /nvshmem_libs/ /usr/local/lib/
RUN ldconfig

# Copied from vLLM's Dockerfile
ARG TARGETPLATFORM

# Ray was removed as a default dependancy by vllm in v18, but it is still required
# for the multi-node setup.

# * Gemma 4 MTP needs transformers 5.9.0
# * Kimi K2.5 tool-call scramble fixed in transformers 5.5.4 (huggingface/transformers#45359)
# * Kimi K2.5 registry-subprocess SIGSEGV on GB200 with nixl-cu12 1.0.1 (vllm-project/vllm#40642) - verified 
#   no crash on this image with vllm 0.22
# * llguidance bumped to 1.7 — vLLM's pin (<1.4) lacks JSON Schema "format": "uri" support
#   (added in llguidance v1.6.0; mirrors upstream vllm-project/vllm#42150)

# ray[cgraph]>=2.48.0 will unconditionally install cupy-cuda12x. Uninstall it and install cupy-cuda13x if CUDA major version is 13.

RUN if [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
      BITSANDBYTES_VER='0.42.0'; \
    else \
      BITSANDBYTES_VER='0.46.1'; \
    fi && \
    python3 -m pip install --no-cache-dir \
      accelerate hf_transfer 'modelscope!=1.15.0' "bitsandbytes>=${BITSANDBYTES_VER:?}" 'timm>=1.0.17' \
      'runai-model-streamer[s3,gcs]>=0.15.3' "ray[cgraph]>=2.48.0" \
      'transformers==5.9.0' \
      'llguidance>=1.7.0,<1.8.0' \
      -c /tmp/constraints.txt && \
    rm /tmp/constraints.txt && \
    if [ "$(echo $CUDA_VERSION | cut -d. -f1)" -eq "13" ]; then \
      python3 -m pip uninstall -y cupy-cuda12x && \
      python3 -m pip install --no-cache-dir --force-reinstall --no-deps cupy-cuda13x; \
    fi

# Swap the PyPI nixl for our MT-UCX build. Runs last because earlier installs (e.g. LMCache's
# `nixl>=1.1.0`) pull the bundled-UCX wheel; purge those, install our wheel + the `nixl` meta
# (the pure-python dispatcher) --no-deps so no bundled-UCX wheel comes back.
RUN --mount=type=bind,from=nixl-builder,source=/wheels,target=/tmp/wheels \
    python3 -m pip uninstall -y nixl nixl-cu12 nixl-cu13 2>/dev/null; \
    rm -rf /usr/local/lib/python3.12/dist-packages/nixl* /usr/local/lib/python3.12/dist-packages/.nixl* && \
    python3 -m pip install --no-cache-dir --no-deps /tmp/wheels/*.whl && \
    python3 -m pip install --no-cache-dir --no-deps nixl && \
    # Drop the meta's nixl-cuNN deps so downstream installs (e.g. modelexpress's `nixl[cu12]`) can't
    # pull a prebuilt bundled-UCX wheel back in; our build is the only backend.
    sed -i '/^Requires-Dist: nixl-cu1[23]/d' /usr/local/lib/python3.12/dist-packages/nixl-*.dist-info/METADATA && \
    # Assert the UCX plugin links HPC-X's MT UCX (LD_LIBRARY_PATH below selects it at runtime).
    P="$(find /usr/local/lib/python3.12/dist-packages -name 'libplugin_UCX.so' | head -1)" && \
    LD_LIBRARY_PATH="/opt/hpcx/ucx/mt/lib:${LD_LIBRARY_PATH}" ldd "$P" | grep -q "/opt/hpcx/ucx/mt/lib/libucp.so.0" || \
      { echo "NIXL UCX plugin not linked to HPC-X MT UCX:"; LD_LIBRARY_PATH="/opt/hpcx/ucx/mt/lib:${LD_LIBRARY_PATH}" ldd "$P"; exit 1; } && \
    LD_LIBRARY_PATH="/opt/hpcx/ucx/mt/lib:${LD_LIBRARY_PATH}" python3 -c "import nixl; print('nixl import OK')"

# Prepend MT UCX so its libucp wins over the base image's non-MT /opt/hpcx/ucx/lib.
ENV LD_LIBRARY_PATH=/opt/hpcx/ucx/mt/lib:${LD_LIBRARY_PATH}

# Re-prepend on login shells too (HPC-X's profile.d regenerates LD_LIBRARY_PATH; zz- sorts last).
RUN printf 'export LD_LIBRARY_PATH=/opt/hpcx/ucx/mt/lib:$LD_LIBRARY_PATH\n' > /etc/profile.d/zz-nixl-ucx-mt.sh

EXPOSE 8080
