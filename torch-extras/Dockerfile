# syntax=docker/dockerfile:1.2

ARG BASE_IMAGE
ARG DEEPSPEED_VERSION="0.14.4"
ARG APEX_COMMIT="a1df80457ba67d60cbdb0d3ddfb08a2702c821a8"
ARG DEEPSPEED_KERNELS_COMMIT="e77acc40b104696d4e73229b787d1ef29a9685b1"
ARG DEEPSPEED_KERNELS_CUDA_ARCH_LIST="80;86;89;90"
ARG XFORMERS_VERSION="0.0.28.post1"
ARG BUILD_MAX_JOBS=""

FROM alpine/git:2.36.3 as apex-downloader
WORKDIR /git
ARG APEX_COMMIT
RUN git clone --filter=blob:none --depth 1 --no-single-branch --no-checkout \
      https://github.com/NVIDIA/apex && \
    cd apex && \
    git checkout "${APEX_COMMIT}" && \
    git submodule update --init --recursive --jobs 8 \
      --depth 1 --filter=blob:none && \
    find -type d -name docs -prune -exec rm -r '{}' ';'


FROM alpine/git:2.36.3 as ds-kernels-downloader
WORKDIR /git
ARG DEEPSPEED_KERNELS_COMMIT
RUN git clone --filter=blob:none --depth 1 --no-single-branch --no-checkout \
      https://github.com/microsoft/DeepSpeed-Kernels ds-kernels && \
    cd ds-kernels && \
    git checkout "${DEEPSPEED_KERNELS_COMMIT}" && \
    git submodule update --init --recursive --jobs 8 \
      --depth 1 --filter=blob:none


# Dependencies requiring NVCC are built ahead of time in a separate stage
# so that the ~2 GiB dev library installations don't have to be included
# in the final image.
FROM ${BASE_IMAGE} as builder-base
RUN export \
      CUDA_MAJOR_VERSION=$(echo $CUDA_VERSION | cut -d. -f1) \
      CUDA_MINOR_VERSION=$(echo $CUDA_VERSION | cut -d. -f2) && \
    export \
      CUDA_PACKAGE_VERSION="${CUDA_MAJOR_VERSION}-${CUDA_MINOR_VERSION}" && \
    apt-get install -y --no-install-recommends --no-upgrade \
      cuda-nvcc-${CUDA_PACKAGE_VERSION} \
      cuda-nvml-dev-${CUDA_PACKAGE_VERSION} \
      libcurand-dev-${CUDA_PACKAGE_VERSION} \
      libcublas-dev-${CUDA_PACKAGE_VERSION} \
      libcusparse-dev-${CUDA_PACKAGE_VERSION} \
      libcusolver-dev-${CUDA_PACKAGE_VERSION} \
      cuda-profiler-api-${CUDA_PACKAGE_VERSION} \
      cuda-nvtx-${CUDA_PACKAGE_VERSION} \
      cuda-nvrtc-dev-${CUDA_PACKAGE_VERSION} && \
    apt-get -qq update && \
    apt-get install -y --no-install-recommends \
      libaio-dev \
      ninja-build && \
    apt-get clean

# Install the cuDNN dev package for building Apex
# The cuDNN runtime is installed in the base torch image
COPY --chmod=755 install_cudnn.sh /tmp/install_cudnn.sh
RUN /tmp/install_cudnn.sh "${CUDA_VERSION}" dev && \
    rm /tmp/install_cudnn.sh

# Add Kitware's apt repository to get a newer version of CMake
RUN apt-get -qq update && apt-get -qq install -y \
      software-properties-common lsb-release && \
    { wget -qO - https://apt.kitware.com/keys/kitware-archive-latest.asc \
    | gpg --dearmor -o /etc/apt/trusted.gpg.d/kitware.gpg; } && \
    apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" && \
    apt-get -qq update && apt-get -qq install -y cmake && apt-get clean

# Update compiler (GCC) and linker (LLD) versions
# gfortran-11 is just for compiler_wrapper.f95
RUN LLVM_VERSION='18' && \
    apt-get -qq update && apt-get -qq install --no-install-recommends -y \
      gcc-11 g++-11 gfortran-11 "lld-$LLVM_VERSION" && \
    apt-get clean && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 11 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 11 && \
    update-alternatives --install \
      /usr/bin/gfortran gfortran /usr/bin/gfortran-11 11 && \
    update-alternatives --install /usr/bin/ld ld "/usr/bin/ld.lld-$LLVM_VERSION" 1

RUN mkdir /wheels /build
WORKDIR /build

# DeepSpeed forces -march=native into the compiler options,
# making the result dependent on the processor architecture
# used on the builder machine.
# The compiler wrapper normalizes -march=native to -march=skylake
# along with a couple other transformations before invoking GCC.
COPY compiler_wrapper.f95 .
ARG AMD64_NATIVE_ARCH="skylake"
ARG ARM64_NATIVE_ARCH="armv8.5-a+nopredres"
RUN if [ "$(uname -m)" = "aarch64" ]; then \
      NATIVE="WRAPPER_NATIVE=\"${ARM64_NATIVE_ARCH}\"" && \
      AVX='WRAPPER_NO_AVX'; \
    else \
      NATIVE="WRAPPER_NATIVE=\"${AMD64_NATIVE_ARCH}\"" && \
      AVX='WRAPPER_AVX="AVX256"'; \
    fi && \
    gfortran -ffree-line-length-512 -cpp -O3 "-D${NATIVE}" "-D${AVX}" ./compiler_wrapper.f95 -o ./compiler && rm ./compiler_wrapper.f95

COPY --chmod=755 effective_cpu_count.sh .
COPY --chmod=755 scale.sh .

ARG BUILD_NVCC_APPEND_FLAGS="-gencode=arch=compute_90a,code=compute_90a"
RUN FLAGS="$BUILD_NVCC_APPEND_FLAGS" && \
    case "${NV_CUDA_LIB_VERSION}" in 12.[89].*) \
      FLAGS="${FLAGS} -gencode=arch=compute_100,code=[sm_100,compute_100]" ;; \
    esac && \
    echo "-Wno-deprecated-gpu-targets -diag-suppress 191,186,177${FLAGS:+ $FLAGS}" > /build/nvcc.conf
ARG BUILD_MAX_JOBS


FROM builder-base as deepspeed-builder

ARG DEEPSPEED_KERNELS_CUDA_ARCH_LIST
RUN --mount=type=bind,from=ds-kernels-downloader,source=/git/ds-kernels,target=ds-kernels/,rw \
    export NVCC_APPEND_FLAGS="$(cat /build/nvcc.conf)" && \
    cd ds-kernels && \
    export CUDA_ARCH_LIST="${DEEPSPEED_KERNELS_CUDA_ARCH_LIST}" && \
    echo "CUDA_ARCH_LIST: ${CUDA_ARCH_LIST}" && \
    python3 -m pip wheel -w /wheels \
      --no-cache-dir --no-build-isolation --no-deps . && \
    python3 -m pip install /wheels/*.whl

# DeepSpeed build flags
# See: https://www.deepspeed.ai/tutorials/advanced-install
ARG DS_BUILD_OPS="0"
ARG DS_BUILD_CCL_COMM="0"
ARG DS_BUILD_CPU_ADAM="1"
ARG DS_BUILD_CPU_LION="1"
# Requires CUTLASS
ARG DS_BUILD_EVOFORMER_ATTN="0"
ARG DS_BUILD_FUSED_ADAM="1"
ARG DS_BUILD_FUSED_LION="1"
ARG DS_BUILD_CPU_ADAGRAD="1"
ARG DS_BUILD_FUSED_LAMB="1"
ARG DS_BUILD_QUANTIZER="1"
ARG DS_BUILD_RANDOM_LTD="1"
# sparse_attn has issues with PyTorch >= 2.0.0 as of DeepSpeed 0.9.4
ARG DS_BUILD_SPARSE_ATTN="0"
ARG DS_BUILD_TRANSFORMER="1"
ARG DS_BUILD_TRANSFORMER_INFERENCE="1"
ARG DS_BUILD_STOCHASTIC_TRANSFORMER="1"
ARG DS_BUILD_UTILS="1"
ARG DS_BUILD_AIO="1"

ARG DEEPSPEED_VERSION

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN export NVCC_APPEND_FLAGS="$(cat /build/nvcc.conf)" && \
    python3 -m pip install -U --no-cache-dir \
      setuptools wheel pip py-cpuinfo && \
    if python3 -m pip show torch | grep 'Version: 2\.[1-9]' > /dev/null; then \
      # DeepSpeed's AIO extension is incompatible with PyTorch 2.1.x's
      # requirement for C++17 (as of DeepSpeed 0.10.1).
      # See: https://github.com/microsoft/DeepSpeed/pull/3976
      export DS_BUILD_AIO='0'; \
    fi && \
    { \
      # DeepSpeed doesn't handle blank environment variables
      # in the same way as unset ones, so clear any blank ones.
      for VAR in \
        DS_BUILD_OPS \
        DS_BUILD_CCL_COMM \
        DS_BUILD_CPU_ADAM \
        DS_BUILD_CPU_LION \
        DS_BUILD_EVOFORMER_ATTN \
        DS_BUILD_FUSED_ADAM \
        DS_BUILD_FUSED_LION \
        DS_BUILD_CPU_ADAGRAD \
        DS_BUILD_FUSED_LAMB \
        DS_BUILD_QUANTIZER \
        DS_BUILD_RANDOM_LTD \
        DS_BUILD_SPARSE_ATTN \
        DS_BUILD_TRANSFORMER \
        DS_BUILD_TRANSFORMER_INFERENCE \
        DS_BUILD_STOCHASTIC_TRANSFORMER \
        DS_BUILD_UTILS \
        DS_BUILD_AIO; \
      do if [[ -z ${!VAR} ]]; then unset ${VAR}; fi; done; \
    } && \
    CC=$(realpath -e ./compiler) \
      MAX_JOBS="${BUILD_MAX_JOBS:-$(./scale.sh "$(./effective_cpu_count.sh)" 4 24)}" \
      python3 -m pip wheel -w /wheels \
      --no-cache-dir --no-build-isolation --no-deps -v \
      deepspeed==${DEEPSPEED_VERSION} && \
    rm ./*
SHELL ["/bin/sh", "-c"]

WORKDIR /wheels


FROM builder-base as apex-builder

RUN LIBNCCL2_VERSION=$(dpkg-query --showformat='${Version}' --show libnccl2) && \
    apt-get -qq update && apt-get install -y --no-install-recommends \
      libnccl-dev=$LIBNCCL2_VERSION && \
    apt-get clean

# --distributed_adam, --distributed_lamb, and --group_norm aren't documented
# in the Apex README, but are defined in its setup.py config.
RUN --mount=type=bind,from=apex-downloader,source=/git/apex,target=apex/,rw \
    export NVCC_APPEND_FLAGS="$(cat /build/nvcc.conf)" && \
    python3 -m pip install -U --no-cache-dir \
      packaging setuptools wheel pip && \
    CUDA_MAJOR_VERSION=$(echo "${CUDA_VERSION}" | cut -d. -f1) && \
    CHECK_VERSION() { \
      dpkg-query --status "$1" 2>/dev/null \
      | sed -ne 's/Version: //p' \
      | grep .; \
    } && \
    LIBCUDNN_VER="$( \
      CHECK_VERSION libcudnn8-dev || \
      CHECK_VERSION "libcudnn9-dev-cuda-${CUDA_MAJOR_VERSION}" || \
      :; \
    )" && \
    export CC=$(realpath -e ./compiler) && \
    export MAX_JOBS="${BUILD_MAX_JOBS:-$(./scale.sh "$(./effective_cpu_count.sh)" 8 24)}" && \
    printf -- '--config-settings="--build-option=%s" ' $( \
      echo \
        --cpp_ext \
        --cuda_ext \
        --distributed_adam \
        --distributed_lamb \
        --permutation_search \
        --xentropy \
        --focal_loss \
        --group_norm \
        --index_mul_2d \
        --deprecated_fused_adam \
        --deprecated_fused_lamb \
        --fast_layer_norm \
        --fmha \
        --fast_multihead_attn \
        --transducer \
        --peer_memory \
        --nccl_p2p \
        --fast_bottleneck && \
      if [ -n "$LIBCUDNN_VER" ]; then \
        echo \
          --bnp \
          --cudnn_gbn \
          --fused_conv_bias_relu; \
      fi; \
    ) > ./apex-extensions.conf && \
    echo "Extensions: $(cat ./apex-extensions.conf)" && \
    cd apex && \
    xargs -a ../apex-extensions.conf python3 -m pip wheel -w /wheels -v --no-cache-dir --no-build-isolation --no-deps ./

WORKDIR /wheels

FROM builder-base as xformers-builder

ARG XFORMERS_VERSION

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN export NVCC_APPEND_FLAGS="$(cat /build/nvcc.conf)" && \
    python3 -m pip install -U --no-cache-dir \
      setuptools wheel pip && \
    CC=$(realpath -e ./compiler) \
      MAX_JOBS=1 \
      PYTHONUNBUFFERED=1 \
      XFORMERS_DISABLE_FLASH_ATTN=1 \
      python3 -m pip wheel -w /wheels -v \
      --no-cache-dir --no-build-isolation --no-deps \
      --no-binary=xformers \
      xformers==${XFORMERS_VERSION} 2> \
    >(grep -Ev --line-buffered 'ptxas info\s*:|bytes spill stores' >&2)

SHELL ["/bin/sh", "-c"]

WORKDIR /build

FROM ${BASE_IMAGE}

RUN apt-get -qq update && \
    apt-get install -y --no-install-recommends libaio-dev && \
    apt-get clean


RUN --mount=type=bind,from=deepspeed-builder,source=/wheels,target=/tmp/wheels \
    python3 -m pip install --no-cache-dir /tmp/wheels/*.whl
RUN --mount=type=bind,from=apex-builder,source=/wheels,target=/tmp/wheels \
    python3 -m pip install --no-cache-dir /tmp/wheels/*.whl
RUN --mount=type=bind,from=xformers-builder,source=/wheels,target=/tmp/wheels \
    python3 -m pip install --no-cache-dir /tmp/wheels/*.whl
