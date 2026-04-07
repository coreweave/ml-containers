# syntax=docker/dockerfile:1.10

ARG BASE_IMAGE
ARG DEEPSPEED_VERSION="0.18.8"
ARG APEX_COMMIT="5c8860f1dc79f9edc3c1ba4050fe841207834564"
ARG DEEPSPEED_KERNELS_COMMIT="e77acc40b104696d4e73229b787d1ef29a9685b1"
ARG DEEPSPEED_KERNELS_CUDA_ARCH_LIST="80;86;89;90"
ARG XFORMERS_VERSION="0.0.35"
ARG BUILD_MAX_JOBS=""
ARG SCCACHE_VERSION="0.14.0"

FROM alpine/curl:8.17.0 AS sccache-downloader
ARG SCCACHE_VERSION
RUN ARCH=$(uname -m) && \
    curl -fsSL "https://github.com/mozilla/sccache/releases/download/v${SCCACHE_VERSION}/sccache-v${SCCACHE_VERSION}-${ARCH}-unknown-linux-musl.tar.gz" \
    | tar xz --strip-components=1 -C /opt/ "sccache-v${SCCACHE_VERSION}-${ARCH}-unknown-linux-musl/sccache" && \
    chmod 755 /opt/sccache

FROM alpine/git:2.36.3 as xformers-downloader
WORKDIR /git
ARG XFORMERS_VERSION
RUN git clone --filter=blob:none --depth 1 --recurse-submodules --shallow-submodules \
      -b "v${XFORMERS_VERSION}" \
      https://github.com/facebookresearch/xformers && \
    rm -rf xformers/.git

FROM alpine/git:2.36.3 as deepspeed-downloader
WORKDIR /git
ARG DEEPSPEED_VERSION
RUN git clone --filter=blob:none --depth 1 \
      -b "v${DEEPSPEED_VERSION}" \
      https://github.com/microsoft/DeepSpeed deepspeed && \
    rm -rf deepspeed/.git

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

# Port of NVIDIA/apex#1985 to fix compilation with PyTorch > v2.11
RUN sed -i 's@\blerp\b@_lerp@g' \
      apex/apex/contrib/csrc/optimizers/multi_tensor_distopt_adam_kernel.cu


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

RUN python3 -m pip install --no-cache-dir -U pip 'setuptools<82' wheel packaging

# Install the cuDNN dev package for building Apex
# The cuDNN runtime is installed in the base torch image
COPY --link --from=torch-common --chmod=755 install_cudnn.sh /tmp/install_cudnn.sh
RUN /tmp/install_cudnn.sh "${CUDA_VERSION}" dev && \
    rm /tmp/install_cudnn.sh

# Add Kitware's apt repository to get a newer version of CMake
RUN apt-get -qq update && apt-get -qq install -y \
      software-properties-common lsb-release && \
    { wget -qO - https://apt.kitware.com/keys/kitware-archive-latest.asc \
    | gpg --dearmor -o /etc/apt/trusted.gpg.d/kitware.gpg; } && \
    apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" && \
    apt-get -qq update && \
    apt-get -qq install -y 'cmake=3.31.6-*' 'cmake-data=3.31.6-*' && \
    apt-get clean && \
    python3 -m pip install --no-cache-dir 'cmake==3.31.6'

# Update compiler (GCC) and linker (LLD) versions
# gfortran-13 is just for compiler_wrapper.f95
RUN LLVM_VERSION='18' && \
    if [ "$(. /etc/os-release && echo "$VERSION_ID")" = '22.04' ]; then \
      add-apt-repository -n ppa:ubuntu-toolchain-r/test; \
    fi && \
    apt-get -qq update && apt-get -qq install --no-install-recommends -y \
      gcc-13 g++-13 gfortran-13 "lld-$LLVM_VERSION" && \
    apt-get clean && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 13 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 13 && \
    update-alternatives --install /usr/bin/cc cc /usr/bin/gcc 1 && \
    update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++ 1 && \
    update-alternatives --install \
      /usr/bin/gfortran gfortran /usr/bin/gfortran-13 13 && \
    update-alternatives --install /usr/bin/ld ld "/usr/bin/ld.lld-$LLVM_VERSION" 1

RUN mkdir /wheels /build
WORKDIR /build

# DeepSpeed forces -march=native into the compiler options,
# making the result dependent on the processor architecture
# used on the builder machine.
# The compiler wrapper normalizes -march=native to -march=skylake
# along with a couple other transformations before invoking GCC.
COPY --link --from=torch-common compiler_wrapper.f95 .
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
ENV CC_WRAPPED="/build/compiler"

COPY --link --from=torch-common --chmod=755 effective_cpu_count.sh .
COPY --link --from=torch-common --chmod=755 scale.sh .

# Setup for sccache and its wrappers.
# Keep the actual sccache binary off the PATH to prevent nesting shenanigans.
COPY --link --from=sccache-downloader /opt/sccache /opt/sccache
COPY --link --from=common --chmod=755 sccache*.sh /opt/
COPY --link --from=common sccache.toml /etc/sccache.toml
RUN sed -Ei 's@^(key_prefix.*)misc@\1torch-extras@' /etc/sccache.toml
ENV SCCACHE_CONF=/etc/sccache.toml

ENV CC=/opt/sccache-cc.sh \
    CXX=/opt/sccache-c++.sh

# Skip nvcc's -MD dependency generation in torch extension builds,
# which sccache cannot cache. These are always built from scratch,
# so incremental build dependency tracking has no value anyway.
ENV TORCH_EXTENSION_SKIP_NVCC_GEN_DEPENDENCIES=1

COPY --link --from=torch-common --chmod=755 nvcc-wrapper.py /build/nvcc-wrapper.py
ENV PYTORCH_NVCC='/build/nvcc-wrapper.py' \
    CMAKE_CUDA_COMPILER='/build/nvcc-wrapper.py' \
    CUDACXX='/build/nvcc-wrapper.py'

ARG BUILD_NVCC_APPEND_FLAGS="-gencode=arch=compute_90a,code=sm_90a"
RUN FLAGS="$BUILD_NVCC_APPEND_FLAGS" && \
    case "${NV_CUDA_LIB_VERSION}" in 12.[0-7].*) ;; *) \
      FLAGS="${FLAGS}$( \
        printf -- ' -gencode=arch=compute_%s,code=sm_%s' 120 120 100 100 100a 100a 103a 103a \
      )" ;; \
    esac && \
    { printf '%s' \
      "-Wno-deprecated-gpu-targets" \
      " --diag-suppress=191,186,177" \
      " -Xcompiler -Wno-sign-compare" \
      ",-Wno-deprecated-declarations" \
      ",-Wno-unused-parameter" \
      ",-Wno-unused-variable" \
      ",-Wno-unused-result" \
      "${FLAGS:+ $FLAGS}" && echo; \
    } > /build/nvcc.conf
ARG BUILD_MAX_JOBS

COPY <<-"EOT" /build/build-wheel.sh
	#!/bin/bash
	set -eo pipefail
	python3 -m pip wheel -w /wheels \
	  --no-cache-dir --no-build-isolation --no-deps -v "$@" 2>&1 \
	| grep -Ev --line-buffered \
	  '^(ptxas (/tmp/|(info|warning)\s*:)|copying .+/|adding .+/|creating build/|Wrapped header:)|bytes spill stores'
EOT
RUN chmod 755 /build/build-wheel.sh

FROM builder-base as deepspeed-builder

ARG DEEPSPEED_KERNELS_CUDA_ARCH_LIST
RUN --mount=type=bind,from=ds-kernels-downloader,source=/git/ds-kernels,target=ds-kernels/,rw \
    --mount=type=secret,id=s3_access_key_id,env=AWS_ACCESS_KEY_ID \
    --mount=type=secret,id=s3_secret_access_key,env=AWS_SECRET_ACCESS_KEY \
    --mount=type=tmpfs,target=/sccache \
    --mount=type=tmpfs,target=/tmp \
    . /opt/sccache-start.sh && \
    export NVCC_APPEND_FLAGS="$(cat /build/nvcc.conf)" && \
    cd ds-kernels && \
    export CUDA_ARCH_LIST="${DEEPSPEED_KERNELS_CUDA_ARCH_LIST}" && \
    echo "CUDA_ARCH_LIST: ${CUDA_ARCH_LIST}" && \
    /build/build-wheel.sh . && \
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

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN --mount=type=bind,from=deepspeed-downloader,source=/git/deepspeed,target=deepspeed/,rw \
    --mount=type=secret,id=s3_access_key_id,env=AWS_ACCESS_KEY_ID \
    --mount=type=secret,id=s3_secret_access_key,env=AWS_SECRET_ACCESS_KEY \
    --mount=type=tmpfs,target=/sccache \
    --mount=type=tmpfs,target=/tmp \
    . /opt/sccache-start.sh && \
    export NVCC_APPEND_FLAGS="$(cat /build/nvcc.conf)" && \
    python3 -m pip install -U --no-cache-dir py-cpuinfo && \
    # BuildKit builders have no GPU, so without this override DeepSpeed's
    # auto-detection would record accelerator_name='cpu' in the package's
    # git_version_info_installed.py, causing AOT-compiled ops to be skipped
    # at runtime in favour of JIT compilation.
    export DS_ACCELERATOR='cuda' && \
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
    cd deepspeed && \
    MAX_JOBS="${BUILD_MAX_JOBS:-$(./scale.sh "$(./effective_cpu_count.sh)" 4 24)}" \
      /build/build-wheel.sh .
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
    --mount=type=secret,id=s3_access_key_id,env=AWS_ACCESS_KEY_ID \
    --mount=type=secret,id=s3_secret_access_key,env=AWS_SECRET_ACCESS_KEY \
    --mount=type=tmpfs,target=/sccache \
    --mount=type=tmpfs,target=/tmp \
    . /opt/sccache-start.sh && \
    export NVCC_APPEND_FLAGS="$(cat /build/nvcc.conf)" && \
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
    xargs -a ../apex-extensions.conf /build/build-wheel.sh .

WORKDIR /wheels

FROM builder-base as xformers-builder

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN --mount=type=bind,from=xformers-downloader,source=/git/xformers,target=xformers/,rw \
    --mount=type=secret,id=s3_access_key_id,env=AWS_ACCESS_KEY_ID \
    --mount=type=secret,id=s3_secret_access_key,env=AWS_SECRET_ACCESS_KEY \
    --mount=type=tmpfs,target=/sccache \
    --mount=type=tmpfs,target=/tmp \
    . /opt/sccache-start.sh && \
    export NVCC_APPEND_FLAGS="$(cat /build/nvcc.conf)" && \
    cd xformers && \
    MAX_JOBS=1 PYTHONUNBUFFERED=1 XFORMERS_DISABLE_FLASH_ATTN=1 \
      /build/build-wheel.sh .
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
