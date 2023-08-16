# syntax=docker/dockerfile:1.2

ARG BASE_IMAGE
ARG DEEPSPEED_VERSION="0.9.4"
ARG FLASH_ATTN_VERSION="2.0.2"
ARG APEX_COMMIT="38a12698bc3cc95987bca270bcd6d025bb0be346"

FROM alpine/git:2.36.3 as flash-attn-downloader
WORKDIR /git
ARG FLASH_ATTN_VERSION
RUN git clone --recurse-submodules --shallow-submodules -j8 --depth 1 \
      --filter=blob:none --also-filter-submodules \
      https://github.com/Dao-AILab/flash-attention -b v${FLASH_ATTN_VERSION}

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


# Dependencies requiring NVCC are built ahead of time in a separate stage
# so that the ~2 GiB dev library installations don't have to be included
# in the final image.
FROM ${BASE_IMAGE} as builder-base
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
      ninja-build \
      # gcc-10/g++-10/lld do not need to be installed here, but they improve the build.
      # gfortran-10 is just for compiler_wrapper.f95.
      gcc-10 g++-10 gfortran-10 lld && \
    apt-get clean && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 10 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 10 && \
    update-alternatives --install \
      /usr/bin/gfortran gfortran /usr/bin/gfortran-10 10 && \
    update-alternatives --install /usr/bin/ld ld /usr/bin/ld.lld 1

RUN mkdir /wheels /build
WORKDIR /build

# DeepSpeed forces -march=native into the compiler options,
# making the result dependent on the processor architecture
# used on the builder machine.
# The compiler wrapper normalizes -march=native to -march=skylake
# along with a couple other transformations before invoking GCC.
COPY compiler_wrapper.f95 .
RUN gfortran -O3 ./compiler_wrapper.f95 -o ./compiler && rm ./compiler_wrapper.f95

COPY --chmod=755 effective_cpu_count.sh .


FROM builder-base as deepspeed-builder
# DeepSpeed build flags
# See: https://www.deepspeed.ai/tutorials/advanced-install
ARG DS_BUILD_OPS="1"
ARG DS_BUILD_CPU_ADAM=""
ARG DS_BUILD_FUSED_ADAM=""
ARG DS_BUILD_FUSED_LAMB=""
# sparse_attn has issues with PyTorch >= 2.0.0 as of DeepSpeed 0.9.4
ARG DS_BUILD_SPARSE_ATTN="0"
ARG DS_BUILD_TRANSFORMER=""
ARG DS_BUILD_TRANSFORMER_INFERENCE=""
ARG DS_BUILD_STOCHASTIC_TRANSFORMER=""
ARG DS_BUILD_UTILS=""
ARG DS_BUILD_AIO=""

ARG DEEPSPEED_VERSION

SHELL ["/bin/bash", "-c"]
RUN python3 -m pip install -U --no-cache-dir \
      setuptools wheel pip && \
    { \
      # DeepSpeed doesn't handle blank environment variables
      # in the same way as unset ones, so clear any blank ones.
      for VAR in \
        DS_BUILD_OPS \
        DS_BUILD_CPU_ADAM \
        DS_BUILD_FUSED_ADAM \
        DS_BUILD_FUSED_LAMB \
        DS_BUILD_SPARSE_ATTN \
        DS_BUILD_TRANSFORMER \
        DS_BUILD_TRANSFORMER_INFERENCE \
        DS_BUILD_STOCHASTIC_TRANSFORMER \
        DS_BUILD_UTILS \
        DS_BUILD_AIO; \
      do if [[ -z ${!VAR} ]]; then unset ${VAR}; fi; done; \
    } && \
    CC=$(realpath -e ./compiler) \
      MAX_JOBS=$(($(./effective_cpu_count.sh) + 2)) \
      python3 -m pip wheel -w /wheels \
      --no-cache-dir --no-build-isolation --no-deps \
      deepspeed==${DEEPSPEED_VERSION} && \
    rm ./*
SHELL ["/bin/sh", "-c"]

WORKDIR /wheels


FROM builder-base as flash-attn-builder

RUN --mount=type=bind,from=flash-attn-downloader,source=/git/flash-attention,target=flash-attention/,rw \
    python3 -m pip install -U --no-cache-dir \
      packaging setuptools wheel pip && \
    export CC=$(realpath -e ./compiler) && \
    export MAX_JOBS=$(($(./effective_cpu_count.sh) / 4)) && \
    export MAX_JOBS=$((MAX_JOBS == 0 ? 1 : MAX_JOBS)) && \
    export NVCC_APPEND_FLAGS='-diag-suppress 186,177' && \
    cd flash-attention && \
    ( \
      for EXT_DIR in $(realpath -s -e \
        . \
        csrc/ft_attention \
        csrc/fused_dense_lib \
        csrc/fused_softmax \
        csrc/layer_norm \
        csrc/rotary \
        csrc/xentropy); \
      do \
          cd $EXT_DIR && \
          python3 setup.py bdist_wheel --dist-dir /wheels && \
          cd - || \
          exit 1; \
      done; \
    )

WORKDIR /wheels


FROM builder-base as apex-builder

RUN LIBNCCL2_VERSION=$(dpkg-query --showformat='${Version}' --show libnccl2) && \
    apt-get -qq update && apt-get install -y --no-install-recommends \
      libnccl-dev=$LIBNCCL2_VERSION && \
    apt-get clean

# --distributed_adam, --distributed_lamb, and --group_norm aren't documented
# in the Apex README, but are defined in its setup.py config.
RUN --mount=type=bind,from=apex-downloader,source=/git/apex,target=apex/,rw \
    python3 -m pip install -U --no-cache-dir \
      packaging setuptools wheel pip && \
    export CC=$(realpath -e ./compiler) && \
    export MAX_JOBS=$(($(./effective_cpu_count.sh) + 2)) && \
    export NVCC_APPEND_FLAGS='-diag-suppress 186,177' && \
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
      if dpkg-query --status libcudnn8-dev > /dev/null 2> /dev/null; then \
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


FROM ${BASE_IMAGE}

RUN apt-get -qq update && \
    apt-get install -y --no-install-recommends libaio-dev && \
    apt-get clean

RUN --mount=type=bind,from=deepspeed-builder,source=/wheels,target=/tmp/wheels \
    python3 -m pip install --no-cache-dir /tmp/wheels/*.whl
RUN --mount=type=bind,from=flash-attn-builder,source=/wheels,target=/tmp/wheels \
    python3 -m pip install --no-cache-dir /tmp/wheels/*.whl
RUN --mount=type=bind,from=apex-builder,source=/wheels,target=/tmp/wheels \
    python3 -m pip install --no-cache-dir /tmp/wheels/*.whl
