# syntax=docker/dockerfile:1.2

ARG BASE_IMAGE
ARG DEEPSPEED_VERSION="0.9.4"
ARG FLASH_ATTN_VERSION="1.0.7"

FROM alpine/git:2.36.3 as flash-attn-downloader
WORKDIR /git
ARG FLASH_ATTN_VERSION
RUN git clone --recurse-submodules --shallow-submodules -j8 --depth 1 \
      https://github.com/HazyResearch/flash-attention -b v${FLASH_ATTN_VERSION} && \
    rm -rf flash-attention/.git


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
      parallel \
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
      python3 -m pip wheel -w /wheels \
      --no-cache-dir --no-build-isolation --no-deps \
      deepspeed==${DEEPSPEED_VERSION} && \
    rm ./*
SHELL ["/bin/sh", "-c"]

WORKDIR /wheels


FROM builder-base as flash-attn-builder
ARG FLASH_ATTN_VERSION

RUN --mount=type=bind,from=flash-attn-downloader,source=/git/flash-attention,target=flash-attention/ \
    python3 -m pip install -U --no-cache-dir \
      packaging setuptools wheel pip && \
    export CC=$(realpath -e ./compiler) && \
    cd flash-attention && \
    parallel 'cd {} && python3 setup.py bdist_wheel --dist-dir /wheels' ::: \
      . \
      csrc/ft_attention \
      csrc/fused_dense_lib \
      csrc/fused_softmax \
      csrc/layer_norm \
      csrc/rotary \
      csrc/xentropy

WORKDIR /wheels


FROM ${BASE_IMAGE}

RUN apt-get -qq update && \
    apt-get install -y --no-install-recommends libaio-dev && \
    apt-get clean

RUN --mount=type=bind,from=deepspeed-builder,source=/wheels,target=/tmp/wheels \
    python3 -m pip install --no-cache-dir /tmp/wheels/*.whl
RUN --mount=type=bind,from=flash-attn-builder,source=/wheels,target=/tmp/wheels \
    python3 -m pip install --no-cache-dir /tmp/wheels/*.whl
RUN rm -r /tmp/wheels
