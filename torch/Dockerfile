# syntax=docker/dockerfile:1.4
ARG BUILDER_BASE_IMAGE="nvidia/cuda:12.0.1-devel-ubuntu22.04"
ARG FINAL_BASE_IMAGE="nvidia/cuda:12.0.1-base-ubuntu22.04"

ARG BUILD_TORCH_VERSION="2.3.0"
ARG BUILD_TORCH_VISION_VERSION="0.18.0"
ARG BUILD_TORCH_AUDIO_VERSION="2.3.0"
ARG BUILD_TRITON_VERSION=""
ARG BUILD_TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6 8.9 9.0+PTX"
# 8.7 is supported in the PyTorch main branch, but not 2.0.0

# Clone PyTorch repositories independently from all other build steps
# for cache-friendliness and parallelization
FROM alpine/git:2.40.1 as downloader-base
WORKDIR /git
RUN git config --global advice.detachedHead false

COPY <<-"EOT" /git/clone.sh
    #!/bin/sh
    REPO="https://github.com/$1";
    DEST="$2";
    REF="$3";

    CLONE() { git clone -j8 --depth=1 --filter=blob:none "$@"; };

    # Try cloning REF as a tag prefixed with "v", otherwise fall back
    # to git checkout for commit hashes
    CLONE --recurse-submodules --shallow-submodules --also-filter-submodules \
      "$REPO" -b "v$REF" "$DEST" || { \
        CLONE --no-single-branch --no-checkout "$REPO" "$DEST" && \
        git -C "$DEST" checkout "$REF" && \
        git -C "$DEST" submodule update --init --filter=blob:none --depth=1 --recursive --jobs 8; \
    };
EOT

RUN chmod 755 /git/clone.sh


FROM downloader-base as pytorch-downloader
ARG BUILD_TORCH_VERSION
RUN ./clone.sh pytorch/pytorch pytorch "${BUILD_TORCH_VERSION}" && \
    rm -rf pytorch/.git

FROM downloader-base as torchvision-downloader
ARG BUILD_TORCH_VISION_VERSION
RUN ./clone.sh pytorch/vision vision "${BUILD_TORCH_VISION_VERSION}" && \
    rm -rf vision/.git

FROM downloader-base as torchaudio-downloader
ARG BUILD_TORCH_AUDIO_VERSION
RUN ./clone.sh pytorch/audio audio "${BUILD_TORCH_AUDIO_VERSION}"
# The torchaudio build requires that this directory remain a full git repository,
# so no rm -rf audio/.git is done for this one.

FROM downloader-base as triton-downloader
ARG BUILD_TRITON_VERSION
RUN if [ -n "${BUILD_TRITON_VERSION}" ]; then \
      ./clone.sh openai/triton triton "${BUILD_TRITON_VERSION}"; \
    else \
      mkdir triton; \
    fi;


## Build PyTorch on a builder image.
FROM ${BUILDER_BASE_IMAGE} as builder
ENV DEBIAN_FRONTEND=noninteractive

ARG BUILD_CCACHE_SIZE="1Gi"

# ninja-build, ccache, and lld are optional but improve the build
RUN apt-get -qq update && apt-get -qq install -y \
      libncurses5 python3 python3-pip git apt-utils ssh ca-certificates \
      libpng-dev libjpeg-dev pkg-config python3-distutils python3-numpy \
      build-essential ninja-build && \
    apt-get clean && \
    /usr/bin/python3 -m pip install --no-cache-dir --upgrade pip && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

RUN mkdir /tmp/ccache-install && \
    cd /tmp/ccache-install && \
    CCACHE_URL='https://github.com/ccache/ccache/releases/download/v4.8.2/ccache-4.8.2-linux-x86_64.tar.xz' && \
    wget -qO - $CCACHE_URL | tar --strip-components 1 -xJf - && \
    make install && \
    cd .. && \
    rm -rf /tmp/ccache-install && \
    ccache -M "${BUILD_CCACHE_SIZE}" && \
    ccache -F 0

# Build-time environment variables
ENV CCACHE_DIR=/ccache \
    CMAKE_C_COMPILER_LAUNCHER=ccache \
    CMAKE_CXX_COMPILER_LAUNCHER=ccache \
    CMAKE_CUDA_COMPILER_LAUNCHER=ccache

# Add Kitware's apt repository to get a newer version of CMake
RUN apt-get -qq update && apt-get -qq install -y \
      software-properties-common lsb-release && \
    { wget -qO - https://apt.kitware.com/keys/kitware-archive-latest.asc \
    | gpg --dearmor -o /etc/apt/trusted.gpg.d/kitware.gpg; } && \
    apt-add-repository -n "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" && \
    apt-get -qq update && apt-get -qq install -y cmake && apt-get clean

# Update compiler (GCC) and linker (LLD) versions
RUN CODENAME="$(lsb_release -cs)" && \
    wget -qO - 'https://apt.llvm.org/llvm-snapshot.gpg.key' > /etc/apt/trusted.gpg.d/apt.llvm.org.asc && \
    apt-add-repository -n "deb https://apt.llvm.org/$CODENAME/ llvm-toolchain-$CODENAME-17 main" && \
    SETUP_TOOLCHAIN() { \
        apt-add-repository -y ppa:ubuntu-toolchain-r/test 2>&1 \
        | sed -e '/connection timed out/{p; Q1}' && \
        apt-get -qq install --no-install-recommends -y gcc-11 g++-11 lld-17 && \
        apt-get clean && \
        update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 11 && \
        update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 11 && \
        update-alternatives --install /usr/bin/ld ld /usr/bin/ld.lld-17 1; \
    } && \
    { SETUP_TOOLCHAIN || { sleep "$(shuf -i10-20 -n1)" && SETUP_TOOLCHAIN; }; }

RUN mkdir /build /build/dist
WORKDIR /build
COPY --chmod=755 effective_cpu_count.sh .
COPY --chmod=755 scale.sh .

COPY <<-"EOT" /build/version-string.sh
    #!/bin/sh
    set -x;
    VERSION="$1";

    IS_HASH() {
      echo "$1" | grep -qxiEe '[0-9a-f]{40}';
    };

    if IS_HASH "$VERSION"; then
      REAL_VERSION="$(cat ./version.txt)";
      SHORT_HASH="$(echo "$VERSION" | cut -c1-7)";
      echo "$REAL_VERSION+$SHORT_HASH";
    else
      echo "$VERSION";
    fi;
EOT
RUN chmod 755 /build/version-string.sh

COPY <<-"EOT" /build/storage-info.sh
    #!/bin/sh
    set -x;
    TARGET="$(realpath "$1")";

    STORAGE_INFO="$(df -h '--output=fstype,used,avail,pcent,target' "$TARGET")" || exit 0;
    printf 'Storage info for %s:\n%s\n' "$TARGET" "$STORAGE_INFO";
EOT
RUN chmod 755 /build/storage-info.sh

## Build torch
RUN --mount=type=bind,from=pytorch-downloader,source=/git/pytorch,target=pytorch/ \
    cd pytorch && pip3 install --no-cache-dir -r requirements.txt

# Build tool & library paths, shared for all libraries to be built
ENV CMAKE_PREFIX_PATH=/usr/bin/ \
    LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64:/usr/local/lib \
    CUDA_BIN_PATH=/usr/local/cuda/bin \
    CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda/ \
    CUDNN_LIB_DIR=/usr/local/cuda/lib64

ARG BUILD_TRITON_VERSION
RUN --mount=type=bind,from=triton-downloader,source=/git/triton,target=triton/,rw \
    --mount=type=cache,target=/ccache \
    if [ -n "$BUILD_TRITON_VERSION" ]; then \
      export MAX_JOBS="$(./scale.sh "$(./effective_cpu_count.sh)" 5 20)" && \
      cd triton/python && \
      python3 -m pip wheel -w ../../dist/ --no-build-isolation --no-deps -vv . && \
      pip3 install ../../dist/*.whl; \
    fi

ARG BUILD_TORCH_VERSION
ARG BUILD_TORCH_CUDA_ARCH_LIST
ENV TORCH_VERSION=$BUILD_TORCH_VERSION
ENV TORCH_CUDA_ARCH_LIST=$BUILD_TORCH_CUDA_ARCH_LIST

# If the directory /opt/nccl-tests exists,
# the base image is assumed to be nccl-tests,
# so it uses the system's special NCCL and UCC installations for the build.
#
# Additionally, this RUN is executed with the downloaded PyTorch repository
# mounted temporarily in "rw" mode, which allows ephemeral writes like
# OverlayFS would that do not mutate the downloaded copy.
# This means the downloaded data never needs to be duplicated in the cache in
# a layer of this build step, and temporary build files are automatically
# cleaned up at the end of the step once the directory is detached.
#
# This step is itself cacheable as long as the downloaded files (and ARCH_LIST)
# remain the same.
RUN --mount=type=bind,from=pytorch-downloader,source=/git/pytorch,target=pytorch/,rw \
    --mount=type=cache,target=/ccache \
    export MAX_JOBS="$(./scale.sh "$(./effective_cpu_count.sh)" 5 20)" && \
    echo "MAX_JOBS: ${MAX_JOBS}" && \
    ./storage-info.sh && \
    cd pytorch && \
    ../storage-info.sh && \
    mkdir build && \
    ln -s /usr/bin/cc build/cc && \
    ln -s /usr/bin/c++ build/c++ && \
    { if [ -d /opt/nccl-tests ]; then \
      export \
        USE_DISTRIBUTED=1 \
        USE_CUDNN=1 \
        USE_NCCL=1 USE_SYSTEM_NCCL=1 \
        UCC_HOME=${HPCX_UCC_DIR} UCX_HOME=${HPCX_UCX_DIR} \
        USE_NCCL_WITH_UCC=1 \
        USE_UCC=1 USE_SYSTEM_UCC=1; fi; } && \
    USE_OPENCV=1 \
    BUILD_TORCH=ON \
    BUILD_TEST=0 \
    CUDA_HOST_COMPILER=cc \
    USE_CUDA=1 \
    USE_NNPACK=1 \
    CC=cc \
    CXX=c++ \
    USE_EIGEN_FOR_BLAS=ON \
    USE_MKL=OFF \
    PYTORCH_BUILD_VERSION="$(../version-string.sh "$TORCH_VERSION")" \
    PYTORCH_BUILD_NUMBER=0 \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    python3 setup.py bdist_wheel --dist-dir ../dist
RUN pip3 install --no-cache-dir --upgrade dist/torch*.whl

## Build torchvision
ARG BUILD_TORCH_VISION_VERSION
ENV TORCH_VISION_VERSION=$BUILD_TORCH_VISION_VERSION
RUN pip3 install --no-cache-dir --upgrade \
    matplotlib numpy typing_extensions requests pillow

RUN --mount=type=bind,from=torchvision-downloader,source=/git/vision,target=vision/,rw \
    --mount=type=cache,target=/ccache \
    export MAX_JOBS="$(./scale.sh "$(./effective_cpu_count.sh)" 5 20)" && \
    cd vision && \
    mkdir build && \
    ln -s /usr/bin/cc build/cc && \
    ln -s /usr/bin/c++ build/c++ && \
    { if [ -d /opt/nccl-tests ]; then \
      export \
        USE_DISTRIBUTED=1 \
        USE_CUDNN=1 \
        USE_NCCL=1 USE_SYSTEM_NCCL=1 \
        UCC_HOME=${HPCX_UCC_DIR} UCX_HOME=${HPCX_UCX_DIR} \
        USE_NCCL_WITH_UCC=1 \
        USE_UCC=1 USE_SYSTEM_UCC=1; fi; } && \
    USE_OPENCV=1 \
    BUILD_TORCH=ON \
    BUILD_TEST=0 \
    CUDA_HOST_COMPILER=cc \
    USE_CUDA=1 \
    FORCE_CUDA=1 \
    USE_NNPACK=1 \
    CC=cc \
    CXX=c++ \
    USE_EIGEN_FOR_BLAS=ON \
    USE_MKL=OFF \
    BUILD_VERSION="$(../version-string.sh "$TORCH_VISION_VERSION")" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    python3 setup.py bdist_wheel --dist-dir ../dist

## Build torchaudio
ARG BUILD_TORCH_AUDIO_VERSION
ENV TORCH_AUDIO_VERSION=$BUILD_TORCH_AUDIO_VERSION
RUN pip3 install --no-cache-dir --upgrade \
    matplotlib numpy typing_extensions requests pillow

RUN --mount=type=bind,from=torchaudio-downloader,source=/git/audio,target=audio/,rw \
    --mount=type=cache,target=/ccache \
    export MAX_JOBS="$(./scale.sh "$(./effective_cpu_count.sh)" 5 20)" && \
    cd audio && \
    mkdir build && \
    ln -s /usr/bin/cc build/cc && \
    ln -s /usr/bin/c++ build/c++ && \
    { if [ -d /opt/nccl-tests ]; then \
      export \
        USE_DISTRIBUTED=1 \
        USE_CUDNN=1 \
        USE_NCCL=1 USE_SYSTEM_NCCL=1 \
        UCC_HOME=${HPCX_UCC_DIR} UCX_HOME=${HPCX_UCX_DIR} \
        USE_NCCL_WITH_UCC=1 \
        USE_UCC=1 USE_SYSTEM_UCC=1; fi; } && \
    USE_OPENCV=1 \
    BUILD_TORCH=ON \
    BUILD_TEST=0 \
    CUDA_HOST_COMPILER=cc \
    USE_CUDA=1 \
    FORCE_CUDA=1 \
    USE_NNPACK=1 \
    CC=cc \
    CXX=c++ \
    USE_EIGEN_FOR_BLAS=ON \
    USE_MKL=OFF \
    BUILD_VERSION="$(../version-string.sh "$TORCH_AUDIO_VERSION")" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    python3 setup.py bdist_wheel --dist-dir ../dist


## Build the final torch image.
FROM ${FINAL_BASE_IMAGE}
ENV DEBIAN_FRONTEND=noninteractive

# Install core packages
RUN apt-get -qq update && apt-get -qq install -y \
      libncurses5 python3 python3-pip python3-distutils python3-numpy \
      libpng16-16 libjpeg-turbo8 libsodium23 \
      curl git apt-utils ssh ca-certificates tmux nano vim-tiny sudo bash \
      rsync htop wget unzip tini && \
    /usr/bin/python3 -m pip install --no-cache-dir --upgrade pip && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1 && \
    update-alternatives --install /usr/bin/vim vim /usr/bin/vim.tiny 1 && \
    apt-get clean

RUN apt-get -qq update && apt-get -qq install -y --no-install-recommends \
        software-properties-common && \
    SETUP_LIBSTDCXX() { \
        apt-add-repository -y ppa:ubuntu-toolchain-r/test 2>&1 \
        | sed -e '/connection timed out/{p; Q1}' && \
        apt-get -qq install -y --no-install-recommends libstdc++6 && \
        apt-get clean; \
    } && \
    { SETUP_LIBSTDCXX || { sleep "$(shuf -i10-20 -n1)" && SETUP_LIBSTDCXX; }; }

ARG BUILD_TORCH_VERSION
ARG BUILD_TORCH_VISION_VERSION
ARG BUILD_TORCH_AUDIO_VERSION
ARG BUILD_TORCH_CUDA_ARCH_LIST
ENV TORCH_VERSION=$BUILD_TORCH_VERSION
ENV TORCH_VISION_VERSION=$BUILD_TORCH_VISION_VERSION
ENV TORCH_AUDIO_VERSION=$BUILD_TORCH_AUDIO_VERSION
ENV TORCH_CUDA_ARCH_LIST=$BUILD_TORCH_CUDA_ARCH_LIST

# - libnvjitlink-X-Y only exists for CUDA versions >= 12-0.
# - Don't mess with libnccl2 when using nccl-tests as a base,
#   checked via the existence of the directory "/opt/nccl-tests".
RUN export \
      CUDA_MAJOR_VERSION=$(echo $CUDA_VERSION | cut -d. -f1) \
      CUDA_MINOR_VERSION=$(echo $CUDA_VERSION | cut -d. -f2) && \
    export \
      CUDA_PACKAGE_VERSION="${CUDA_MAJOR_VERSION}-${CUDA_MINOR_VERSION}" && \
    apt-get -qq update && \
    apt-get -qq install --no-upgrade -y \
      libcurand-${CUDA_PACKAGE_VERSION} \
      libcufft-${CUDA_PACKAGE_VERSION} \
      libcublas-${CUDA_PACKAGE_VERSION} \
      cuda-nvrtc-${CUDA_PACKAGE_VERSION} \
      libcusparse-${CUDA_PACKAGE_VERSION} \
      libcusolver-${CUDA_PACKAGE_VERSION} \
      cuda-cupti-${CUDA_PACKAGE_VERSION} \
      libnvjpeg-${CUDA_PACKAGE_VERSION} \
      libnvtoolsext1 && \
    { if [ $CUDA_MAJOR_VERSION -ge 12 ]; then \
      apt-get -qq install --no-upgrade -y libnvjitlink-${CUDA_PACKAGE_VERSION}; fi; } && \
    { if [ ! -d /opt/nccl-tests ]; then \
      export NCCL_PACKAGE_VERSION="2.*+cuda${CUDA_MAJOR_VERSION}.${CUDA_MINOR_VERSION}" && \
      apt-get -qq install --no-upgrade -y "libnccl2=$NCCL_PACKAGE_VERSION"; fi; } && \
    apt-get clean && \
    ldconfig

WORKDIR /usr/src/app

# Install custom PyTorch wheels.
RUN --mount=type=bind,from=builder,source=/build/dist,target=. \
    pip3 install --no-cache-dir -U ./*.whl
