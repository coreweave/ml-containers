# syntax=docker/dockerfile:1.2
ARG BUILDER_BASE_IMAGE="nvidia/cuda:12.0.1-devel-ubuntu20.04"
ARG FINAL_BASE_IMAGE="nvidia/cuda:12.0.1-base-ubuntu20.04"

ARG BUILD_TORCH_VERSION="2.0.0-rc1"
ARG BUILD_TORCH_VISION_VERSION="0.15.0-rc1"
ARG BUILD_TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6 8.9 9.0+PTX"
# 8.7 is not supported until PyTorch 2.0.0-rc2

# Clone PyTorch repositories independently from all other build steps
# for cache-friendliness and parallelization
FROM alpine/git:2.36.3 as pytorch-downloader
WORKDIR /git
ARG BUILD_TORCH_VERSION
RUN git clone --recurse-submodules --shallow-submodules -j8 --depth 1 \
      https://github.com/pytorch/pytorch -b v${BUILD_TORCH_VERSION} && \
    rm -rf pytorch/.git

FROM alpine/git:2.36.3 as torchvision-downloader
WORKDIR /git
ARG BUILD_TORCH_VISION_VERSION
RUN git clone --recurse-submodules --shallow-submodules -j8 --depth 1 \
      https://github.com/pytorch/vision -b v${BUILD_TORCH_VISION_VERSION} && \
    rm -rf vision/.git

## Build PyTorch on a builder image.
FROM ${BUILDER_BASE_IMAGE} as builder
ENV DEBIAN_FRONTEND=noninteractive

# ninja-build, gcc-10, g++-10, and lld are optional but improve the build
RUN apt-get -qq update && apt-get -qq install -y \
      libncurses5 python3 python3-pip git apt-utils ssh ca-certificates \
      python3-distutils python3-numpy build-essential ninja-build \
      gcc-10 g++-10 lld && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 10 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 10 && \
    update-alternatives --install /usr/bin/ld ld /usr/bin/ld.lld 1 && \
    pip3 install --no-cache-dir --upgrade pip && \
    apt-get clean

# Add Kitware's apt repository to get a newer version of CMake
RUN apt-get -qq update && apt-get -qq install -y \
      software-properties-common lsb-release && \
    { wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null \
    | gpg --dearmor -o /etc/apt/trusted.gpg.d/kitware.gpg; } && \
    apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" && \
    apt-get -qq update && apt-get -qq install -y cmake && apt-get clean

# Workaround for linking against an HPC-X distribution that contains CMake
# modules with invalid absolute paths when using nccl-tests as a base
# See: https://github.com/coreweave/nccl-tests/pull/12
RUN if [ -d /opt/nccl-tests ]; then \
    grep -Irl "/build-result/hpcx-.*-x86_64" ${HPCX_DIR} \
    | xargs --delimiter='\n' --no-run-if-empty \
        sed -i -e "s:/build-result/hpcx-.*-x86_64:${HPCX_DIR}:g"; \
    fi

RUN mkdir /build /build/dist
WORKDIR /build

## Build torch
RUN --mount=type=bind,from=pytorch-downloader,source=/git/pytorch,target=pytorch/ \
    cd pytorch && pip3 install --no-cache-dir -r requirements.txt

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
    cd pytorch && \
    mkdir build && \
    ln -s /usr/bin/cc build/cc && \
    ln -s /usr/bin/c++ build/c++ && \
    { if [ -d /opt/nccl-tests ]; then \
      export \
        USE_DISTRIBUTED=1 \
        USE_NCCL=1 USE_SYSTEM_NCCL=1 \
        UCC_HOME=${HPCX_UCC_DIR} UCX_HOME=${HPCX_UCX_DIR} \
        USE_NCCL_WITH_UCC=1 \
        USE_UCC=1 USE_SYSTEM_UCC=1; fi; } && \
    USE_OPENCV=1 \
    BUILD_TORCH=ON \
    BUILD_TEST=0 \
    CMAKE_PREFIX_PATH="/usr/bin/" \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/lib \
    CUDA_BIN_PATH=/usr/local/cuda/bin \
    CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda/ \
    CUDNN_LIB_DIR=/usr/local/cuda/lib64 \
    CUDA_HOST_COMPILER=cc \
    USE_CUDA=1 \
    USE_NNPACK=1 \
    CC=cc \
    CXX=c++ \
    USE_EIGEN_FOR_BLAS=ON \
    USE_MKL=OFF \
    PYTORCH_BUILD_VERSION="${TORCH_VERSION}" \
    PYTORCH_BUILD_NUMBER=0 \
    TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    python3 setup.py bdist_wheel --dist-dir ../dist
RUN pip3 install --no-cache-dir --upgrade dist/torch*.whl

## Build torchvision
ARG BUILD_TORCH_VISION_VERSION
ENV TORCH_VISION_VERSION=$BUILD_TORCH_VISION_VERSION
RUN pip3 install --no-cache-dir --upgrade \
    matplotlib numpy typing_extensions requests pillow

RUN --mount=type=bind,from=torchvision-downloader,source=/git/vision,target=vision/,rw \
    cd vision && \
    mkdir build && \
    ln -s /usr/bin/cc build/cc && \
    ln -s /usr/bin/c++ build/c++ && \
    { if [ -d /opt/nccl-tests ]; then \
      export \
        USE_DISTRIBUTED=1 \
        USE_NCCL=1 USE_SYSTEM_NCCL=1 \
        UCC_HOME=${HPCX_UCC_DIR} UCX_HOME=${HPCX_UCX_DIR} \
        USE_NCCL_WITH_UCC=1 \
        USE_UCC=1 USE_SYSTEM_UCC=1; fi; } && \
    USE_OPENCV=1 \
    BUILD_TORCH=ON \
    BUILD_TEST=0 \
    CMAKE_PREFIX_PATH="/usr/bin/" \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/lib \
    CUDA_BIN_PATH=/usr/local/cuda/bin \
    CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda/ \
    CUDNN_LIB_DIR=/usr/local/cuda/lib64 \
    CUDA_HOST_COMPILER=cc \
    USE_CUDA=1 \
    USE_NNPACK=1 \
    CC=cc \
    CXX=c++ \
    USE_EIGEN_FOR_BLAS=ON \
    USE_MKL=OFF \
    BUILD_VERSION="${TORCH_VISION_VERSION}" \
    TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    python3 setup.py bdist_wheel --dist-dir ../dist


## Build the final torch image.
FROM ${FINAL_BASE_IMAGE}
ENV DEBIAN_FRONTEND=noninteractive

# Install core packages
RUN apt-get -qq update && apt-get -qq install -y \
      libncurses5 python3 python3-pip python3-distutils python3-numpy \
      curl git apt-utils ssh ca-certificates tmux nano vim sudo bash rsync \
      htop wget unzip tini && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1 && \
    pip3 install --no-cache-dir --upgrade pip && \
    apt-get clean

ARG BUILD_TORCH_VERSION
ARG BUILD_TORCH_VISION_VERSION
ARG BUILD_TORCH_CUDA_ARCH_LIST
ENV TORCH_VERSION=$BUILD_TORCH_VERSION
ENV TORCH_VISION_VERSION=$BUILD_TORCH_VISION_VERSION
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
      libnvtoolsext1 && \
    { if [ $CUDA_MAJOR_VERSION -ge 12 ]; then \
      apt-get -qq install --no-upgrade -y libnvjitlink-${CUDA_PACKAGE_VERSION}; fi; } && \
    { if [ ! -d /opt/nccl-tests ]; then \
      apt-get -qq install --no-upgrade -y libnccl2; fi; } && \
    apt-get clean

WORKDIR /usr/src/app

# Install custom PyTorch wheels.
RUN --mount=type=bind,from=builder,source=/build/dist,target=. \
    pip3 install --no-cache-dir -U ./*.whl