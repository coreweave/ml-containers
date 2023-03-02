ARG NCCL_TESTS_VERSION="12.0.1-devel-ubuntu20.04-nccl2.16.5-1-4a46534"
ARG BUILD_TORCH_VERSION="2.0.0-rc1"
ARG BUILD_TORCH_VISION_VERSION="0.15.0-rc1"
ARG BUILD_TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6 8.9 9.0+PTX"
# 8.7 is not supported until PyTorch 2.0.0-rc2

## Build PyTorch on a builder image.
FROM ghcr.io/coreweave/nccl-tests:${NCCL_TESTS_VERSION} as builder
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

RUN mkdir /build
WORKDIR /build

## Build torch
ARG BUILD_TORCH_VERSION
ENV TORCH_VERSION=$BUILD_TORCH_VERSION
RUN git clone --recursive --depth 1 https://github.com/pytorch/pytorch -b v${TORCH_VERSION} && \
    cd pytorch && \
    git submodule sync && \
    git submodule update --init --recursive --jobs 8

RUN cd pytorch && pip3 install -r requirements.txt

ARG BUILD_TORCH_CUDA_ARCH_LIST
ENV TORCH_CUDA_ARCH_LIST=$BUILD_TORCH_CUDA_ARCH_LIST
RUN cd pytorch && \
    mkdir build && \
    ln -s /usr/bin/cc build/cc && \
    ln -s /usr/bin/c++ build/c++ && \
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
    USE_NCCL=1 \
    USE_SYSTEM_NCCL=1 \
    USE_SYSTEM_UCC=1 \
    PYTORCH_BUILD_VERSION="${TORCH_VERSION}" \
    PYTORCH_BUILD_NUMBER=0 \
    TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    python3 setup.py bdist_wheel
RUN cd pytorch && pip3 install --no-cache-dir dist/torch*.whl

## Build torchvision
ARG BUILD_TORCH_VISION_VERSION
ENV TORCH_VISION_VERSION=$BUILD_TORCH_VISION_VERSION
RUN git clone --recursive --depth 1 https://github.com/pytorch/vision -b v${TORCH_VISION_VERSION} && \
    cd vision && \
    git submodule sync && \
    git submodule update --init --recursive --jobs 8

RUN pip3 install --no-cache-dir --upgrade \
    matplotlib numpy typing_extensions requests pillow

RUN cd vision && \
    mkdir build && \
    ln -s /usr/bin/cc build/cc && \
    ln -s /usr/bin/c++ build/c++ && \
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
    USE_NCCL=1 \
    USE_SYSTEM_NCCL=1 \
    USE_SYSTEM_UCC=1 \
    BUILD_VERSION="${TORCH_VISION_VERSION}" \
    TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    python3 setup.py bdist_wheel

RUN mkdir dist && cp pytorch/dist/torch*.whl dist/ && \
                  cp vision/dist/torchvision*.whl dist/

## Build the final torch-nccl image.
FROM ghcr.io/coreweave/nccl-tests:${NCCL_TESTS_VERSION}
ENV DEBIAN_FRONTEND=noninteractive
ARG BUILD_TORCH_VERSION
ARG BUILD_TORCH_VISION_VERSION
ARG BUILD_TORCH_CUDA_ARCH_LIST
ENV TORCH_VERSION=$BUILD_TORCH_VERSION
ENV TORCH_VISION_VERSION=$BUILD_TORCH_VISION_VERSION
ENV TORCH_CUDA_ARCH_LIST=$BUILD_TORCH_CUDA_ARCH_LIST

# Kept only for compatibility with torch-base
ARG BUILD_CUDA_PACKAGE_VERSION="12-0"
ENV CUDA_PACKAGE_VERSION=$BUILD_CUDA_PACKAGE_VERSION

# Install core packages
RUN apt-get -qq update && apt-get -qq install -y \
      libncurses5 python3 python3-pip python3-distutils python3-numpy \
      curl git apt-utils ssh ca-certificates tmux nano vim sudo bash rsync \
      htop wget unzip tini && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1 && \
    pip3 install --no-cache-dir --upgrade pip && \
    apt-get clean

RUN apt-get -qq install -y libnvtoolsext1 && apt-get clean

WORKDIR /usr/src/app

# Install custom PyTorch wheels.
COPY --from=builder /build/dist wheels
RUN pip3 install --no-cache-dir -U wheels/*.whl && rm -r wheels
