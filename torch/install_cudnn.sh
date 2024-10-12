#!/bin/sh

CUDA_VERSION="$1";
if [ -z "$CUDA_VERSION" ]; then
    exit 14;
fi;

INSTALL_DEV="$2";
if [ "$INSTALL_DEV" = "dev" ]; then
    echo "Ensuring installation of cuDNN (dev)";
    DEV_SUFFIX="-dev";
    DEV_PREFIX="";
elif [ "$INSTALL_DEV" = "runtime" ]; then
    echo "Ensuring installation of cuDNN (runtime)";
    DEV_SUFFIX="";
    DEV_PREFIX="lib";
else
    exit 15;
fi;

CHECK_VERSION() {
    dpkg-query --status "$1" 2>/dev/null \
    | sed -ne 's/Version: //p' \
    | grep .;
}

CUDA_MAJOR_VERSION=$(echo "$CUDA_VERSION" | cut -d. -f1);
LIBCUDNN_VER="$(
    CHECK_VERSION "libcudnn8${DEV_SUFFIX}" || \
    CHECK_VERSION "libcudnn9${DEV_SUFFIX}-cuda-${CUDA_MAJOR_VERSION}" || \
    :;
)" || exit 16;

if [ -z "$LIBCUDNN_VER" ]; then
    apt-get -qq update && \
    apt-get -qq install --no-upgrade -y "${DEV_PREFIX}cudnn9-cuda-${CUDA_MAJOR_VERSION}" && \
    apt-get clean && \
    ldconfig;
else
    echo "Found cuDNN version ${LIBCUDNN_VER}"
fi;
