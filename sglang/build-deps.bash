#!/bin/bash
set -eo pipefail
export DEBIAN_FRONTEND=noninteractive
_CONSTRAINTS="$(python3 -m pip list | sed -En 's@^(torch(vision|audio)?)\s+(\S+)$@\1==\3@p')"
_PIP_INSTALL() {
  python3 -m pip install --no-cache-dir --constraint=/dev/stdin <<< "${_CONSTRAINTS}" "$@"
}

# Install build dependencies explicitly because wheel builds disable isolation.
_PIP_INSTALL -U pip 'setuptools<82' wheel build ninja 'cmake==3.31.10' \
  'scikit-build-core>=0.10' 'setuptools-scm>=8.0' 'setuptools-rust>=1.11'
# CMake is explicit rather than inherited from torch-extras dependencies.
cmake --version

# protobuf-compiler: needed by tonic-build (via prost-build) when compiling the
# sglang-grpc Rust crate.
# The plain CUDA-base Torch image has runtime libraries and nvcc, but not
# the development libraries inherited from the previous NCCL devel image.
CUDA_PACKAGE_VERSION="$(echo "${CUDA_VERSION}" | cut -d. -f1,2 | tr . -)"
apt-get -qq update && apt-get -q install --no-install-recommends -y \
  protobuf-compiler build-essential \
  "libcublas-dev-${CUDA_PACKAGE_VERSION}" \
  "libcurand-dev-${CUDA_PACKAGE_VERSION}" \
  "libcusparse-dev-${CUDA_PACKAGE_VERSION}" \
  "libcusolver-dev-${CUDA_PACKAGE_VERSION}" \
  "cuda-nvrtc-dev-${CUDA_PACKAGE_VERSION}" \
  "cuda-cupti-dev-${CUDA_PACKAGE_VERSION}" \
  "libnccl-dev=$(dpkg-query -W -f='${Version}' libnccl2)"

# rustup only; --default-toolchain none defers to rust-toolchain.toml on first cargo run.
curl --proto '=https' --tlsv1.2 --retry 3 --retry-delay 2 -sSf https://sh.rustup.rs \
  | sh -s -- -y --no-modify-path --profile minimal --default-toolchain none
export PATH="/root/.cargo/bin:${PATH}"
