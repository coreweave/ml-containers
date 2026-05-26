#!/bin/bash
set -xeo pipefail
export DEBIAN_FRONTEND=noninteractive

TORCH_CUDA_ARCH_LIST=''

while getopts 'a:' OPT; do
  case "${OPT}" in
    a) TORCH_CUDA_ARCH_LIST="${OPTARG}" ;;
    *) exit 92 ;;
  esac
done

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0 10.0+PTX}"

mkdir -p /wheels/logs

_BUILD() { python3 -m build -w -n -v -o /wheels "${1:-.}"; }
_LOG() { tee -a "/wheels/logs/${1:?}"; }
_CONSTRAINTS="$(python3 -m pip list | sed -En 's@^(torch(vision|audio)?)\s+(\S+)$@\1==\3@p')"
_PIP_INSTALL() {
  python3 -m pip install --no-cache-dir \
  --constraint=/dev/stdin <<< "${_CONSTRAINTS}" \
  "$@"
}

# Python build deps. `setuptools-rust>=1.10` is required for sglang's gRPC
# Rust extension (rust/sglang-grpc) since v0.5.12; we build with `--no-isolation`
# so it must be present in the host environment.
_PIP_INSTALL -U pip setuptools wheel build ninja \
  'scikit-build-core>=0.10' 'setuptools-scm>=8.0' 'setuptools-rust>=1.10'

# protobuf-compiler: needed by tonic-build (via prost-build) when compiling the
# sglang-grpc Rust crate.
apt-get -qq update && apt-get -q install --no-install-recommends -y \
  protobuf-compiler

# Rust toolchain: sglang/rust/sglang-grpc requires edition 2024 (rustc >= 1.85);
# its rust-toolchain.toml pins channel 1.90, which rustup will fetch lazily.
curl --proto '=https' --tlsv1.2 --retry 3 --retry-delay 2 -sSf https://sh.rustup.rs \
  | sh -s -- -y --no-modify-path --profile minimal --default-toolchain 1.90
export PATH="/root/.cargo/bin:${PATH}"

# sglang (includes sgl-kernel)
: "${SGLANG_COMMIT:?}"
(
echo 'Building sglang'
git clone --recursive --filter=blob:none https://github.com/sgl-project/sglang
cd sglang
git checkout "${SGLANG_COMMIT}"

# Build sgl-kernel (scikit-build-core + CMake; deps via FetchContent).
(
cd sgl-kernel
# CMAKE_POLICY_VERSION_MINIMUM=3.5 silences the cmake 4.x breakage on any
# FetchContent sub-project (e.g. dlpack inside mscclpp) that still declares
# cmake_minimum_required(VERSION < 3.5).
_CMAKE_PARALLEL=32
_COMPILE_THREADS=16
[ "$(uname -m)" != 'aarch64' ] || { _CMAKE_PARALLEL=20; _COMPILE_THREADS=10; }
CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DSGL_KERNEL_COMPILE_THREADS=${_COMPILE_THREADS}" \
CMAKE_BUILD_PARALLEL_LEVEL="${_CMAKE_PARALLEL}" \
  python3 -m pip wheel --no-build-isolation --no-deps -v -w /wheels . |& _LOG sglang.log
)

# Build sglang python package (includes setuptools-rust extension sglang-grpc).
_BUILD python |& _LOG sglang.log
)

apt-get clean
