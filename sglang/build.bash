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
_PIP_INSTALL -U pip 'setuptools<82' wheel build ninja \
  'scikit-build-core>=0.10' 'setuptools-scm>=8.0' 'setuptools-rust>=1.10'

# protobuf-compiler: needed by tonic-build (via prost-build) when compiling the
# sglang-grpc Rust crate.
apt-get -qq update && apt-get -q install --no-install-recommends -y \
  protobuf-compiler

# rustup only; --default-toolchain none defers to rust-toolchain.toml on first cargo run.
curl --proto '=https' --tlsv1.2 --retry 3 --retry-delay 2 -sSf https://sh.rustup.rs \
  | sh -s -- -y --no-modify-path --profile minimal --default-toolchain none
export PATH="/root/.cargo/bin:${PATH}"

# sglang (includes sglang-kernel)
: "${SGLANG_COMMIT:?}"
(
echo 'Building sglang'
git clone --recursive --filter=blob:none https://github.com/sgl-project/sglang
cd sglang
git checkout "${SGLANG_COMMIT}"

# Relax exact torch-family version pins to be compatible with the base image
TORCH_VERSION="$(python3 -c 'import torch; print(torch.__version__.partition("+")[0])')"
sed -Ei \
  -e "s@\"torch==[0-9]+\.[0-9]+\.[0-9]+\"@\"torch>=${TORCH_VERSION}\"@" \
  -e 's@"torchaudio==[0-9]+\.[0-9]+\.[0-9]+"@"torchaudio>=2.11.0"@' \
  -e 's@"torchao==[0-9]+\.[0-9]+\.[0-9]+"@"torchao>=0.17.0"@' \
  -e 's@"torchcodec==[0-9]+\.[0-9]+\.[0-9]+@"torchcodec@' \
  python/pyproject.toml

# Surface the Rust toolchain file at the repo root so rustup's CWD-upward
# walk finds it when setuptools-rust invokes cargo from python/.
# Since v0.5.16 the toolchain file lives at the cargo workspace root (rust/)
# rather than inside rust/sglang-grpc/.
ln -sf rust/rust-toolchain.toml .

# Build the AOT kernel package `sglang-kernel`, which python/pyproject.toml pins
# to an exact version (scikit-build-core + CMake; deps via FetchContent).
# Since v0.5.16 its sources live at python/sglang/kernels/aot/ instead of the
# top-level sgl-kernel/ directory.
(
cd python/sglang/kernels/aot
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

# Build sglang python package. Since v0.5.16 setup.py auto-discovers every
# crate in the rust/ cargo workspace declaring [package.metadata.sglang]
# python-module, so this builds sglang-grpc, sglang-mm and sglang-server (the
# CUDA pyproject.toml sets no [tool.sglang] rust-extensions allowlist).
_BUILD python |& _LOG sglang.log
)

apt-get clean
