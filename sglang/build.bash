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

_PIP_INSTALL -U pip setuptools wheel build ninja 'scikit-build-core>=0.10' 'setuptools-scm>=8.0'
# Do not install cmake via pip — the base image provides cmake from the Kitware
# PPA (3.x). pip's cmake package installs 4.x which removed backward compat
# with cmake_minimum_required(VERSION < 3.5) used by some sub-project deps.

# sglang (includes sgl-kernel)
: "${SGLANG_COMMIT:?}"
(
echo 'Building sglang'
git clone --recursive --filter=blob:none https://github.com/sgl-project/sglang
cd sglang
git checkout "${SGLANG_COMMIT}"

# Relax exact torch-family version pins to be compatible with the base image
sed -Ei \
  -e 's@"torch==[0-9]+\.[0-9]+\.[0-9]+"@"torch>=2.8.0"@' \
  -e 's@"torchaudio==[0-9]+\.[0-9]+\.[0-9]+"@"torchaudio>=2.8.0"@' \
  -e 's@"torchao==[0-9]+\.[0-9]+\.[0-9]+"@"torchao>=0.9.0"@' \
  -e 's@"torchcodec==[0-9]+\.[0-9]+\.[0-9]+@"torchcodec@' \
  python/pyproject.toml

# Build sgl-kernel (scikit-build-core + CMake; deps via FetchContent)
# Use pip wheel instead of python -m build to skip dep-validation checks that
# fail with --no-isolation when the base image's torch/setuptools versions
# don't satisfy scikit-build-core's internally-declared constraints.
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

# Build sglang python package
_BUILD python |& _LOG sglang.log
)

# decord and xgrammar aren't available on PyPI for ARM64

if [ ! "$(uname -m)" = 'x86_64' ]; then
  # xgrammar (for sglang)
  _PIP_INSTALL nanobind
  (
  git clone --recursive --filter=blob:none -b v0.1.32 https://github.com/mlc-ai/xgrammar
  cd xgrammar
  # Use pip wheel --no-build-isolation --no-deps (same as sgl-kernel) to skip
  # the build-dep version check: xgrammar pins nanobind==2.5.0 exactly but we
  # install whatever is current; scikit-build-core still picks up CMAKE_ARGS.
  CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release -GNinja \
    -Dnanobind_DIR=$(python3 -c 'import nanobind; print(nanobind.cmake_dir())')" \
    python3 -m pip wheel --no-build-isolation --no-deps -v -w /wheels . |& _LOG xgrammar.log
  )

  # decord (for sglang)
  : "${DECORD_COMMIT:?}"
  (
  apt-get -qq update && apt-get -q install --no-install-recommends -y \
    build-essential python3-dev python3-setuptools \
    make cmake ffmpeg \
    libavcodec-dev libavfilter-dev libavformat-dev libavutil-dev
  git clone --recursive --filter=blob:none https://github.com/dmlc/decord
  cd decord
  git checkout "${DECORD_COMMIT}"
  (
  mkdir build && cd build
  cmake -S.. -B. -DUSE_CUDA=0 -DCMAKE_BUILD_TYPE=Release -GNinja |& _LOG decord.log
  cmake --build . |& _LOG decord.log
  cp libdecord.so /wheels/libdecord.so
  )
  cd python
  _BUILD . |& _LOG decord.log
  )
fi

apt-get clean
