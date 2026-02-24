#!/bin/bash
set -xeo pipefail
export DEBIAN_FRONTEND=noninteractive

TORCH_CUDA_ARCH_LIST=''
FILTER_ARCHES=''

while getopts 'a:f' OPT; do
  case "${OPT}" in
    a) TORCH_CUDA_ARCH_LIST="${OPTARG}" ;;
    f) FILTER_ARCHES='1' ;;
    *) exit 92 ;;
  esac
done

printf 'Using %s=%s\n' \
  FLASHINFER_COMMIT "${FLASHINFER_COMMIT:-<None>}" \
  SGLANG_COMMIT "${SGLANG_COMMIT:-<None>}" \
  DECORD_COMMIT "${DECORD_COMMIT:-<None>}"

export NVCC_APPEND_FLAGS='--diag-suppress 174,177,2361'
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0 10.0 12.0+PTX}"

mkdir -p /wheels/logs

_CLONE() {
  git clone --filter=tree:0 --no-single-branch --no-checkout "${1:?}" "${2:?}" && \
  git -C "${2:?}" checkout "${3:?}" && \
  git -C "${2:?}" submodule update --init --recursive --jobs 8 --depth 1;
}
_BUILD() { python3 -m build -w -n -v -o /wheels "${@:-.}"; }
_LOG() { tee -a "/wheels/logs/${1:?}"; }
_CONSTRAINTS="$(python3 -m pip list | sed -En 's@^(torch(vision|audio)?)\s+(\S+)$@\1==\3@p')"
_PIP_INSTALL() {
  python3 -m pip install --no-cache-dir \
  --constraint=/dev/stdin <<< "${_CONSTRAINTS}" \
  "$@"
}

_PIP_INSTALL -U pip setuptools wheel build pybind11 ninja 'cmake<4.0.0' 'scikit-build-core>=0.10' 'setuptools-scm>=8.0'

# flashinfer
: "${FLASHINFER_COMMIT:?}"
(
echo "Building flashinfer-ai/flashinfer @ ${FLASHINFER_COMMIT}"
_CLONE https://github.com/flashinfer-ai/flashinfer flashinfer "${FLASHINFER_COMMIT}"
cd flashinfer
# flashinfer v0.6+ uses TVM for AOT kernel compilation
_PIP_INSTALL -U optree 'apache-tvm-ffi>=0.1.5,<0.2' requests
FLASHINFER_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" python3 -m flashinfer.aot
NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:+$NVCC_APPEND_FLAGS } --diag-suppress 20281,174" \
  _BUILD . \
  |& _LOG flashinfer.log \
  | sed '/^Killed$/{p; Q1}'
)

# sglang (sgl-kernel CUDA extension + Python package)
: "${SGLANG_COMMIT:?}"
(
echo "Building sgl-project/sglang @ ${SGLANG_COMMIT}"
_CLONE https://github.com/sgl-project/sglang sglang "${SGLANG_COMMIT}"
cd sglang

# sgl-kernel (CUDA extension, compiled via scikit-build-core/CMake)
(
cd sgl-kernel

ARCH_TRIPLE="$(gcc -print-multiarch)"
LIB_DIR="/usr/lib/${ARCH_TRIPLE:?}"
test -d "${LIB_DIR:?}"

_BUILD \
  -Cbuild-dir=build \
  -Ccmake.define.SGL_KERNEL_ENABLE_SM100A=1 \
  -Ccmake.define.SGL_KERNEL_ENABLE_SM90A=1 \
  -Ccmake.define.SGL_KERNEL_ENABLE_BF16=1 \
  -Ccmake.define.SGL_KERNEL_ENABLE_FP8=1 \
  -Ccmake.define.SGL_KERNEL_ENABLE_FP4=1 \
  . \
  |& _LOG sgl-kernel.log \
  | sed '/^Killed$/{p; Q1}'
)

# sglang Python package (no CUDA compilation)
# Relax torch pin to allow the base image's torch version
TORCH_VER="$(python3 -c 'import torch; print(torch.__version__.split("+")[0])')"
sed -i "s/\"torch==2\.9\.1\"/\"torch>=${TORCH_VER}\"/" python/pyproject.toml
sed -i "s/\"torchaudio==2\.9\.1\"/\"torchaudio>=${TORCH_VER}\"/" python/pyproject.toml
_BUILD python |& _LOG sglang.log
)

# decord isn't available on PyPI for ARM64
if [ ! "$(uname -m)" = 'x86_64' ]; then
  : "${DECORD_COMMIT:?}"
  (
  apt-get -qq update && apt-get -q install --no-install-recommends -y \
    build-essential python3-dev python3-setuptools \
    make cmake ffmpeg \
    libavcodec-dev libavfilter-dev libavformat-dev libavutil-dev
  _CLONE https://github.com/dmlc/decord decord "${DECORD_COMMIT}"
  cd decord
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

# Remove dependency wheels that pip may have downloaded into /wheels/
# Only our compiled wheels (flashinfer, sgl-kernel, sglang, decord) should remain.
rm -vf /wheels/torch-*.whl /wheels/torchvision-*.whl /wheels/torchaudio-*.whl

apt-get clean
