#!/bin/bash
set -xeo pipefail
export DEBIAN_FRONTEND=noninteractive

TORCH_CUDA_ARCH_LIST=''
FILTER_ARCHES=''
BUILD_TRITON=''

while getopts 'a:ft' OPT; do
  case "${OPT}" in
    a) TORCH_CUDA_ARCH_LIST="${OPTARG}" ;;
    f) FILTER_ARCHES='1' ;;
    t) BUILD_TRITON='1' ;;
    *) exit 92 ;;
  esac
done

export NVCC_APPEND_FLAGS='-gencode=arch=compute_100,code=[sm_100,compute_100] -gencode=arch=compute_100a,code=sm_100a --diag-suppress 174'
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

_PIP_INSTALL -U pip setuptools wheel build pybind11 ninja cmake

# triton (not compatible with torch 2.6)
if [ "${BUILD_TRITON}" = 1 ]; then (
  : "${TRITON_COMMIT:?}"
  echo 'Building triton-lang/triton'
  git clone --recursive --filter=blob:none https://github.com/triton-lang/triton
  cd triton
  git checkout "${TRITON_COMMIT}"
  _BUILD python |& _LOG triton.log
); fi

# flashinfer
: "${FLASHINFER_COMMIT:?}"
: "${CUTLASS_COMMIT:?}"
(
echo 'Building flashinfer-ai/flashinfer'
git clone --recursive --filter=blob:none https://github.com/flashinfer-ai/flashinfer
cd flashinfer
git checkout "${FLASHINFER_COMMIT}"
sed -i 's/name = "flashinfer-python"/name = "flashinfer"/' pyproject.toml
git -C 3rdparty/cutlass checkout "${CUTLASS_COMMIT}"
_PIP_INSTALL -U optree
NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:+$NVCC_APPEND_FLAGS } --diag-suppress 20281,174" \
  FLASHINFER_ENABLE_AOT=1 _BUILD . |& _LOG flashinfer.log
)

# Setup cutlass repo for vLLM to use
git clone --recursive --filter=blob:none https://github.com/NVIDIA/cutlass
git -C cutlass checkout "${CUTLASS_COMMIT}"

# vLLM
: "${VLLM_COMMIT:?}"
(
echo 'Building vllm-project/vllm'
export VLLM_CUTLASS_SRC_DIR="${PWD}/cutlass"
test -d "${VLLM_CUTLASS_SRC_DIR}"
git clone --recursive --filter=blob:none https://github.com/vllm-project/vllm
cd vllm
git checkout "${VLLM_COMMIT}"
# For lsmod
apt-get -qq update && apt-get -qq install --no-install-recommends -y kmod
python3 use_existing_torch.py
_PIP_INSTALL -r requirements-build.txt
USE_CUDNN=1 USE_CUSPARSELT=1 _BUILD . |& _LOG vllm.log
)

# sglang
: "${SGLANG_COMMIT:?}"
(
echo 'Building sglang'
git clone --recursive --filter=blob:none https://github.com/sgl-project/sglang
cd sglang
git checkout "${SGLANG_COMMIT}"
(
cd sgl-kernel
git -C 3rdparty/cutlass checkout "${CUTLASS_COMMIT}"
git -C 3rdparty/flashinfer/3rdparty/cutlass checkout "${CUTLASS_COMMIT}"

ARCH_TRIPLE="$(gcc -print-multiarch)"
LIB_DIR="/usr/lib/${ARCH_TRIPLE:?}"
test -d "${LIB_DIR:?}"
PYTHON_API_VER="$(
  python3 --version | sed -En 's@Python ([0-9])\.([0-9]+)\..*@cp\1\2@p'
)"
ARCH_FILTER=()
if [ "${FILTER_ARCHES}" = 1 ]; then
  ARCH_FILTER=(-e 's@"-gencode=arch=compute_[78][0-9],code=sm_[78][0-9]",@#\0@')
fi

sed -Ei \
  "${ARCH_FILTER[@]}" \
  -e 's@/usr/lib/x86_64-linux-gnu@'"${LIB_DIR}"'@' \
  -e 's@(\s+)(\w.+manylinux2014_x86_64.+)@\1pass  # \2@' \
  -e 's@\{"py_limited_api": "cp39"}@{"py_limited_api": "'"${PYTHON_API_VER:-cp310}"'"}@' \
  setup.py
SGL_KERNEL_ENABLE_BF16=1 SGL_KERNEL_ENABLE_FP8=1 SGL_KERNEL_ENABLE_SM90A=1 \
  _BUILD . |& _LOG sglang.log
)
_BUILD python |& _LOG sglang.log
)

# decord and xgrammar aren't available on PyPI for ARM64

if [ ! "$(uname -m)" = 'x86_64' ]; then
  # xgrammar (for sglang)
  (
  git clone --recursive --filter=blob:none -b v0.1.11 https://github.com/mlc-ai/xgrammar && \
  cd xgrammar
  (
  mkdir build && cd build
  cmake -S.. -B. -DCMAKE_BUILD_TYPE=Release -GNinja |& _LOG xgrammar.log
  cmake --build . |& _LOG xgrammar.log
  )
  _BUILD python |& _LOG xgrammar.log
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
