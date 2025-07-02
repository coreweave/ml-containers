#!/bin/bash
set -xeo pipefail
export DEBIAN_FRONTEND=noninteractive

TORCH_CUDA_ARCH_LIST=''
FILTER_ARCHES=''

while getopts 'a:ft' OPT; do
  case "${OPT}" in
    a) TORCH_CUDA_ARCH_LIST="${OPTARG}" ;;
    f) FILTER_ARCHES='1' ;;
    *) exit 92 ;;
  esac
done

printf 'Using %s=%s\n' \
  FLASHINFER_COMMIT "${FLASHINFER_COMMIT:-<None>}" \
  CUTLASS_COMMIT "${CUTLASS_COMMIT:-<None>}" \
  VLLM_COMMIT "${VLLM_COMMIT:-<None>}" \
  SGLANG_COMMIT "${SGLANG_COMMIT:-<None>}" \
  DECORD_COMMIT "${DECORD_COMMIT:-<None>}" \
  TRITON_COMMIT "${TRITON_COMMIT:-<None>}"

export NVCC_APPEND_FLAGS='-gencode=arch=compute_100,code=sm_100 -gencode=arch=compute_100a,code=sm_100a -gencode=arch=compute_120,code=[sm_120,compute_120] --diag-suppress 174,177,2361'
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0 10.0 12.0+PTX}"

mkdir -p /wheels/logs

_CLONE() {
  git clone --filter=tree:0  --no-single-branch --no-checkout "${1:?}" "${2:?}" && \
  git -C "${2:?}" checkout "${3:?}" && \
  git -C "${2:?}" submodule update --init --recursive --jobs 8 --depth 1;
}
_BUILD() { python3 -m build -w -n -v -o /wheels "${@:-.}"; }
_LOG() { tee -a "/wheels/logs/${1:?}"; }
_CONSTRAINTS="$(python3 -m pip list | sed -En 's@^(torch(vision|audio)?|vllm)\s+(\S+)$@\1==\3@p')"
_PIP_INSTALL() {
  python3 -m pip install --no-cache-dir \
  --constraint=/dev/stdin <<< "${_CONSTRAINTS}" \
  "$@"
}

_PIP_INSTALL -U pip setuptools wheel build pybind11 ninja 'cmake<4.0.0' 'scikit-build-core>=0.10'

# triton (not compatible with torch 2.6)
if [ -n "${TRITON_COMMIT}" ]; then (
  echo 'Building triton-lang/triton'
  _CLONE https://github.com/triton-lang/triton triton "${TRITON_COMMIT}"
  cd triton
  _BUILD python |& _LOG triton.log
); fi

# flashinfer
: "${FLASHINFER_COMMIT:?}"
(
echo 'Building flashinfer-ai/flashinfer'
_CLONE https://github.com/flashinfer-ai/flashinfer flashinfer "${FLASHINFER_COMMIT}"
cd flashinfer
if [ -n "${CUTLASS_COMMIT}" ]; then
  git -C 3rdparty/cutlass checkout "${CUTLASS_COMMIT}"
fi
_PIP_INSTALL -U optree
python3 -m flashinfer.aot
NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:+$NVCC_APPEND_FLAGS } --diag-suppress 20281,174" \
  _BUILD . \
  |& _LOG flashinfer.log \
  | sed '/^Killed$/{p; Q1}'
)


# vLLM
if [ -n "${VLLM_COMMIT}" ]; then (
  if [ -n "${CUTLASS_COMMIT}" ]; then
    # Setup cutlass repo for vLLM to use
    _CLONE https://github.com/NVIDIA/cutlass cutlass "${CUTLASS_COMMIT}"
    git -C cutlass checkout "${CUTLASS_COMMIT}"
    export VLLM_CUTLASS_SRC_DIR="${PWD}/cutlass"
    test -d "${VLLM_CUTLASS_SRC_DIR}"
  fi

  echo 'Building vllm-project/vllm'
  _CLONE https://github.com/vllm-project/vllm vllm "${VLLM_COMMIT}"
  cd vllm
  # For lsmod
  apt-get -qq update && apt-get -qq install --no-install-recommends -y kmod
  python3 use_existing_torch.py
  _PIP_INSTALL -r requirements-build.txt
  _BUILD . |& _LOG vllm.log
); fi

# sglang
: "${SGLANG_COMMIT:?}"
(
echo 'Building sglang'
_CLONE https://github.com/sgl-project/sglang sglang "${SGLANG_COMMIT}"
cd sglang
(
cd sgl-kernel

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

_BUILD \
  -Cbuild-dir=build \
  -Ccmake.define.SGL_KERNEL_ENABLE_SM100A=1 \
  -Ccmake.define.SGL_KERNEL_ENABLE_SM90A=1 \
  -Ccmake.define.SGL_KERNEL_ENABLE_BF16=1 \
  -Ccmake.define.SGL_KERNEL_ENABLE_FP8=1 \
  -Ccmake.define.SGL_KERNEL_ENABLE_FP4=1 \
  . \
  |& _LOG sglang.log \
  | sed '/^Killed$/{p; Q1}'
)
_BUILD python |& _LOG sglang.log
)

# decord isn't available on PyPI for ARM64

if [ ! "$(uname -m)" = 'x86_64' ]; then
  # decord (for sglang)
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

apt-get clean
