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

printf 'Using %s=%s\n' \
  FLASHINFER_COMMIT "${FLASHINFER_COMMIT:-<None>}" \
  SGLANG_COMMIT "${SGLANG_COMMIT:-<None>}"

export NVCC_APPEND_FLAGS='--diag-suppress 174,177,2361'
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0a 10.0a 12.0+PTX}"

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
_PIP_INSTALL -U optree 'apache-tvm-ffi>=0.1.5,<0.2' requests pynvml nvidia-nvshmem-cu12
# Convert TORCH_CUDA_ARCH_LIST to FlashInfer's format:
#   - Add 'a' suffix to >=9.0 arches for architecture-specific instructions
#   - Strip +PTX (FlashInfer AOT only generates native SASS, not PTX)
# See vllm-tensorizer/Dockerfile for the reference pattern.
FLASHINFER_ARCH_LIST="$(echo "${TORCH_CUDA_ARCH_LIST}" | sed -E 's@\b(9|10|12)\.0\b@\1.0a@g; s@\+PTX\b@@g' | xargs)"
FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_ARCH_LIST}" python3 -m flashinfer.aot
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
# Remove torch/torchaudio pins — the base image provides these.
sed -i -E '/torch(audio)?[><=~]/d' python/pyproject.toml
_BUILD python |& _LOG sglang.log
)

apt-get clean
