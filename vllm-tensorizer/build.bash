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

_PIP_INSTALL -U pip setuptools wheel build pybind11 ninja cmake setuptools_scm

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

apt-get clean