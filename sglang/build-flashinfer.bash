#!/bin/bash
set -eo pipefail
export DEBIAN_FRONTEND=noninteractive
export PATH="/root/.cargo/bin:${PATH}"
case "${SGLANG_PACKAGE_PROFILE:-legacy}" in
  legacy) [ -z "${FLASHINFER_COMMIT:-}" ]; exit ;;
  baseline|upgrade) : "${FLASHINFER_COMMIT:?}" ;;
  *) echo "Unknown package profile: ${SGLANG_PACKAGE_PROFILE}" >&2; exit 1 ;;
esac

TORCH_CUDA_ARCH_LIST=''
while getopts 'a:' OPT; do
  case "${OPT}" in
    a) TORCH_CUDA_ARCH_LIST="${OPTARG}" ;;
    *) exit 92 ;;
  esac
done
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0 10.0+PTX}"
_RUN() { local label="$1"; shift; python3 /opt/build_progress.py --label "$label" --log-path "/build-logs/${label}.log" -- "$@"; }
_BUILD() { _RUN "$1" python3 -m build -w -n -v -o "${3:-/wheels}" "${2:-.}"; }
_RUN flashinfer-dependencies python3 -m pip install --no-cache-dir --constraint=/wheels/constraints.txt -r /wheels/build-requirements.txt
cd /build/flashinfer
export FLASHINFER_BUILD_NO_PIP=1 BUILD_NIXL_EP=0
export CUDA_HOME=/usr/local/cuda CUDA_PATH=/usr/local/cuda
export FLASHINFER_NVCC=/opt/nvcc-wrapper.py
export MAX_JOBS="${MAX_JOBS:-8}" FLASHINFER_NVCC_THREADS="${FLASHINFER_NVCC_THREADS:-4}"
CUDA_RELEASE="$(/usr/local/cuda/bin/nvcc --version | sed -nE 's/.*release ([0-9]+)\.([0-9]+),.*/\1\2/p')"
[ -n "${CUDA_RELEASE}" ]
export FLASHINFER_LOCAL_VERSION="cu${CUDA_RELEASE}"
FLASHINFER_CUDA_ARCH_LIST='8.0 8.6 8.9 9.0a 10.0a 10.3a'
[ "$(uname -m)" != aarch64 ] || FLASHINFER_CUDA_ARCH_LIST='9.0a 10.0a 10.3a'
export FLASHINFER_CUDA_ARCH_LIST
printf '%s\n' "$FLASHINFER_CUDA_ARCH_LIST" > /wheels/flashinfer-architectures.txt
_BUILD flashinfer-python .
_BUILD flashinfer-cubin flashinfer-cubin
if [ "${SGLANG_PACKAGE_PROFILE}" = upgrade ]; then
  mkdir -p /wheels/provider-validation
  for PROVIDER_ARCH in ${FLASHINFER_CUDA_ARCH_LIST}; do
    PROVIDER_TAG="sm${PROVIDER_ARCH//./}"
    PROVIDER_OUTPUT="/wheels/providers/${PROVIDER_TAG}"
    mkdir -p "${PROVIDER_OUTPUT}"
    (
      cd flashinfer-jit-cache-provider
      rm -rf build ./*.egg-info flashinfer_jit_cache_provider/jit_cache
      rm -f flashinfer_jit_cache_provider/manifest.json flashinfer_jit_cache_provider/_build_meta.py
      FLASHINFER_JIT_CACHE_PROVIDER_ARCH="${PROVIDER_ARCH}" _BUILD "flashinfer-provider-${PROVIDER_TAG}" . "${PROVIDER_OUTPUT}"
    )
    _RUN "flashinfer-verify-${PROVIDER_TAG}" python3 scripts/verify_jit_cache_provider_artifact.py \
      --artifact-dir "${PROVIDER_OUTPUT}" --provider "${PROVIDER_TAG}" \
      --version "$(cat version.txt)+${FLASHINFER_LOCAL_VERSION}" \
      --provider-platform-tag "linux_$(uname -m)" \
      --cuobjdump /usr/local/cuda/bin/cuobjdump --cuda-architecture-policy strict
    mv "${PROVIDER_OUTPUT}/provider-validation.json" "/wheels/provider-validation/${PROVIDER_TAG}.json"
    mv "${PROVIDER_OUTPUT}/"*.whl /wheels/
  done
  FLASHINFER_JIT_CACHE_PROVIDER_ARCHS="${FLASHINFER_CUDA_ARCH_LIST}" \
    _BUILD flashinfer-jit-cache flashinfer-jit-cache
else
  _BUILD flashinfer-jit-cache flashinfer-jit-cache
fi
