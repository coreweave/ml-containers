#!/bin/bash
set -eo pipefail
export DEBIAN_FRONTEND=noninteractive
export PATH="/root/.cargo/bin:${PATH}"
case "${SGLANG_PACKAGE_PROFILE:-legacy}" in
  legacy) [ -z "${FLASHINFER_COMMIT:-}" ]; exit ;;
  baseline) : "${FLASHINFER_COMMIT:?}" ;;
  *) echo "Unknown package profile: ${SGLANG_PACKAGE_PROFILE}" >&2; exit 1 ;;
esac

TORCH_CUDA_ARCH_LIST="${BUILD_TORCH_CUDA_ARCH_LIST:-8.0 8.6 8.9 9.0 10.0+PTX}"
[ "$(uname -m)" != aarch64 ] || TORCH_CUDA_ARCH_LIST='9.0 10.0+PTX'
PHASE=all
WHEEL_DIR=/wheels
while getopts 'a:p:o:' OPT; do
  case "${OPT}" in
    a) TORCH_CUDA_ARCH_LIST="${OPTARG}" ;;
    p) PHASE="${OPTARG}" ;;
    o) WHEEL_DIR="${OPTARG}" ;;
    *) exit 92 ;;
  esac
done
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0 10.0+PTX}"
_RUN() { local label="$1"; shift; python3 /opt/build_progress.py --label "$label" --log-path "/build-logs/${label}.log" -- "$@"; }
_BUILD() { _RUN "$1" python3 -m build -w -n -v -o "${3:-$WHEEL_DIR}" "${2:-.}"; }
case "$PHASE" in
  all|dependencies|python|cubin|shim|provider-*) ;;
  *) echo "Unknown FlashInfer build phase: $PHASE" >&2; exit 92 ;;
esac
if [ "$PHASE" = all ] || [ "$PHASE" = dependencies ]; then
  _RUN flashinfer-dependencies python3 -m pip install --no-cache-dir --constraint=/wheels/constraints.txt -r /wheels/build-requirements.txt
  [ "$PHASE" != dependencies ] || exit 0
fi
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
if [ "$PHASE" = all ] || [ "$PHASE" = python ]; then
  _BUILD flashinfer-python .
fi
if [ "$PHASE" = all ] || [ "$PHASE" = cubin ]; then
  _BUILD flashinfer-cubin flashinfer-cubin
fi
# FlashInfer 0.7 ships a shim plus one native provider per CUDA target.
export FLASHINFER_JIT_CACHE_PROVIDER_ARCHS="$FLASHINFER_CUDA_ARCH_LIST"
if [[ "$PHASE" = provider-* ]]; then
  case " ${FLASHINFER_CUDA_ARCH_LIST} " in
    *" ${PHASE#provider-} "*) ;;
    *)
      # The three SM8x stages are intentionally empty on arm64.
      case "$(uname -m):${PHASE#provider-}" in
        aarch64:8.0|aarch64:8.6|aarch64:8.9) exit 0 ;;
        *) echo "Unsupported provider phase: $PHASE" >&2; exit 92 ;;
      esac ;;
  esac
fi
for provider_arch in $FLASHINFER_CUDA_ARCH_LIST; do
  [ "$PHASE" = all ] || [ "$PHASE" = "provider-${provider_arch}" ] || continue
  # Each provider shares a source package directory; discard the previous
  # provider's staged payload and setuptools output before selecting a new SM.
  rm -rf flashinfer-jit-cache-provider/build \
    flashinfer-jit-cache-provider/flashinfer_jit_cache_provider/jit_cache
  export FLASHINFER_JIT_CACHE_PROVIDER_ARCH="$provider_arch"
  _BUILD "flashinfer-provider-${provider_arch}" flashinfer-jit-cache-provider
done
if [ "$PHASE" = all ] || [ "$PHASE" = shim ]; then
  _BUILD flashinfer-jit-cache flashinfer-jit-cache
fi
