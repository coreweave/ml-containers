#!/bin/bash
# Match vLLM: use published FlashInfer wheels and the CUDA-specific provider shim.
set -eo pipefail
export DEBIAN_FRONTEND=noninteractive
case "${SGLANG_PACKAGE_PROFILE:-legacy}" in
  legacy) [ -z "${FLASHINFER_VERSION:-}" ]; exit ;;
  baseline) : "${FLASHINFER_VERSION:?}" "${FLASHINFER_CUDA_LABEL:?}" ;;
  *) exit 1 ;;
esac
PHASE=all
WHEEL_DIR=/wheels
while getopts 'p:o:' OPT; do
  case "$OPT" in
    p) PHASE="$OPTARG" ;;
    o) WHEEL_DIR="$OPTARG" ;;
    *) exit 92 ;;
  esac
done
case "$PHASE" in
  all|dependencies|python|cubin|cache) ;;
  *) echo "Unknown FlashInfer download phase: $PHASE" >&2; exit 92 ;;
esac
_RUN() { local label="$1"; shift; python3 /opt/build_progress.py --label "$label" --log-path "/build-logs/${label}.log" -- "$@"; }
if [ "$PHASE" = all ] || [ "$PHASE" = dependencies ]; then
  _RUN flashinfer-dependencies python3 -m pip install --no-cache-dir --constraint=/wheels/constraints.txt -r /wheels/build-requirements.txt
  [ "$PHASE" != dependencies ] || exit 0
fi
if [ "$PHASE" = all ] || [ "$PHASE" = python ]; then
  _RUN flashinfer-python python3 -m pip download --only-binary=:all: --no-deps \
    --dest "$WHEEL_DIR" --constraint /wheels/constraints.txt \
    --extra-index-url https://flashinfer.ai/whl/ "flashinfer-python==${FLASHINFER_VERSION}"
fi
if [ "$PHASE" = all ] || [ "$PHASE" = cubin ]; then
  _RUN flashinfer-cubin python3 -m pip download --only-binary=:all: --no-deps \
    --dest "$WHEEL_DIR" --constraint /wheels/constraints.txt \
    --extra-index-url https://flashinfer.ai/whl/ "flashinfer-cubin==${FLASHINFER_VERSION}"
fi
if [ "$PHASE" = all ] || [ "$PHASE" = cache ]; then
  _RUN flashinfer-jit-cache python3 -m pip download --only-binary=:all: \
    --dest "$WHEEL_DIR" --constraint /wheels/constraints.txt \
    --index-url "https://flashinfer.ai/whl/${FLASHINFER_CUDA_LABEL}" \
    "flashinfer-jit-cache==${FLASHINFER_VERSION}+${FLASHINFER_CUDA_LABEL}"
  TARGETS='8.0 8.9 9.0a 10.0a 10.3a'
  [ "$(uname -m)" != aarch64 ] || TARGETS='9.0a 10.0a 10.3a'
  printf '%s\n' "$TARGETS" > /wheels/flashinfer-architectures.txt
fi
