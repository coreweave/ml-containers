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
  APEX_COMMIT "${APEX_COMMIT:-<None>}" \
  SGLANG_COMMIT "${SGLANG_COMMIT:-<None>}" \
  SLIME_COMMIT "${SLIME_COMMIT:-<None>}"

export NVCC_APPEND_FLAGS='--diag-suppress 174,177,2361'
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0 8.6 8.9 9.0a 10.0a 12.0+PTX}"

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

_PIP_INSTALL -U pip setuptools wheel build pybind11 ninja 'cmake<4.0.0' 'setuptools-scm>=8.0'

# transformer_engine 2.10.0 (overrides TE 2.4 from base)
(
echo 'Building transformer_engine 2.10.0'
NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS} --threads 4" \
  python3 -m pip wheel -v --no-cache-dir --no-build-isolation --wheel-dir /wheels \
  "transformer_engine[pytorch]==2.10.0" \
  |& _LOG transformer_engine.log \
  | sed '/^Killed$/{p; Q1}'
)

# apex
: "${APEX_COMMIT:?}"
(
echo "Building NVIDIA/apex @ ${APEX_COMMIT}"
_CLONE https://github.com/NVIDIA/apex apex "${APEX_COMMIT}"
cd apex
NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS} --threads 4" \
  python3 -m pip wheel -v --no-cache-dir --no-build-isolation --no-deps --wheel-dir /wheels \
  --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" \
  . \
  |& _LOG apex.log \
  | sed '/^Killed$/{p; Q1}'
)

# patched sglang (Python-only, no CUDA recompilation)
: "${SGLANG_COMMIT:?}"
(
echo "Building patched sglang @ ${SGLANG_COMMIT}"
_CLONE https://github.com/sgl-project/sglang sglang "${SGLANG_COMMIT}"
cd sglang
git apply /build/v0.5.7/sglang.patch --3way
if grep -R -n '^<<<<<<< ' python/; then
  echo "sglang patch failed to apply cleanly" && exit 1
fi
# Relax torch pin to allow the base image's torch version
TORCH_VER="$(python3 -c 'import torch; print(torch.__version__.split("+")[0])')"
grep -q '"torch>=2\.5\.1"\|"torch==2\.9\.1"' python/pyproject.toml || { echo "ERROR: torch pin changed upstream; update sed patterns in build.bash"; exit 1; }
sed -i "s/\"torch>=2\.5\.1\"/\"torch>=${TORCH_VER}\"/" python/pyproject.toml
sed -i "s/\"torch==2\.9\.1\"/\"torch>=${TORCH_VER}\"/" python/pyproject.toml
_BUILD python |& _LOG sglang-patched.log
)

# int4_qat kernel (CUDA extension from slime source)
: "${SLIME_COMMIT:?}"
(
echo "Building slime int4_qat kernel @ ${SLIME_COMMIT}"
_CLONE https://github.com/THUDM/slime slime "${SLIME_COMMIT}"
cd slime/slime/backends/megatron_utils/kernels/int4_qat
python3 -m pip wheel -v --no-cache-dir --no-build-isolation --no-deps --wheel-dir /wheels . |& _LOG int4_qat.log
)

# Remove dependency wheels that conflict with system/base-image binaries.
# The base image provides CUDA libs (NCCL, cuBLAS, cuDNN, etc.) via apt and
# torch/triton via compiled wheels — pip nvidia_* packages shadow these with
# older versions (e.g. nvidia-nccl-cu12==2.27.5 missing ncclAlltoAll).
# Pure Python deps (onnxscript, einops, pydantic, etc.) are harmless to keep.
rm -vf /wheels/nvidia_*.whl /wheels/torch-*.whl /wheels/torchvision-*.whl \
       /wheels/torchaudio-*.whl /wheels/triton-*.whl
ls /wheels/*.whl

apt-get clean
