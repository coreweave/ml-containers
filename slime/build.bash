#!/bin/bash
set -xeo pipefail
export DEBIAN_FRONTEND=noninteractive

export NVCC_APPEND_FLAGS='-gencode=arch=compute_100,code=[sm_100,compute_100] -gencode=arch=compute_100a,code=sm_100a --diag-suppress 174'

mkdir -p /wheels/logs

_BUILD() { python3 -m build -w -n -v -o /wheels "${1:-.}"; }
_LOG() { tee -a "/wheels/logs/${1:?}"; }
_CONSTRAINTS="$(python3 -m pip list | sed -En 's@^(torch(vision|audio)?)\s+(\S+)$@\1==\3@p')"
_PIP_INSTALL() {
  python3 -m pip install --no-cache-dir \
  --constraint=/dev/stdin <<< "${_CONSTRAINTS}" \
  "$@"
}

_PIP_INSTALL -U pip setuptools wheel build pybind11 ninja cmake 'setuptools-scm>=8.0'

# transformer_engine 2.10.0 (overrides TE 2.4 from base)
(
echo 'Building transformer_engine 2.10.0'
NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS} --threads 4" \
  pip -v wheel --no-build-isolation --wheel-dir /wheels \
  "transformer_engine[pytorch]==2.10.0" |& _LOG transformer_engine.log
)

# apex
: "${APEX_COMMIT:?}"
(
echo "Building NVIDIA/apex @ ${APEX_COMMIT}"
git clone --filter=blob:none https://github.com/NVIDIA/apex
cd apex
git checkout "${APEX_COMMIT}"
NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS} --threads 4" \
  pip -v wheel --no-build-isolation --no-deps --wheel-dir /wheels \
  --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" \
  . |& _LOG apex.log
)

# patched sglang (Python-only, no CUDA recompilation)
: "${SGLANG_COMMIT:?}"
(
echo "Building patched sglang @ ${SGLANG_COMMIT}"
git clone https://github.com/sgl-project/sglang
cd sglang
git checkout "${SGLANG_COMMIT}"
git apply /build/sglang.patch --3way
if grep -R -n '^<<<<<<< ' python/; then
  echo "sglang patch failed to apply cleanly" && exit 1
fi
# Relax torch pin to allow the base image's torch version
TORCH_VER="$(python3 -c 'import torch; print(torch.__version__.split("+")[0])')"
sed -i "s/\"torch>=2\.5\.1\"/\"torch>=${TORCH_VER}\"/" python/pyproject.toml
sed -i "s/\"torch==2\.9\.1\"/\"torch>=${TORCH_VER}\"/" python/pyproject.toml
_BUILD python |& _LOG sglang-patched.log
)

# int4_qat kernel (CUDA extension from slime source)
: "${SLIME_COMMIT:?}"
(
echo "Building slime int4_qat kernel @ ${SLIME_COMMIT}"
git clone --filter=blob:none https://github.com/THUDM/slime
cd slime
git checkout "${SLIME_COMMIT}"
cd slime/backends/megatron_utils/kernels/int4_qat
pip -v wheel --no-build-isolation --no-deps --wheel-dir /wheels . |& _LOG int4_qat.log
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
