#!/bin/bash
set -xeo pipefail
export DEBIAN_FRONTEND=noninteractive

TORCH_CUDA_ARCH_LIST=''
SGLANG_PACKAGE_PROFILE="${SGLANG_PACKAGE_PROFILE:-legacy}"
case "${SGLANG_PACKAGE_PROFILE}" in
  legacy) [ -z "${FLASHINFER_COMMIT:-}" ] || { echo 'Legacy builds cannot select FlashInfer source' >&2; exit 1; } ;;
  baseline|upgrade) : "${FLASHINFER_COMMIT:?}" ;;
  *) echo "Unknown package profile: ${SGLANG_PACKAGE_PROFILE}" >&2; exit 1 ;;
esac

while getopts 'a:' OPT; do
  case "${OPT}" in
    a) TORCH_CUDA_ARCH_LIST="${OPTARG}" ;;
    *) exit 92 ;;
  esac
done

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0 10.0+PTX}"

mkdir -p /wheels/logs

_BUILD() { python3 -m build -w -n -v -o "${2:-/wheels}" "${1:-.}"; }
_LOG() { tee -a "/wheels/logs/${1:?}"; }
_CONSTRAINTS="$(python3 -m pip list | sed -En 's@^(torch(vision|audio)?)\s+(\S+)$@\1==\3@p')"
_PIP_INSTALL() {
  if [ -f /wheels/constraints.txt ]; then
    python3 -m pip install --no-cache-dir --constraint=/wheels/constraints.txt "$@"
  else
    python3 -m pip install --no-cache-dir \
      --constraint=/dev/stdin <<< "${_CONSTRAINTS}" "$@"
  fi
}

# Install build dependencies explicitly because wheel builds disable isolation.
_PIP_INSTALL -U pip 'setuptools<82' wheel build ninja \
  'scikit-build-core>=0.10' 'setuptools-scm>=8.0' 'setuptools-rust>=1.11'

# protobuf-compiler: needed by tonic-build (via prost-build) when compiling the
# sglang-grpc Rust crate.
apt-get -qq update && apt-get -q install --no-install-recommends -y \
  protobuf-compiler

# rustup only; --default-toolchain none defers to rust-toolchain.toml on first cargo run.
curl --proto '=https' --tlsv1.2 --retry 3 --retry-delay 2 -sSf https://sh.rustup.rs \
  | sh -s -- -y --no-modify-path --profile minimal --default-toolchain none
export PATH="/root/.cargo/bin:${PATH}"

# sglang (includes sglang-kernel)
: "${SGLANG_COMMIT:?}"
(
echo 'Building sglang'
git clone --recursive --filter=blob:none https://github.com/sgl-project/sglang
cd sglang
git checkout "${SGLANG_COMMIT}"

if [ "${SGLANG_PACKAGE_PROFILE}" != legacy ]; then
  [ "$(git rev-parse HEAD)" = "${SGLANG_COMMIT}" ]
  git submodule update --init --recursive --jobs 8
  if [ "${SGLANG_PACKAGE_PROFILE}" = upgrade ]; then
    for PATCH in /build/patches/flashinfer-0.7/*.patch; do
      git apply --check "${PATCH}"
      git apply "${PATCH}"
    done
  fi
  git clone --filter=blob:none --no-checkout https://github.com/flashinfer-ai/flashinfer /build/flashinfer
  git -C /build/flashinfer checkout "${FLASHINFER_COMMIT}"
  [ "$(git -C /build/flashinfer rev-parse HEAD)" = "${FLASHINFER_COMMIT}" ]
  git -C /build/flashinfer submodule update --init --recursive --jobs 8
fi

# Relax exact torch-family version pins to be compatible with the base image
TORCH_VERSION="$(python3 -c 'import torch; print(torch.__version__.partition("+")[0])')"
sed -Ei \
  -e "s@\"torch==[0-9]+\.[0-9]+\.[0-9]+\"@\"torch>=${TORCH_VERSION}\"@" \
  -e 's@"torchaudio==[0-9]+\.[0-9]+\.[0-9]+"@"torchaudio>=2.11.0"@' \
  -e 's@"torchao==[0-9]+\.[0-9]+\.[0-9]+"@"torchao>=0.17.0"@' \
  -e 's@"torchcodec==[0-9]+\.[0-9]+\.[0-9]+@"torchcodec@' \
  python/pyproject.toml

if [ "${SGLANG_PACKAGE_PROFILE}" != legacy ]; then
  python3 /build/packages.py constraints \
    --profile "${SGLANG_PACKAGE_PROFILE}" \
    --sglang-source /build/sglang --flashinfer-source /build/flashinfer \
    --output /wheels/constraints.txt --requirements /wheels/build-requirements.txt \
    --snapshot /wheels/packages-before.json
  _PIP_INSTALL -r /wheels/build-requirements.txt
  (
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
    _BUILD . |& _LOG flashinfer.log
    _BUILD flashinfer-cubin |& _LOG flashinfer.log
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
          FLASHINFER_JIT_CACHE_PROVIDER_ARCH="${PROVIDER_ARCH}" _BUILD . "${PROVIDER_OUTPUT}"
        ) |& _LOG flashinfer.log
        python3 scripts/verify_jit_cache_provider_artifact.py \
          --artifact-dir "${PROVIDER_OUTPUT}" --provider "${PROVIDER_TAG}" \
          --version "$(cat version.txt)+${FLASHINFER_LOCAL_VERSION}" \
          --provider-platform-tag "linux_$(uname -m)" \
          --cuobjdump /usr/local/cuda/bin/cuobjdump --cuda-architecture-policy strict \
          |& _LOG flashinfer.log
        mv "${PROVIDER_OUTPUT}/provider-validation.json" "/wheels/provider-validation/${PROVIDER_TAG}.json"
        mv "${PROVIDER_OUTPUT}/"*.whl /wheels/
      done
      FLASHINFER_JIT_CACHE_PROVIDER_ARCHS="${FLASHINFER_CUDA_ARCH_LIST}" \
        _BUILD flashinfer-jit-cache |& _LOG flashinfer.log
    else
      _BUILD flashinfer-jit-cache |& _LOG flashinfer.log
    fi
  )
fi

# Surface the Rust toolchain file at the repo root so rustup's CWD-upward
# walk finds it when setuptools-rust invokes cargo from python/.
# Since v0.5.16 the toolchain file lives at the cargo workspace root (rust/)
# rather than inside rust/sglang-grpc/.
ln -sf rust/rust-toolchain.toml .

# Build the AOT kernel package `sglang-kernel`, which python/pyproject.toml pins
# to an exact version (scikit-build-core + CMake; deps via FetchContent).
# Since v0.5.16 its sources live at python/sglang/kernels/aot/ instead of the
# top-level sgl-kernel/ directory.
(
cd python/sglang/kernels/aot
# CMAKE_POLICY_VERSION_MINIMUM=3.5 silences the cmake 4.x breakage on any
# FetchContent sub-project (e.g. dlpack inside mscclpp) that still declares
# cmake_minimum_required(VERSION < 3.5).
_CMAKE_PARALLEL=32
_COMPILE_THREADS=16
[ "$(uname -m)" != 'aarch64' ] || { _CMAKE_PARALLEL=20; _COMPILE_THREADS=10; }
CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DSGL_KERNEL_COMPILE_THREADS=${_COMPILE_THREADS}" \
CMAKE_BUILD_PARALLEL_LEVEL="${_CMAKE_PARALLEL}" \
  python3 -m pip wheel --no-build-isolation --no-deps -v -w /wheels . |& _LOG sglang.log
)

# Build sglang python package. Since v0.5.16 setup.py auto-discovers every
# crate in the rust/ cargo workspace declaring [package.metadata.sglang]
# python-module, so this builds sglang-grpc, sglang-mm and sglang-server (the
# CUDA pyproject.toml sets no [tool.sglang] rust-extensions allowlist).
_BUILD python |& _LOG sglang.log
if [ "${SGLANG_PACKAGE_PROFILE}" != legacy ]; then
  FLASHINFER_ARCHITECTURES='8.0 8.6 8.9 9.0a 10.0a 10.3a'
  [ "$(uname -m)" != aarch64 ] || FLASHINFER_ARCHITECTURES='9.0a 10.0a 10.3a'
  python3 /build/packages.py wheels \
    --profile "${SGLANG_PACKAGE_PROFILE}" \
    --sglang-source /build/sglang --flashinfer-source /build/flashinfer \
    --wheel-dir /wheels --architectures "${FLASHINFER_ARCHITECTURES}" \
    --snapshot /wheels/packages-before.json --constraints /wheels/constraints.txt \
    --cuobjdump /usr/local/cuda/bin/cuobjdump --output /wheels/wheel-info.json
fi
)

apt-get clean
