#!/bin/bash
set -eo pipefail
export DEBIAN_FRONTEND=noninteractive
export PATH="/root/.cargo/bin:${PATH}"
TORCH_CUDA_ARCH_LIST=''
while getopts 'a:' OPT; do
  case "${OPT}" in
    a) TORCH_CUDA_ARCH_LIST="${OPTARG}" ;;
    *) exit 92 ;;
  esac
done
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0 10.0+PTX}"
_RUN() { local label="$1"; shift; python3 /opt/build_progress.py --label "$label" --log-path "/build-logs/${label}.log" -- "$@"; }
cd /build/sglang
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
  _RUN sglang-kernel python3 -m pip wheel --no-build-isolation --no-deps -v -w /wheels .
)

# Build sglang python package. Since v0.5.16 setup.py auto-discovers every
# crate in the rust/ cargo workspace declaring [package.metadata.sglang]
# python-module, so this builds sglang-grpc, sglang-mm and sglang-server (the
# CUDA pyproject.toml sets no [tool.sglang] rust-extensions allowlist).
_RUN sglang-python python3 -m build -w -n -v -o /wheels python
if [ "${SGLANG_PACKAGE_PROFILE}" != legacy ]; then
  FLASHINFER_ARCHITECTURES="$(cat /wheels/flashinfer-architectures.txt)"
  _RUN package-wheel-audit python3 /build/packages.py wheels \
    --profile "${SGLANG_PACKAGE_PROFILE}" \
    --sglang-source /build/sglang --flashinfer-source /build/flashinfer \
    --wheel-dir /wheels --architectures "${FLASHINFER_ARCHITECTURES}" \
    --snapshot /wheels/packages-before.json --constraints /wheels/constraints.txt \
    --cuobjdump /usr/local/cuda/bin/cuobjdump --output /wheels/wheel-info.json
fi

apt-get clean
