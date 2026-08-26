#!/bin/bash
set -xeo pipefail
export DEBIAN_FRONTEND=noninteractive

_CONSTRAINTS="$(
  python3 -m pip list | sed -En 's@^(torch(vision|audio)?)\s+(\S+)$@\1==\3@p'
)"
_PIP_INSTALL() {
  python3 -m pip install --no-cache-dir \
  --constraint=/dev/stdin <<< "${_CONSTRAINTS}" \
  "$@"
}

_PIP_INSTALL /wheels/*.whl

# Make PyTorch's shared libs (libc10.so etc.) visible to the dynamic linker
# so that torchao's CUDA extensions can load them at runtime.
python3 -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))" \
  > /etc/ld.so.conf.d/torch.conf
ldconfig

# Compile and exercise the lazy HiCache hash extension during the image build.
# This catches missing C++ headers or libcrypto linkage before request traffic
# reaches the Mamba radix-cache event path. Keep the build-host-specific module
# out of the final image so runtime compilation targets the serving host CPU.
TORCH_EXTENSIONS_DIR="$(mktemp -d)"
export TORCH_EXTENSIONS_DIR
trap 'rm -rf "${TORCH_EXTENSIONS_DIR}"' EXIT

python3 - <<'PY'
from sglang.srt.mem_cache.cpp_utils.native_hash import get_native_hash

digest = get_native_hash([1, 2, 3], None)
assert len(digest) == 64, digest
PY
