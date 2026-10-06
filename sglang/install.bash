#!/bin/bash
set -xeo pipefail
export DEBIAN_FRONTEND=noninteractive

_CONSTRAINTS="$(
  python3 -m pip list | sed -En 's@^(torch(vision|audio)?)\s+(\S+)$@\1==\3@p'
)"
_PIP_INSTALL() {
  if [ "${SGLANG_PACKAGE_PROFILE:-legacy}" = legacy ]; then
    python3 -m pip install --no-cache-dir \
      --constraint=/dev/stdin <<< "${_CONSTRAINTS}" "$@"
  else
    python3 -m pip install --no-cache-dir \
      --constraint=/wheels/constraints.txt "$@"
  fi
}

_INSTALL_WHEELS() {
  if [ "${SGLANG_PACKAGE_PROFILE:-legacy}" = legacy ]; then
    _PIP_INSTALL "$@"
    return
  fi
  local wheel sglang_count=0
  local -a requests=()
  for wheel in "$@"; do
    case "${wheel##*/}" in
      sglang-*.whl)
        requests+=("${wheel}[runai]")
        sglang_count=$((sglang_count + 1))
        ;;
      *) requests+=("${wheel}") ;;
    esac
  done
  if [ "${sglang_count}" -ne 1 ]; then
    echo "Expected exactly one SGLang wheel for its runtime loader extra; found ${sglang_count}" >&2
    return 1
  fi
  _PIP_INSTALL "${requests[@]}"
}

_INSTALL_WHEELS /wheels/*.whl

if [ "${SGLANG_PACKAGE_PROFILE:-legacy}" != legacy ]; then
  python3 -m pip check
  mkdir -p /opt/sglang
  cp /wheels/constraints.txt /opt/sglang/constraints.txt
  cp /wheels/packages.py /opt/sglang/packages.py
  cp /wheels/wheel-info.json /opt/sglang/wheel-info.json
  python3 /wheels/packages.py audit \
    --wheel-info /wheels/wheel-info.json \
    --snapshot /wheels/packages-before.json \
    --constraints /opt/sglang/constraints.txt \
    --output /opt/sglang/build-info.json
  CUDA_VISIBLE_DEVICES='' timeout 120s python3 - <<'PY'
import os

import torch
from flash_attn.cute import flash_attn_func
from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla
from flashinfer.fused_moe import trtllm_fp4_block_scale_moe
from sglang.kernels.ops.attention.flash_attn.cute import utils as sglang_cute_utils

assert callable(flash_attn_func)
assert callable(trtllm_batch_decode_with_kv_cache_mla)
assert callable(trtllm_fp4_block_scale_moe)
assert callable(sglang_cute_utils.ex2_emulation_2)
if os.environ['SGLANG_PACKAGE_PROFILE'] == 'upgrade':
    import cutlass.cute as cute
    assert callable(cute.arch.sub_packed_f32x2)
else:
    import quack.activation
    assert callable(quack.activation.sub_packed_f32x2)
assert not torch.cuda.is_initialized()
print('FlashAttention, FlashInfer and SGLang CuTe API import checks passed; no GPU kernels executed')
PY
fi

# Make PyTorch's shared libs (libc10.so etc.) visible to the dynamic linker
# so that torchao's CUDA extensions can load them at runtime.
python3 -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))" \
  > /etc/ld.so.conf.d/torch.conf
ldconfig

# Exercise sgl-deep-ep's actual NCCL discovery logic without importing the
# package, whose top-level GPU prerequisite check cannot run during image build.
python3 - <<'PY'
import glob
import os
import runpy
from importlib.metadata import PackageNotFoundError, distribution

try:
    deep_ep = distribution("sgl-deep-ep")
except PackageNotFoundError:
    print("sgl-deep-ep is not installed; skipping its NCCL discovery check")
else:
    finder = deep_ep.locate_file("deep_ep/utils/find_pkgs.py")
    find_nccl_root = runpy.run_path(str(finder))["find_nccl_root"]
    root = find_nccl_root()

    assert os.path.realpath(root) == os.path.realpath(os.environ["EP_NCCL_ROOT_DIR"]), root
    assert glob.glob(os.path.join(root, "lib", "libnccl.so*")), root
    assert os.path.isfile(os.path.join(root, "include", "nccl.h")), root
    print("DeepEP NCCL root:", root)
PY

# Compile and exercise the lazy HiCache hash extension during the image build.
# This catches missing C++ headers or libcrypto linkage before request traffic
# reaches the Mamba radix-cache event path.
python3 - <<'PY'
from sglang.srt.mem_cache.cpp_utils.native_hash import get_native_hash

digest = get_native_hash([1, 2, 3], None)
assert len(digest) == 64, digest
PY
