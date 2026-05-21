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

# pip resolves sglang's runtime deps (xgrammar, llguidance, torchao,
# transformers, decord2/torchcodec, flashinfer, …) from PyPI using the pins
# in upstream sglang's pyproject.toml — no explicit overrides needed here.
_PIP_INSTALL /wheels/*.whl

# Make PyTorch's shared libs (libc10.so etc.) visible to the dynamic linker
# so that torchao's CUDA extensions can load them at runtime.
python3 -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))" \
  > /etc/ld.so.conf.d/torch.conf
ldconfig
