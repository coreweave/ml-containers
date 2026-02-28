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

SGLANG_EXTRA_PIP_DEPENDENCIES=('decord2')
if [ "$(uname -m)" = 'x86_64' ]; then
  SGLANG_EXTRA_PIP_DEPENDENCIES+=('xgrammar>=0.1.10')
fi

_PIP_INSTALL \
  'aiohttp' 'fastapi' \
  'hf_transfer' 'huggingface_hub' 'interegular' 'modelscope' \
  'orjson' 'packaging' 'pillow' 'prometheus-client>=0.20.0' \
  'psutil' 'pydantic' 'python-multipart' 'pyzmq>=25.1.2' \
  'torchao>=0.7.0' 'uvicorn' 'uvloop' \
  'cuda-python' 'outlines==0.1.11' \
  'pybase64' \
  "${SGLANG_EXTRA_PIP_DEPENDENCIES[@]}"
