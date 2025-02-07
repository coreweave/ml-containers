#!/bin/bash
set -xeo pipefail

_CONSTRAINTS="$(
  python3 -m pip list | sed -En 's@^(torch(vision|audio)?)\s+(\S+)$@\1==\3@p'
)"
_PIP_INSTALL() {
  python3 -m pip install --no-cache-dir \
  --constraint=/dev/stdin <<< "${_CONSTRAINTS}" \
  "$@"
}

_PIP_INSTALL /wheels/*.whl
if [ -x /wheels/libdecord.so ]; then
  apt-get -qq update && apt-get -q install --no-install-recommends \
    libavfilter7 libavformat58 && \
  apt-get clean
  cp /wheels/libdecord.so /usr/local/lib/ && ldconfig
fi

SGLANG_EXTRA_PIP_DEPENDENCIES=()
if [ "$(uname -m)" = 'x86_64' ]; then
  SGLANG_EXTRA_PIP_DEPENDENCIES=('decord' 'xgrammar>=0.1.10')
fi
_PIP_INSTALL \
  'aiohttp' 'fastapi' \
  'hf_transfer' 'huggingface_hub' 'interegular' 'modelscope' \
  'orjson' 'packaging' 'pillow' 'prometheus-client>=0.20.0' \
  'psutil' 'pydantic' 'python-multipart' 'pyzmq>=25.1.2' \
  'torchao>=0.7.0' 'uvicorn' 'uvloop' \
  'cuda-python' 'outlines>=0.0.44,<0.1.0' \
  "${SGLANG_EXTRA_PIP_DEPENDENCIES[@]}"
