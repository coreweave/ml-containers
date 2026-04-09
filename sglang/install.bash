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
if [ -x /wheels/libdecord.so ]; then
  apt-get -qq update && apt-get -q install --no-install-recommends -y \
    libavfilter7 libavformat58 && \
  apt-get clean
  cp /wheels/libdecord.so /usr/local/lib/ && ldconfig
fi

_PIP_INSTALL \
  'aiohttp' 'fastapi' \
  'hf_transfer' 'huggingface_hub' 'interegular' 'modelscope' \
  'msgspec' 'orjson' 'packaging' 'pillow' 'prometheus-client>=0.20.0' \
  'psutil' 'pydantic' 'python-multipart' 'pyzmq>=25.1.2' \
  'torchao>=0.9.0' 'uvicorn' 'uvloop' \
  'cuda-python==12.9' 'outlines==0.1.11' \
  'llguidance>=0.7.11,<0.8.0' \
  'xgrammar==0.1.32'
