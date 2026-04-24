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
  "cuda-python==$(echo "${CUDA_VERSION}" | cut -d. -f1-2)" 'outlines==0.1.11' \
  'llguidance>=0.7.11,<0.8.0' \
  'xgrammar==0.1.32'

# Make PyTorch's shared libs (libc10.so etc.) visible to the dynamic linker
# so that torchao's CUDA extensions can load them at runtime.
python3 -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))" \
  > /etc/ld.so.conf.d/torch.conf
ldconfig
