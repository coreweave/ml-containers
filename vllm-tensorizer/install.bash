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
