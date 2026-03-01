#!/bin/bash
set -xeo pipefail
export DEBIAN_FRONTEND=noninteractive

: "${MEGATRON_COMMIT:?}"
: "${SLIME_COMMIT:?}"

_CONSTRAINTS="$(
  python3 -m pip list | sed -En 's@^(torch(vision|audio)?)\s+(\S+)$@\1==\3@p'
)"
_PIP_INSTALL() {
  python3 -m pip install --no-cache-dir \
  --constraint=/dev/stdin <<< "${_CONSTRAINTS}" \
  "$@"
}

# ====== Phase 1: Install compiled wheels ======
# TE 2.10 (overrides base TE 2.4), apex (overrides base apex),
# patched sglang (overrides base sglang), int4_qat kernel
_PIP_INSTALL --force-reinstall --no-deps /wheels/*.whl

# Smoke test: catch NCCL/CUDA loader regressions at build time.
# The base image provides system NCCL via apt — pip nvidia_* wheels must not
# shadow it. If this fails, a transitive dep wheel leaked through build.bash.
python3 -c "import torch; print(f'torch {torch.__version__} OK')"

# ====== Phase 2: Clone + patch + install Megatron-LM (editable) ======
# Megatron-LM is cloned in the final stage (not the builder) because it uses
# editable install (pip install -e) and needs source present at runtime.
git clone --filter=tree:0 --no-single-branch --no-checkout https://github.com/NVIDIA/Megatron-LM /root/Megatron-LM
git -C /root/Megatron-LM checkout "${MEGATRON_COMMIT}"
git -C /root/Megatron-LM submodule update --init --recursive --jobs 8 --depth 1
cd /root/Megatron-LM
git apply /wheels/v0.5.7/megatron.patch --3way
if grep -R -n '^<<<<<<< ' megatron/; then
  echo "Megatron patch failed to apply cleanly" && exit 1
fi
_PIP_INSTALL -e .

# ====== Phase 3: Install pip dependencies ======
# mbridge
_PIP_INSTALL git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c --no-deps

# flash-linear-attention
_PIP_INSTALL flash-linear-attention==0.4.1

# tilelang
_PIP_INSTALL 'tilelang==0.1.8' -f https://tile-ai.github.io/whl/nightly/cu128/

# torch_memory_saver
_PIP_INSTALL git+https://github.com/fzyzcjy/torch_memory_saver.git@dc6876905830430b5054325fa4211ff302169c6b --force-reinstall

# Megatron-Bridge
_PIP_INSTALL --no-build-isolation git+https://github.com/fzyzcjy/Megatron-Bridge.git@35b4ebfc486fb15dcc0273ceea804c3606be948a

# nvidia-modelopt
_PIP_INSTALL --no-build-isolation 'nvidia-modelopt[torch]>=0.37.0'

# cudnn workaround (https://github.com/pytorch/pytorch/issues/168167)
_PIP_INSTALL nvidia-cudnn-cu12==9.16.0.29

# numpy 1.x (required by Megatron)
_PIP_INSTALL 'numpy<2'

# slime runtime dependencies
_PIP_INSTALL -r /wheels/requirements.txt

# ====== Phase 4: Clone + install slime (editable) ======
# Slime uses editable install so source must be present at runtime.
git clone --filter=tree:0 --no-single-branch --no-checkout https://github.com/THUDM/slime /root/slime
git -C /root/slime checkout "${SLIME_COMMIT}"
cd /root/slime
_PIP_INSTALL -e . --no-deps

# ====== Phase 5: Runtime apt packages + cleanup ======
apt-get -qq update && apt-get -qq install --no-install-recommends -y \
  nvtop dnsutils && \
apt-get clean

rm -rf /root/.cache/pip
