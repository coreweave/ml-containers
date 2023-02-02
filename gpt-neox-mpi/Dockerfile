FROM ghcr.io/coreweave/nccl-tests:11.7.1-devel-ubuntu20.04-nccl2.14.3-1-a0cb1a6

ENV DEBIAN_FRONTEND=noninteractive

#### System package (uses default Python 3 version in Ubuntu 20.04)
RUN apt-get update -y && \
    apt-get install -y \
        git python3 python3-dev libpython3-dev python3-pip pdsh && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1 && \
    pip install --upgrade pip && \
    pip install gpustat

#### Python packages
RUN pip install torch==1.13.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install packaging>=14.0 && pip cache purge

## Install APEX
ARG APEX_COMMIT=537424d24d55e3a166c930828e4780549edc6151
RUN pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" \
        --global-option="--cuda_ext" git+https://github.com/NVIDIA/apex.git@${APEX_COMMIT}

# Get the gpt-neox source code
WORKDIR /
RUN git clone https://github.com/EleutherAI/gpt-neox.git

# Use the-eye.eu instead of the dead mystic.the-eye.eu mirror for dataset links
RUN sed -i 's/mystic.the-eye/the-eye/g' /gpt-neox/tools/corpora.py

RUN pip install -r /gpt-neox/requirements/requirements.txt && \
    pip install -r /gpt-neox/requirements/requirements-onebitadam.txt && \
    pip install -r /gpt-neox/requirements/requirements-sparseattention.txt && \
    pip install protobuf==3.20.* && \
    pip install git+https://github.com/EleutherAI/best-download.git && \
    pip cache purge

RUN python /gpt-neox/megatron/fused_kernels/setup.py install

# Clear staging
RUN mkdir -p /tmp && chmod 0777 /tmp

WORKDIR /gpt-neox
