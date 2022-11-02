FROM coreweave/nccl-tests:2022-09-28_16-34-19.392_EDT

ENV DET_PYTHON_EXECUTABLE="/usr/bin/python3.8"
ENV DET_SKIP_PIP_INSTALL="SKIP"

# Run updates and install packages for build
RUN echo "Dpkg::Options { "--force-confdef"; "--force-confnew"; };" > /etc/apt/apt.conf.d/local
RUN apt-get -qq update && \
    apt-get -qq install -y --no-install-recommends software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    add-apt-repository universe && \
    apt-get -qq update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y curl tzdata build-essential daemontools && \
    apt-get install -y --no-install-recommends \
       python3.8 \
       python3.8-distutils \
       python3.8-dev \
       python3.8-venv \
       git && \
    apt-get clean

# python3.8 -m ensurepip --default-pip && \
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.8 get-pip.py
RUN python3.8 -m pip install --no-cache-dir --upgrade pip

ARG PYTORCH_VERSION=1.12.1
ARG TORCHVISION_VERSION=0.13.1
ARG TORCHAUDIO_VERSION=0.12.1
ARG TORCH_CUDA=116
ARG TORCH_INDEX=whl

RUN python3.8 -m pip install --no-cache-dir install torch==${PYTORCH_VERSION}+cu${TORCH_CUDA} \ 
        torchvision==${TORCHVISION_VERSION}+cu${TORCH_CUDA} \
        torchaudio==${TORCHAUDIO_VERSION}+cu${TORCH_CUDA} \
        --extra-index-url https://download.pytorch.org/${TORCH_INDEX}/cu${TORCH_CUDA}

RUN python3.8 -m pip install --no-cache-dir install packaging

RUN mkdir -p /tmp/build && \
        cd /tmp/build && \
        git clone https://github.com/NVIDIA/apex && \
        cd apex && \
        python3.8 -m pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ && \
        cd /tmp && \
        rm -r /tmp/build

#### Python packages
RUN python3.8 -m pip install --no-cache-dir determined==0.19.2

#### Clone GPT-Neox for requirements
RUN git clone https://github.com/EleutherAI/gpt-neox.git && cd gpt-neox && \
        python3.8 -m pip install --no-cache-dir -r requirements/requirements.txt && \
        python3.8 -m pip install --no-cache-dir -r requirements/requirements-onebitadam.txt && \
        python3.8 -m pip install -r requirements/requirements-sparseattention.txt

RUN python3.8 -m pip install --no-cache-dir pybind11
RUN python3.8 -m pip install --no-cache-dir protobuf==3.19.4
RUN update-alternatives --install /usr/bin/python3 python /usr/bin/python3.8 2
RUN echo 2 | update-alternatives --config python



