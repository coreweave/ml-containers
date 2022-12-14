FROM coreweave/nccl-tests:2022-11-06_19-21-22.11_EST

# setup python and conda

RUN DEBIAN_FRONTEND=noninteractive apt-get -qq update && \
        DEBIAN_FRONTEND=noninteractive apt-get -qq install -y --no-install-recommends \
        python3 python3-dev python3-pip git libssl-dev pkg-config

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh

ENV PATH=/opt/conda/bin:$PATH

# create conda environment from file. this step installs torch.
COPY ./environment.yaml /opt/nccl-tests/environment.yaml
RUN conda env create -f environment.yaml

SHELL ["conda", "run", "-n", "tr11-176B-ml", "/bin/bash", "-c"]

# setup rust and then tokenizers
RUN conda install -y -c conda-forge rust

RUN git clone https://github.com/huggingface/tokenizers && \
    cd tokenizers && \
    git checkout bigscience_fork && \
    pip install setuptools_rust && \
    pip install -e bindings/python

# install apex
RUN git clone https://github.com/NVIDIA/apex && \
    cd apex && \
    pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ --upgrade

# install deepspeed from here:
# https://github.com/microsoft/DeepSpeed/tree/olruwase/elastic-ckpt-refresh
RUN pip install git+https://github.com/microsoft/DeepSpeed.git@olruwase/elastic-ckpt-refresh --upgrade

# clone bigscience repo
RUN git clone https://github.com/bigscience-workshop/bigscience.git

# https://github.com/bigscience-workshop/Megatron-DeepSpeed/tree/olruwase/ds_ckpt_reshape
RUN git clone --single-branch --branch olruwase/ds_ckpt_reshape https://github.com/bigscience-workshop/Megatron-DeepSpeed.git
