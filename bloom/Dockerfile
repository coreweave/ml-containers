FROM coreweave/nccl-tests:2022-11-06_19-21-22.11_EST

# setup python and conda

RUN apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip git libssl-dev pkg-config

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh

ENV PATH=/opt/conda/bin:$PATH

RUN conda create -y -n tr11-176B-ml python=3.8

SHELL ["conda", "run", "-n", "tr11-176B-ml", "/bin/bash", "-c"]

RUN pip install transformers && \
    pip install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/test/cu115/torch_test.html -U

# setup rust and then tokenizers
RUN conda install -y -c conda-forge rust

RUN git clone https://github.com/huggingface/tokenizers && \
    cd tokenizers && \
    git checkout bigscience_fork && \
    pip install setuptools_rust && \
    pip install -e bindings/python

# install only the dependencies for megatron-deepspeed from here:
# https://github.com/bigscience-workshop/Megatron-DeepSpeed/tree/olruwase/ds_ckpt_reshape
RUN git clone --single-branch --branch olruwase/ds_ckpt_reshape https://github.com/bigscience-workshop/Megatron-DeepSpeed.git && \
    cd Megatron-DeepSpeed && \
    pip install -r requirements.txt

# install apex
RUN git clone https://github.com/NVIDIA/apex && \
    cd apex && \
    pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ --upgrade

# install deepspeed from here:
# https://github.com/microsoft/DeepSpeed/tree/olruwase/elastic-ckpt-refresh
RUN pip install git+https://github.com/microsoft/DeepSpeed.git@olruwase/elastic-ckpt-refresh --upgrade
