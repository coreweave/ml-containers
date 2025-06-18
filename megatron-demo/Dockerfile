ARG BASE_IMAGE=ghcr.io/coreweave/ml-containers/megatron:es-megatron-tensorizer-5fabc1e
FROM ${BASE_IMAGE}

WORKDIR /usr/src/app/megatron-lm 
COPY sbatch_demo.sh .
COPY srun_demo.sh .

### update sbatch to point to this container
ARG IMAGE_TAG
ENV IMAGE_TAG=${IMAGE_TAG}
RUN sed -i "s|megatron-demo:TAG|megatron-demo:${IMAGE_TAG}|g" /usr/src/app/megatron-lm/sbatch_demo.sh

### Install Git LFS
RUN apt-get update && \
    apt-get install -y git-lfs && \
    git lfs install

### Get tokenizer (with LFS)
RUN mkdir tokenizers && \
    cd tokenizers && \
    git clone https://huggingface.co/NovelAI/nerdstash-tokenizer-v2 && \
    cd nerdstash-tokenizer-v2 && \
    git lfs pull && \
    rm -rf .git

WORKDIR /usr/src/app/megatron-lm

### Get data
RUN mkdir -p /usr/src/app/megatron-lm/coreweave-datasets/smol/tokenized/nerdstash_v2-uint16/
RUN wget -qO- 'https://blobstore.object.ord1.coreweave.com/mldev/datasets/demo-dataset.tbz' \
| tar -C /usr/src/app/megatron-lm/coreweave-datasets/smol/tokenized/nerdstash_v2-uint16/ -xjf -
