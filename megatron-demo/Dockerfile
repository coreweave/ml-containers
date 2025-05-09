ARG BASE_IMAGE=ghcr.io/coreweave/ml-containers/megatron:es-megatron-tensorizer-5fabc1e
FROM FROM ${BASE_IMAGE}

WORKDIR /usr/src/app/megatron-lm 
COPY sbatch_script.sh .
COPY srun_demo.sh .

