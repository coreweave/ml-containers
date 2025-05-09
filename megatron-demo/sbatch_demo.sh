#!/bin/bash

#SBATCH --partition h100
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 8
#SBATCH --gpus-per-node 8
#SBATCH --constraint gpu
#SBATCH --job-name test
#SBATCH --output test.%j
#SBATCH --export all
#SBATCH --exclusive

export NCCL_SOCKET_IFNAME=eth0
export SHARP_COLL_ENABLE_PCI_RELAXED_ORDERING=1
export NCCL_COLLNET_ENABLE=0
export NCCL_IB_HCA=ibp
export UCX_NET_DEVICES=ibp0:1,ibp1:1,ibp2:1,ibp3:1,ibp4:1,ibp5:1,ibp6:1,ibp7:1

export MASTER_PORT="$(expr 10000 + "$(echo -n "${SLURM_JOB_ID:?}" | tail -c 4)")"
export MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST:?}" | head -n 1)"


CPU_BIND='map_ldom:0,0,0,0,1,1,1,1'

CONTAINER_IMAGE="ghcr.io#coreweave/ml-containers/megatron-demo:TAG"

srun --container-image "${CONTAINER_IMAGE}" \
     --container-mounts /mnt/data:/mnt/data,/mnt/home:/mnt/home \
     --export=ALL \
     --mpi=pmix \
     --kill-on-bad-exit=1 \
     ${CPU_BIND:+"--cpu-bind=$CPU_BIND"} \
     bash -c ". /usr/src/app/megatron-lm/srun_demo.sh"
