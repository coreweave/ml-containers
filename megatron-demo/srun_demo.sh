#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1

export WORLD_SIZE="${SLURM_NTASKS:?}"
export RANK="${SLURM_PROCID:?}"
export LOCAL_RANK="${SLURM_LOCALID:?}"
export CUDA_DEVICE_ORDER='PCI_BUS_ID'

OUTDIR="${OUTDIR:-/mnt/data/test}"
echo "Outputs will be saved to: $OUTDIR"
echo "Set the OUTDIR environment variable to override this location."

CKPTDIR_LOAD="${CKPTDIR_LOAD:-${OUTDIR}/checkpoints}"
CKPTDIR_SAVE="${CKPTDIR_SAVE:-${OUTDIR}/checkpoints}"

mkdir -p "${CKPTDIR_SAVE}"
touch "${CKPTDIR_SAVE}/progress.txt"


cd /usr/src/app/megatron-lm

WARNING_FILTERS=(
'-Wignore::DeprecationWarning'
'-Wignore::FutureWarning'
'-Wignore::UserWarning:megatron.core.tensor_parallel.layers'  # "async_grad_allreduce is deprecated"
'-Wignore::UserWarning:megatron.core.optimizer.distrib_optimizer'  # "pre_hook" method deprecations
)

python3 "${WARNING_FILTERS[@]:?}" \
        "/usr/src/app/megatron-lm/pretrain_gpt.py" \
        --train-iters 1000000 \
        --lr 4e-05 \
        --lr-decay-iters 998000 \
        --lr-decay-style cosine \
        --min-lr 4e-06 \
        --lr-warmup-iters 2000 \
        --clip-grad 1.0 \
        --bf16 \
        --use-flash-attn \
        --rotary-seq-len-interpolation-factor 32 \
        --no-fp8-wgrad \
        --use-distributed-optimizer \
        --distributed-backend nccl \
        --data-cache-path cache \
        --split 949,50,1 \
        --seed 42 \
        --use-checkpoint-args \
        --no-masked-softmax-fusion \
        --attention-softmax-in-fp32 \
        --transformer-impl transformer_engine \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --rotary-base 500000 \
        --rotary-percent 1.0 \
        --use-rope-scaling \
        --micro-batch-size 1 \
        --tensor-model-parallel-size 4 \
        --pipeline-model-parallel-size 1 \
        --context-parallel-size 1 \
        --sequence-parallel \
        --overlap-grad-reduce \
        --overlap-param-gather \
        --log-interval 1 \
        --tensorboard-log-interval 1 \
        --save-interval 3500 \
        --eval-interval 100 \
        --eval-iters 10 \
        --logging-level 20 \
        --log-params-norm \
        --log-num-zeros-in-grad \
        --log-throughput \
        --log-progress \
        --timing-log-level 0 \
        --timing-log-option all \
        --log-timers-to-tensorboard \
        --log-validation-ppl-to-tensorboard \
        --log-memory-to-tensorboard \
        --log-world-size-to-tensorboard \
        --wandb-project test \
        --wandb-save-dir "${OUTDIR}/logs" \
        --tensorboard-dir "${OUTDIR}/tensorboard" \
        --ffn-hidden-size 11008 \
        --num-attention-heads 32 \
        --num-layers 32 \
        --hidden-size 4096 \
        --seq-length 8192 \
        --max-position-embeddings 8192 \
        --untie-embeddings-and-output-weights \
        --normalization RMSNorm \
        --swiglu \
        --position-embedding-type rope \
        --disable-bias-linear \
        --group-query-attention \
        --num-query-groups 8 \
        --tokenizer-type GPTSentencePieceTokenizer \
        --tokenizer-model /usr/src/app/megatron-lm/tokenizers/nerdstash-tokenizer-v2/tokenizer.model \
        --data-path \
          /usr/src/app/megatron-lm/coreweave-datasets/smol/tokenized/nerdstash_v2-uint16/chunk.0 \
          /usr/src/app/megatron-lm/coreweave-datasets/smol/tokenized/nerdstash_v2-uint16/chunk.0 \
        --wandb-exp-name "${SLURM_JOB_ID:?}/test" \
        --load "${CKPTDIR_LOAD}" \
        --save "${CKPTDIR_SAVE}" \
        --dataloader-type cyclic \
