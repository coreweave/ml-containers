# SLIME Patches

These patches originate from the [THUDM/SLIME](https://github.com/THUDM/SLIME)
project at commit `7014942c17625732dbe9239c0aa9d8e7d7265ccc`
(`docker/patch/latest/`).

Patches are organized into folders named after the SGLang version they target.

## v0.5.9/megatron.patch

**Source:** `THUDM/SLIME@7014942 docker/patch/latest/megatron.patch`

**Target:** [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) at
commit `3714d81d418c9f1bca4594fc35f9e8289f652862`

**What it does:** Modifies Megatron-LM to support SLIME's RL post-training
workflow. Key changes:

- **Partial checkpoint loading** — skips missing keys instead of raising
  `KeyError`, adds `allow_partial_load=True`. Required because the training
  and rollout models have different parameter sets.
- **Multi-token prediction rework** — handles `None` position_ids, detaches
  embeddings, rewrites activation checkpointing for RL compatibility.
- **MoE routing replay hooks** — injects SLIME's `routing_replay` into
  TopKRouter for MoE routing coordination between training and rollout.
- **MLA YaRN RoPE Triton kernel fixes** — adds `k_dim_ceil` parameter and
  rewrites indexing/masking in fused MLA kernels.
- **INT4 QAT fake quantization** — STE (straight-through estimator) support
  via env vars in TransformerEngine extensions.
- **ReloadableProcessGroup compatibility** — removes `group` parameter from
  P2POp constructors.
- **Misc** — `weights_only=False` for `torch.load`, `trust_remote_code=True`
  for HF tokenizer, post-attention/MLP layernorm config fields.

**To update:** Copy `docker/patch/latest/megatron.patch` from the SLIME repo
at the target commit. Update the folder name and paths in `install.bash`.
Verify the `MEGATRON_COMMIT` ARG in the Dockerfile matches what the patch
expects.

## v0.5.9/sglang.patch

**Source:** `THUDM/SLIME@7014942 docker/patch/latest/sglang.patch`

**Target:** [sgl-project/sglang](https://github.com/sgl-project/sglang) at
v0.5.9 (`bbe9c7eeb520b0a67e92d133dfc137a3688dc7f2`).

**What it does:** Adds SLIME's RL coordination hooks to SGLang's serving
runtime. This enables Megatron-LM training to drive SGLang rollout inference
via Ray, including disaggregated prefill/decode scheduling, KV cache transfer,
and weight synchronization.

**To update:** Copy `docker/patch/latest/sglang.patch` from the SLIME repo
at the target commit. If the upstream patch targets a different SGLang version
than our base image, regenerate against the correct commit and resolve any
conflicts. The patch must target the same SGLang commit used to compile
sgl-kernel in the base sglang image.
