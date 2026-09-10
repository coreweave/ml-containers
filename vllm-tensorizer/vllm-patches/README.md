# vLLM source patches

This development image advances the vLLM source to immutable upstream main
`40e6042ec83eb8f2971f21043a5da40496bd188a` and completes DeepSeek V4.1 Flash
integration from [PR #56214](https://github.com/vllm-project/vllm/pull/56214),
head `e47aa780bccf59f59dfa2cbb18e17a10b4fe69ba`.

The base already includes the merged native kernels (#56215), tokenizer and
Rust/Python frontend (#56208), and model definitions (#56228). The remaining
patch wires model/quantization registries, DSpark, Engram lookback, cache
accounting and offload, and carries the upstream regression tests. It is
included in the built wheel to keep the initial integration self-contained.
No downstream duplicate of this patch is needed.

Conflict reconciliation preserves the newer main implementations of inverse
RoPE Kernel DSL dispatch, DSML parser framing, GLM parser registration,
ModelOpt gathered projections, and shared-expert padding. Engram validation
accepts both Qwen4 PLE and DeepSeek V4.1 embeddings while preserving Qwen4
parallel settings. The NIXL descriptor fast path excludes scratch regions.

The previous GLM-5.3 compiled patch is removed: upstream main already accepts
512-wide cache heads. FlashInfer advances to `0.6.18.post1`, immutable commit
`8bc3b578027791336c6ae87db5c9d76f82cef8bc`, matching vLLM requirements.
Torch 2.13.0, CUDA 13.2.1, DeepGEMM and DeepEP stay at their existing compatible
pins; the latter two match the pinned vLLM source exactly.

V4.1 requires a rebuilt runtime because the prior released base lacks its
native kernels and Rust frontend. Once this image exists, its remaining
Python integration can be iterated in the downstream infr patch layer.
This branch is a development integration, not evidence of production rollout.

## SM120 development stack

The numbered SM120 patches extend the initial V4.1 development integration.
Keep their cache geometry aligned with `../flashinfer-patches`.
See [the SM120 validation guide](../tests/sm120/README.md) for source pins,
reproduction commands, earlier test evidence, and remaining validation.
