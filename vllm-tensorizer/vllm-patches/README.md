# vLLM source patches

These patches are applied to the pinned vLLM source before the wheel and its
native extensions are built. Keep this layer limited to source and dependency
changes that affect compiled artifacts; Python/runtime backports belong in
the downstream runtime image layer.

## GLM-5.3-Flash compiled subset

- vLLM base: `0.29.0rc2` at
  `586f1d6d2da011744e1bae26c8686dc206bf648c`.
- Upstream feature: vLLM PR
  [#53906](https://github.com/vllm-project/vllm/pull/53906), original feature
  commit `933876c388fb129ad82590660e6506614559cb86`.
- FlashInfer is pinned to
  `69ff11fc4954396d98326656dc85debd2223f637`, the immutable commit for
  `v0.6.18`; CUDA remains `13.2.1`.
- The vLLM 0.29 source already has the required FlashAttention revision and
  FlashInfer 0.6.18 dependency metadata. Those obsolete patch sections were
  removed instead of being carried forward redundantly.
- Patch SHA256:
  `1298bf7716156a3dbb282ac717bb0f0158b28f88ecf0b81e50713aebd7bf3ff0`.

The pre-build patch contains exactly one file:

- `csrc/libtorch_stable/cache_kernels.cu` adds compiled `head_dim=512`
  support required by GLM-5.3.

## Compatibility contract

Recheck this patch against every vLLM source bump. Remove it once the pinned
source accepts `head_dim=512` without this delta. Any future native-source or
dependency change requires a new compiled image and digest.
