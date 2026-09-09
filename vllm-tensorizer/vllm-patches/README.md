# vLLM source patches

These patches are applied to the pinned vLLM source before the wheel and its
native extensions are built. Keep this layer limited to source and dependency
changes that affect compiled artifacts; Python/runtime backports belong in
the downstream runtime image layer.

## GLM-5.3-Flash compiled subset

- vLLM base: `0.29.0` at
  `74c96922ecb9017f413318c76d1af83aa2ab45a5`.
- Upstream feature: vLLM PR
  [#53906](https://github.com/vllm-project/vllm/pull/53906), original feature
  commit `933876c388fb129ad82590660e6506614559cb86`.
- Final upstream head and compatibility contract:
  `4500c80c080328dfe62435d083f4063e00d987df` (merged as
  `98ed0856f31fa3aaf5e27464e2b4ef5a8ee6b2f5`).
- NVIDIA follow-ups included in the resolved backport include
  `031c899dde9729776d8bf12d9f4fcf8de51996f8` and
  `0f82208cb1d01e723bb25b607217a1adfb0cfc30`.
- ROCm-only commit `724f07381c58eb996c78452f96135fe2ca89c525`
  is intentionally excluded from this CUDA image.
- FlashInfer is pinned to
  `69ff11fc4954396d98326656dc85debd2223f637`, the immutable commit for
  `v0.6.18`; CUDA remains `13.2.1`.
- That FlashInfer source already contains the DeepSeek no-group routing fix at
  `08ddfbcd2e89b2f4b68391825817909e30d445e2`, so the former local FlashInfer
  patch is removed instead of being carried forward redundantly.
- The CUDA dependency metadata pins `flashinfer-python` and
  `flashinfer-cubin` to `0.6.18`, matching the locally built wheels and the GLM
  NoPE MLA API contract.
- The FlashInfer provider-wheel build uses CUTLASS DSL `4.6.2`, matching the
  release's provider-build floor and vLLM's runtime pin. The
  inherited global `flash-attn-4` package is removed because vLLM 0.29 bundles
  its own namespaced FA4 implementation and the inherited package pins an
  incompatible CUTLASS DSL and `apache-tvm-ffi` pair.
- Patch SHA256:
  `fb38f52c981be5ecf6b327c450c22d084e9df3ce343add75814dad2154b491dc`.

The pre-build patch contains exactly these files:

- `csrc/libtorch_stable/cache_kernels.cu` adds compiled `head_dim=512`
  support required by GLM-5.3.

vLLM 0.29 already pins
`vllm-project/flash-attention@06bdd47c0d0383daf6a2ff0c418faff9c6da16e5`
and the stable FlashInfer `0.6.18` packages, so the patch no longer carries
those changes.

The remaining resolved PR files are Python/runtime source under `vllm/` and are
intentionally excluded from this compiled image. The separate two-file
narrow-block-table safety fix is also runtime-only and is intentionally
excluded.

## Compatibility contract

The downstream runtime patch must use the same vLLM base and reviewed upstream
head recorded above. A future upstream delta that changes only Python/runtime
source updates only the downstream layer. A delta that changes CMake,
CUDA/C++, native bindings, native dependency revisions, or the FlashInfer
requirement requires a new compiled image and digest.

## MNNVL Lamport mailbox fix

vLLM 0.29 contains upstream commit
`a047e2543da570a64d1bbfeac4fe44eff3e87a81` ([#53000](https://github.com/vllm-project/vllm/pull/53000)),
so the former downstream patch is no longer needed.
