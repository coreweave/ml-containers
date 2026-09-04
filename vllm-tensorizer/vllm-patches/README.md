# vLLM source patches

These patches are applied to the pinned vLLM source before the wheel and its
native extensions are built. Keep this layer limited to source and dependency
changes that affect compiled artifacts; Python/runtime backports belong in
the downstream runtime image layer.

## GLM-5.3-Flash compiled subset

- vLLM base: `0.28.0` at
  `2cf0a6915ce544dc493a0990f2ea38d81601128a`.
- Upstream feature: vLLM PR
  [#53906](https://github.com/vllm-project/vllm/pull/53906), original feature
  commit `933876c388fb129ad82590660e6506614559cb86`.
- Reviewed upstream head and compatibility contract:
  `e34f7f71edc7aa46e906326688d1c4f3a6fce8a1`.
- NVIDIA follow-ups included in the resolved backport include
  `031c899dde9729776d8bf12d9f4fcf8de51996f8` and
  `0f82208cb1d01e723bb25b607217a1adfb0cfc30`.
- ROCm-only commit `724f07381c58eb996c78452f96135fe2ca89c525`
  is intentionally excluded from this CUDA image.
- FlashInfer is pinned to
  `e62941a1da605fb9b3c8c50b23c9720df12cf6b4`, the immutable commit for
  `v0.6.18rc10`; CUDA remains `13.2.1`.
- That FlashInfer source already contains the DeepSeek no-group routing fix at
  `08ddfbcd2e89b2f4b68391825817909e30d445e2`, so the former local FlashInfer
  patch is removed instead of being carried forward redundantly.
- The CUDA dependency metadata pins `flashinfer-python` and
  `flashinfer-cubin` to `0.6.18rc10`, matching the locally built wheels and the
  GLM NoPE MLA API contract.
- The FlashInfer provider-wheel build uses CUTLASS DSL `4.6.2`, matching the
  release candidate's provider-build floor and vLLM's runtime pin. The
  inherited global `flash-attn-4` package is removed because vLLM 0.28 bundles
  its own namespaced FA4 implementation and the inherited package pins an
  incompatible CUTLASS DSL and `apache-tvm-ffi` pair.
- Patch SHA256:
  `9c51ffcdadc9d629fadecd57448477032d377f12aacd936e6275e3ca99054389`.

The pre-build patch contains exactly these files:

- `cmake/external_projects/vllm_flash_attn.cmake` pins
  `vllm-project/flash-attention@06bdd47c0d0383daf6a2ff0c418faff9c6da16e5`.
- `csrc/libtorch_stable/cache_kernels.cu` adds compiled `head_dim=512`
  support required by GLM-5.3.
- `requirements/cuda.txt` aligns the vLLM wheel metadata with the installed
  FlashInfer `0.6.18rc10` wheels.

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

`0002-fix-mnnvl-lamport-mailbox.patch` is the native source hunk from upstream
vLLM commit `a047e2543da570a64d1bbfeac4fe44eff3e87a81` ([#53000](https://github.com/vllm-project/vllm/pull/53000)).
It fixes multicast publication and cleanup in
`csrc/custom_all_gather_reduce_scatter.cuh`.

The pinned vLLM `0.28.0` source predates this fix. The first release tag that
contains it is `v0.28.1rc0`, so the patch must remain until the image source is
updated to a revision containing that commit.

Patch SHA256:
`d4e98e4b302ecdaf463120b7ca1b1f567c18525b742ae83d2d9a445c042173e9`.
