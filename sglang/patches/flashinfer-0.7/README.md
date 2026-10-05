# FlashInfer 0.7 compatibility

Apply `0001-flashinfer-0.7-compatibility.patch` only to SGLang
`94602c9c2b7cbdb8efd5c52802dac6a1c180089e`, before building its wheels.
Run `git apply --check` before applying it. The baseline profile applies no patch.

This is the relevant subset of upstream [SGLang PR 40709](https://github.com/sgl-project/sglang/pull/40709),
merged as `eae808903ff571d9e00e65cea8ef03ba9a1b309f`:

- FlashInfer 0.7.0.post1, CUTLASS DSL 4.8.0 and QuACK 0.6.5 requirements.
- Matching requirements in the vendored FlashAttention CuTe package.
- CuTe's replacement for the removed QuACK packed subtraction helper.
- Updated runtime minimum FlashInfer version.
- Draft MLA plan lengths computed from the CSR plan before normal planning.

The backport does not change Torch, SGLang's unrelated source, backend selection,
or the legacy 0.5.17 build. The MLA comment is shortened to the local invariant;
the upstream computation and replay guard are retained.

Package metadata checks are not GPU correctness or performance evidence. The
candidate needs the planned GB300 checks, actual provider selection proof and
same-image producer/consumer measurements before promotion.
