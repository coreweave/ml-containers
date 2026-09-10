# DeepSeek V4.1 Flash on SM120

This draft stack extends the development image in
[PR 227](https://github.com/coreweave/ml-containers/pull/227).
It is independent of the Dynamo frontend patch stack and does not change
Grace Blackwell deployment configuration.

## Source and build contract

- Parent image branch: `ace/dsv41-flash`, commit
  `a5124aedb9c720b888274de6832bd2bb3d7e2b49`.
- vLLM source: `40e6042ec83eb8f2971f21043a5da40496bd188a`, plus the parent's
  reconciled V4.1 integration from PR 56214 at
  `e47aa780bccf59f59dfa2cbb18e17a10b4fe69ba`.
- FlashInfer source: `8bc3b578027791336c6ae87db5c9d76f82cef8bc`, version
  `0.6.18.post1`.
- The parent development image deliberately includes initial V4.1 runtime
  integration in its compiled wheel. This stack uses that same build path so
  vLLM cache allocation and FlashInfer kernels ship together. It does not add
  these Python changes to the released runtime's separate patch queue.

The first layer selects 64-entry sliding-window pages only on SM120 and
instantiates FlashInfer's existing dual-cache prefill template for 128-entry
extra pages. The JIT module name changes to `sparse_mla_sm120_dsv41` to prevent
loading a stock precompiled module with incompatible dispatch. FlashInfer
patches run before its Python, cubin, and JIT-cache wheels are built.
Other CUDA capability families retain their existing sliding-window page size.

## Earlier validation baseline

Earlier isolated checks passed on an NVIDIA RTX PRO 6000 Blackwell Server
Edition GPU with compute capability 12.0. They used a derived runtime based on
`vllm/vllm-openai:deepseekv41-flash-0909@sha256:00d577a6a63281e15336029d5bcee4e9a2cf182214a4f20ba6111b1c8e79893d`,
vLLM `0.1.dev20904+g179dd0fa9`, FlashInfer `0.6.18`, torch `2.13.0+cu130`,
and CUDA runtime `13.0`. The rebased image instead follows the parent branch's
CUDA `13.2.1` build. Earlier results do not validate that rebuilt image.

The stock image rejected 32-entry sliding-window pages on SM120. After that
page-size correction, stock prefill still rejected `extra_page_block_size=128`.
Eight sparse-attention cases then passed with 8 heads, 64-entry sliding-window
pages, 64/128-entry extra pages, 1/65 query tokens, and full/runtime lengths.
The maximum absolute output error was at most `0.00341796875`; relative RMSE
was below `0.024`. The reference dequantizes the actual packed FP8 operands,
so these results include internal attention precision loss but not initial KV
quantization error.

## Reproduce kernel checks

Run the probe in the built image on an SM120 GPU. It loads no model weights.
The two commands exercise the ratio-1 compressed-cache page in decode and
prefill. Both must print a JSON record with `status` equal to `passed`.

```bash
python3 vllm-tensorizer/tests/sm120/check_sm120_cache_geometry.py --num-tokens 1 --extra-block-size 128 --padded-pages
python3 vllm-tensorizer/tests/sm120/check_sm120_cache_geometry.py --num-tokens 65 --extra-block-size 128 --padded-pages
```

Repeat with `--extra-block-size 64` and with `--full-lengths` to cover the eight
cases. The probe checks finite outputs and log-sum-exp, `atol=0.01, rtol=0.05`
for output, `atol=0.05, rtol=0.05` for log-sum-exp, and relative RMSE below 0.05.

The rebased source patches pass application checks and Python syntax checks.
A new image build and SM120 GPU run are pending. Record the final image digest,
instance type, GPU count, package versions, commands, and results before
marking this stack ready. Kernel checks alone do not prove full-model serving,
DSpark, concurrency, or multimodal support.

## Indexer allocation

The second layer selects logical indexer blocks of `64 * compress_ratio` on
SM120. The indexer backend accepts logical block sizes 64 and 128 there, so
both compression ratios retain 64 physical entries per packed page. Other
capability families retain their existing block-size selection. A packed
128-entry page cannot be split into two 64-entry pages because its FP8 scale
footer belongs to the whole page.

```bash
python3 vllm-tensorizer/tests/sm120/check_sm120_indexer.py
```

The probe uses the real model cache-spec constructor and backend block-size
selection, then runs RMSNorm, RoPE, FP8 storage, DeepGEMM scheduling, and paged
scoring for compression ratios 1 and 2. It checks shuffled physical pages,
unequal context lengths, an empty request, and five deterministic repetitions.
Earlier kernel checks on the documented baseline passed with maximum absolute
error `7.62939453125e-06` and relative RMSE below `7.3e-08`. The revised probe
also verifies that the model and allocator actually choose the tested geometry;
that revision has not yet run on a GPU.
