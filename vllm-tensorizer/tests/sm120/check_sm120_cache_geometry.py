# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Exercise the V4.1 sparse cache geometry against the FlashInfer reference."""

import argparse
import json

import torch
from flashinfer.mla._sparse_mla_sm120 import (
    _sparse_mla_sm120_paged_attention as sparse_attention,
)

def _cast_scale_inv_to_ue8m0(scales_inv: torch.Tensor) -> torch.Tensor:
    """Round inverse scale to the nearest power-of-2 (FlashMLA convention)."""
    return torch.pow(2, torch.clamp_min(scales_inv, 1e-4).log2().ceil())


def _fp32_to_ue8m0_bytes(scale_fp32: torch.Tensor) -> torch.Tensor:
    """Extract the IEEE-754 exponent byte of an FP32 power-of-2 scale."""
    bits = scale_fp32.to(torch.float32).view(torch.int32)
    return ((bits >> 23) & 0xFF).to(torch.uint8)


def quantize_kv_dsv4(kv_bf16: torch.Tensor) -> torch.Tensor:
    """Pack bf16 KV into DSv4 FP8 FOOTER format."""
    d_nope, d_rope, tile_size, num_tiles = 448, 64, 64, 7
    data_stride = d_nope + d_rope * 2  # 576
    scale_bytes = num_tiles + 1  # 8
    bpt = data_stride + scale_bytes  # 584
    nb, bs, hk, d = kv_bf16.shape
    assert d == 512 and hk == 1
    kv = kv_bf16.squeeze(2)

    block_bytes = bs * bpt
    result_flat = torch.zeros(nb, block_bytes, dtype=torch.uint8, device=kv.device)

    for ti in range(num_tiles):
        tile = kv[..., ti * tile_size : (ti + 1) * tile_size].float()
        amax = tile.abs().amax(dim=-1).clamp(min=1e-4)
        scale = _cast_scale_inv_to_ue8m0(amax / 448.0)
        fp8 = (tile / scale.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
        ue8m0 = _fp32_to_ue8m0_bytes(scale)

        for tok in range(bs):
            data_off = tok * data_stride + ti * tile_size
            result_flat[:, data_off : data_off + tile_size] = fp8[:, tok].view(
                torch.uint8
            )
            scale_off = bs * data_stride + tok * scale_bytes + ti
            result_flat[:, scale_off] = ue8m0[:, tok]

    rope = kv[..., d_nope:].to(torch.bfloat16).contiguous().view(torch.uint8)
    rope = rope.reshape(nb, bs, d_rope * 2)
    for tok in range(bs):
        rope_off = tok * data_stride + d_nope
        result_flat[:, rope_off : rope_off + d_rope * 2] = rope[:, tok]

    return result_flat.view(nb, bs, 1, bpt)


def dequantize_kv_dsv4(packed: torch.Tensor) -> torch.Tensor:
    """Unpack DSV4 FP8 FOOTER → bf16. Inverse of :func:`quantize_kv_dsv4`."""
    d_nope, d_rope, tile_size, num_tiles = 448, 64, 64, 7
    data_stride = d_nope + d_rope * 2
    scale_bytes = num_tiles + 1
    bpt = data_stride + scale_bytes
    nb, bs, _, _ = packed.shape
    result = torch.zeros(nb, bs, 512, dtype=torch.bfloat16, device=packed.device)
    p = packed.view(nb, bs * bpt)

    for tok in range(bs):
        data_off = tok * data_stride
        scale_off = bs * data_stride + tok * scale_bytes
        for ti in range(num_tiles):
            fp8_off = data_off + ti * tile_size
            fp8 = p[:, fp8_off : fp8_off + tile_size].view(torch.float8_e4m3fn).float()
            ue8m0 = p[:, scale_off + ti]
            scale = torch.pow(2.0, ue8m0.float() - 127.0)
            result[:, tok, ti * tile_size : (ti + 1) * tile_size] = (
                fp8 * scale.unsqueeze(-1)
            ).to(torch.bfloat16)
        rope_off = data_off + d_nope
        rope_bytes = p[:, rope_off : rope_off + d_rope * 2].contiguous()
        result[:, tok, d_nope:] = rope_bytes.view(torch.bfloat16).reshape(nb, d_rope)

    return result.view(nb, bs, 1, 512)


def _ref_sparse_attn(
    q: torch.Tensor,
    kv_dequant: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int,
    attn_sink: torch.Tensor | None = None,
    topk_length: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense SDPA over sparse-gathered KV."""
    num_tokens, num_heads, d_qk = q.shape
    topk = indices.shape[-1]

    kv_flat = kv_dequant.view(-1, d_qk).float()
    q_f = q.float()

    idx_fixed = indices.clamp(min=0)
    invalid = indices < 0
    if topk_length is not None:
        ar = torch.arange(topk, device=q.device).unsqueeze(0)
        invalid = invalid | (ar >= topk_length.unsqueeze(-1))

    gathered = kv_flat.index_select(0, idx_fixed.view(-1)).view(num_tokens, topk, d_qk)
    P = torch.einsum("thd,tkd->thk", q_f, gathered) * sm_scale
    P[invalid.unsqueeze(1).expand_as(P)] = float("-inf")

    lse_e = torch.logsumexp(P, dim=-1)
    lse_safe = lse_e.clone()
    lse_safe[lse_safe == float("-inf")] = float("+inf")
    weights = torch.exp(P - lse_safe.unsqueeze(-1))
    out_f = torch.einsum("thk,tkd->thd", weights, gathered[..., :d_v])

    LN2 = float(torch.log(torch.tensor(2.0)).item())
    lse_log2 = lse_e / LN2

    if attn_sink is not None:
        sink = attn_sink.float()
        sink_log2 = sink / LN2
        factor = torch.sigmoid(lse_e.float() - sink.unsqueeze(0))
        out_f = out_f * factor.unsqueeze(-1)
        lse_log2 = torch.where(
            lse_log2 == float("-inf"),
            sink_log2.unsqueeze(0).expand_as(lse_log2),
            lse_log2 + torch.log2(1.0 + torch.exp2(sink_log2.unsqueeze(0) - lse_log2)),
        )

    return out_f.to(torch.bfloat16), lse_log2


def _make_decode_scratch(
    num_tokens: int,
    num_heads: int,
    topk: int,
    d_v: int,
    device: torch.device,
    *,
    extra_topk: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_splits = (topk + 63) // 64 + (extra_topk + 63) // 64
    return (
        torch.empty(
            (num_tokens, num_heads, num_splits, d_v),
            dtype=torch.bfloat16,
            device=device,
        ),
        torch.empty(
            (num_tokens, num_heads, num_splits),
            dtype=torch.float32,
            device=device,
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--swa-block-size", type=int, default=64)
    parser.add_argument("--extra-block-size", type=int, default=128)
    parser.add_argument("--num-tokens", type=int, default=65)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--padded-pages", action="store_true")
    parser.add_argument("--full-lengths", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(20260910)
    device = torch.device("cuda")
    assert torch.cuda.get_device_capability()[0] == 12
    query_count, heads = args.num_tokens, args.num_heads
    total_main, total_extra = 512, 2048
    main = torch.randn(total_main // args.swa_block_size, args.swa_block_size, 1, 512, device=device, dtype=torch.bfloat16) * 0.5
    extra = torch.randn(total_extra // args.extra_block_size, args.extra_block_size, 1, 512, device=device, dtype=torch.bfloat16) * 0.5
    main_packed = quantize_kv_dsv4(main)
    extra_packed = quantize_kv_dsv4(extra)
    main_dequant = dequantize_kv_dsv4(main_packed)
    extra_dequant = dequantize_kv_dsv4(extra_packed)
    if args.padded_pages:
        def pad_pages(packed):
            page_bytes = packed.shape[1] * 584
            page_stride = (page_bytes + 575) // 576 * 576
            storage = torch.full((packed.shape[0], page_stride), 255, device=device, dtype=torch.uint8)
            view = storage[:, :page_bytes].view(packed.shape)
            view.copy_(packed)
            return view
        main_packed = pad_pages(main_packed)
        extra_packed = pad_pages(extra_packed)
    query = torch.randn(query_count, heads, 512, device=device, dtype=torch.bfloat16) * 0.5
    main_indices = torch.randint(total_main, (query_count, 128), device=device, dtype=torch.int32)
    extra_indices = torch.randint(total_extra, (query_count, 512), device=device, dtype=torch.int32)
    main_lengths = 64 + torch.arange(query_count, device=device, dtype=torch.int32) % 65
    extra_lengths = 256 + torch.arange(query_count, device=device, dtype=torch.int32) % 257
    if args.full_lengths:
        main_lengths.fill_(128)
        extra_lengths.fill_(512)
    main_indices[:, 3] = -1
    extra_indices[:, 11] = -1
    sinks = torch.randn(heads, device=device)
    virtual_kv = torch.cat((main_dequant.reshape(-1, 512), extra_dequant.reshape(-1, 512))).reshape(-1, 1, 1, 512)
    main_reference = main_indices.masked_fill(torch.arange(128, device=device)[None, :] >= main_lengths[:, None], -1)
    extra_reference = extra_indices.masked_fill(torch.arange(512, device=device)[None, :] >= extra_lengths[:, None], -1)
    shifted_extra = torch.where(extra_reference >= 0, extra_reference + total_main, -1)
    virtual_indices = torch.cat((main_reference, shifted_extra), dim=1)
    expected, expected_lse = _ref_sparse_attn(query, virtual_kv, virtual_indices, 512**-0.5, 512, sinks)
    actual = torch.empty_like(expected)
    actual_lse = torch.empty_like(expected_lse)
    mid_out, mid_lse = _make_decode_scratch(query_count, heads, 128, 512, device, extra_topk=512)
    sparse_attention(query, main_packed, main_indices, actual, actual_lse, 512**-0.5,
                     attn_sink=sinks, topk_length=None if args.full_lengths else main_lengths,
                     extra_kv_cache=extra_packed, extra_indices=extra_indices,
                     extra_topk_length=None if args.full_lengths else extra_lengths, mid_out=mid_out, mid_lse=mid_lse)
    torch.cuda.synchronize()
    assert torch.isfinite(actual).all(), "attention output is not finite"
    assert torch.isfinite(actual_lse).all(), "attention LSE is not finite"
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=5e-2)
    torch.testing.assert_close(actual_lse, expected_lse, atol=5e-2, rtol=5e-2)
    error = actual.float() - expected.float()
    relative_rmse = (error.square().mean() / expected.float().square().mean()).sqrt().item()
    assert relative_rmse < 0.05, relative_rmse
    print(json.dumps({"status": "passed", **vars(args), "max_abs_error": error.abs().max().item(), "relative_rmse": relative_rmse}))


if __name__ == "__main__":
    main()
