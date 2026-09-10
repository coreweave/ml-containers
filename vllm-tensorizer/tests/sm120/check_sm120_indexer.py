import json
import torch
from vllm.models.deepseek_v4_1.common.ops.indexer_k_store import indexer_k_norm_rope_store
from vllm.utils.deep_gemm import fp8_fp4_paged_mqa_logits, get_paged_mqa_logits_metadata
from vllm.config import CacheConfig, VllmConfig, set_current_vllm_config
from vllm.models.deepseek_v4_1.attention import DeepseekV4IndexerCache
from vllm.v1.worker.utils import select_common_block_size

assert torch.cuda.get_device_capability()[0] == 12
config = VllmConfig()
config.cache_config = CacheConfig(block_size=128, cache_dtype="fp8_ds_mla")
torch.manual_seed(20260910)
device = 'cuda'
head_dim, heads, query_count, max_len = 128, 32, 4, 192
context_lens = torch.tensor([[65], [0], [127], [192]], device=device, dtype=torch.int32)
for ratio in (1, 2):
    with set_current_vllm_config(config):
        layer = DeepseekV4IndexerCache(
            head_dim=132,
            dtype=torch.uint8,
            prefix=f"test.indexer.{ratio}",
            cache_config=config.cache_config,
            compress_ratio=ratio,
        )
        spec = layer.get_kv_cache_spec(config)
    kernel_block_size = select_common_block_size(
        spec.block_size, [layer.get_attn_backend()]
    )
    assert spec.block_size == 64 * ratio
    assert kernel_block_size == spec.block_size
    page_size = spec.num_states
    assert page_size == 64
    table_width = (max_len + page_size - 1) // page_size
    pages = query_count * table_width
    storage = torch.zeros((pages, spec.page_size_bytes), device=device, dtype=torch.uint8)
    cache = storage[:, :page_size * 132].view(pages, page_size, 132)
    table = torch.randperm(pages, device=device, dtype=torch.int32).view(query_count, table_width)
    positions = torch.arange(max_len * ratio, device=device, dtype=torch.int64).repeat(query_count)
    logical_keys = positions // ratio
    request_ids = torch.arange(query_count, device=device).repeat_interleave(max_len * ratio)
    physical_pages = table[request_ids, logical_keys // page_size].long()
    slot_mapping = physical_pages * page_size + logical_keys % page_size
    k_pre = torch.randn(positions.numel(), head_dim, device=device, dtype=torch.bfloat16)
    rms_weight = torch.randn(head_dim, device=device, dtype=torch.bfloat16) * 0.5 + 1
    angles = torch.arange(max_len * ratio, device=device)[:, None] * torch.linspace(0.001, 0.1, 32, device=device)[None, :]
    rope = torch.cat((angles.cos(), angles.sin()), dim=-1)
    indexer_k_norm_rope_store(k_pre, positions, rope, rms_weight, 1e-6, cache,
                            slot_mapping, ratio, False)
    kv_values = storage[:, :page_size * head_dim].contiguous().view(torch.float8_e4m3fn).float().view(pages, page_size, head_dim)
    kv_scales = storage[:, page_size * head_dim:page_size * 132].contiguous().view(torch.float32).view(pages, page_size)
    dequantized = kv_values * kv_scales[:, :, None]
    physical_ids = table[:, torch.arange(max_len, device=device) // page_size].long()
    keys = dequantized[physical_ids, torch.arange(max_len, device=device)[None, :] % page_size]
    query = (torch.randn(query_count, 1, heads, head_dim, device=device) * 0.5).to(torch.float8_e4m3fn)
    weights = torch.randn(query_count, heads, device=device)
    reference = torch.einsum('bhd,bkd->bhk', query[:, 0].float(), keys).relu()
    reference = torch.einsum('bhk,bh->bk', reference, weights)
    valid = torch.arange(max_len, device=device)[None, :] < context_lens
    schedule = get_paged_mqa_logits_metadata(context_lens, page_size, torch.cuda.get_device_properties(0).multi_processor_count)
    previous = None
    for repetition in range(5):
        actual = fp8_fp4_paged_mqa_logits((query, None), cache.unsqueeze(2), weights,
                                         context_lens, table, schedule, max_len, False)
        torch.cuda.synchronize()
        actual_valid = actual[valid]
        assert torch.isfinite(actual_valid).all()
        torch.testing.assert_close(actual_valid, reference[valid], atol=0.02, rtol=0.005)
        if previous is not None:
            assert torch.equal(actual_valid, previous)
        previous = actual_valid.clone()
    difference = previous - reference[valid]
    relative_rmse = (difference.square().mean() / reference[valid].square().mean()).sqrt().item()
    assert relative_rmse < 0.005
    print(json.dumps({'status': 'passed', 'compress_ratio': ratio, 'logical_block_size': spec.block_size,
                      'physical_entries': page_size, 'page_stride': spec.page_size_bytes,
                      'max_abs_error': difference.abs().max().item(), 'relative_rmse': relative_rmse,
                      'repeat_bitwise_stable': True}), flush=True)
