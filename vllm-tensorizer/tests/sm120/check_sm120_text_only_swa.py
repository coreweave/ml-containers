import argparse
import json
from types import SimpleNamespace
import torch
from flashinfer.mla import trtllm_batch_decode_sparse_mla_dsv4
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadataBuilder
from vllm.v1.kv_cache_interface import SlidingWindowMLASpec
from check_sm120_cache_geometry import quantize_kv_dsv4, dequantize_kv_dsv4, _ref_sparse_attn

parser = argparse.ArgumentParser()
parser.add_argument('--num-tokens', type=int, default=16)
parser.add_argument('--vision-enabled', action='store_true')
args = parser.parse_args()
assert torch.cuda.get_device_capability()[0] == 12
torch.manual_seed(20260910)
device = torch.device('cuda')
config = VllmConfig()
config.scheduler_config.max_num_batched_tokens = 128
config.scheduler_config.max_num_seqs = 1
config.model_config = SimpleNamespace(
    hf_config=SimpleNamespace(sliding_window=128, vision_max_n_token=1024,
                              vision_n_layers=8, compress_ratios=[0, 1, 2]),
    max_model_len=262144,
    multimodal_config=SimpleNamespace(language_model_only=not args.vision_enabled),
)
spec = SlidingWindowMLASpec(block_size=64, num_kv_heads=1, head_size=512,
                            dtype=torch.uint8, sliding_window=128,
                            state_content_bytes=584, alignment=576)
with set_current_vllm_config(config):
    builder = DeepseekSparseSWAMetadataBuilder(spec, ['test.swa'], config, device)
    if args.vision_enabled:
        assert builder.max_image_tokens == 1024
        assert builder.prefill_index_width == 1152
        print(json.dumps({'status': 'passed', 'vision_enabled': True, 'index_width': builder.prefill_index_width}))
        raise SystemExit(0)
    tokens = args.num_tokens
    prefix = 144
    seq_len = prefix + tokens
    table = torch.tensor([[3, 1, 0, 2]], device=device, dtype=torch.int32)
    positions = torch.arange(prefix, seq_len, device=device, dtype=torch.int64)
    slots = table[0, positions // 64].long() * 64 + positions % 64
    starts_cpu = torch.tensor([0, tokens], dtype=torch.int32)
    common = CommonAttentionMetadata(
        query_start_loc=starts_cpu.to(device), query_start_loc_cpu=starts_cpu,
        seq_lens=torch.tensor([seq_len], device=device, dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([seq_len], dtype=torch.int32),
        num_reqs=1, num_actual_tokens=tokens, max_query_len=tokens, max_seq_len=seq_len,
        block_table_tensor=table, slot_mapping=slots, positions=positions,
    )
    metadata = builder.build(0, common)
    indices = metadata.prefill_swa_indices
    lengths = metadata.prefill_swa_lens
    print(json.dumps({'stage': 'metadata', 'vision_enabled': False, 'index_width': indices.shape[-1], 'num_tokens': tokens}), flush=True)
    source = torch.randn(4, 64, 1, 512, device=device, dtype=torch.bfloat16) * 0.5
    packed = quantize_kv_dsv4(source)
    dequant = dequantize_kv_dsv4(packed)
    query = torch.randn(tokens, 8, 512, device=device, dtype=torch.bfloat16) * 0.5
    sinks = torch.randn(8, device=device)
    workspace = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=device)
    actual = trtllm_batch_decode_sparse_mla_dsv4(
        query=query, swa_kv_cache=packed, workspace_buffer=workspace,
        sparse_indices=indices, bmm1_scale=512**-0.5, sinks=sinks,
        kv_layout='NHD', swa_topk_lens=lengths,
    )
    torch.cuda.synchronize()
    assert builder.max_image_tokens == 0
    assert indices.shape[-1] == 128
    assert lengths.max().item() <= 128
    expected, _ = _ref_sparse_attn(query, dequant, indices.squeeze(1), 512**-0.5, 512, sinks, lengths)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.05)
    error = actual.float() - expected.float()
    relative_rmse = (error.square().mean() / expected.float().square().mean()).sqrt().item()
    assert relative_rmse < 0.05
    print(json.dumps({'status': 'passed', 'num_tokens': tokens, 'index_width': indices.shape[-1], 'max_abs_error': error.abs().max().item(), 'relative_rmse': relative_rmse}), flush=True)
