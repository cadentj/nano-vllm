import math
import unittest

import torch

from nanovllm.kernel import paged_flash_attn_decode, paged_flash_attn_varlen


def make_paged_kvcache(seqlens_k, num_kv_heads, head_dim, block_size, dtype, device):
    blocks_per_seq = [(seqlen + block_size - 1) // block_size for seqlen in seqlens_k]
    max_blocks = max(blocks_per_seq)
    total_blocks = sum(blocks_per_seq)
    num_seqs = len(seqlens_k)

    k_cache = torch.zeros(total_blocks, block_size, num_kv_heads, head_dim, dtype=dtype, device=device)
    v_cache = torch.zeros_like(k_cache)
    block_table = torch.full((num_seqs, max_blocks), -1, dtype=torch.int32, device=device)
    phys_ids = torch.randperm(total_blocks, device=device).to(dtype=torch.int32)

    k_dense = []
    v_dense = []
    next_block = 0
    for seq_idx, seqlen in enumerate(seqlens_k):
        k_seq = torch.randn(seqlen, num_kv_heads, head_dim, dtype=dtype, device=device)
        v_seq = torch.randn(seqlen, num_kv_heads, head_dim, dtype=dtype, device=device)
        k_dense.append(k_seq)
        v_dense.append(v_seq)

        for block_idx in range(blocks_per_seq[seq_idx]):
            phys = phys_ids[next_block].item()
            start = block_idx * block_size
            end = min(start + block_size, seqlen)
            block_table[seq_idx, block_idx] = phys
            k_cache[phys, : end - start] = k_seq[start:end]
            v_cache[phys, : end - start] = v_seq[start:end]
            next_block += 1

    return k_cache, v_cache, block_table, k_dense, v_dense


def naive_attn_dense(q, k, v, softmax_scale, causal=True):
    num_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    gqa_ratio = num_heads // num_kv_heads
    out = torch.empty_like(q)
    sq = q.shape[0]
    sk = k.shape[0]
    row = torch.arange(sq, device=q.device)[:, None]
    col = torch.arange(sk, device=q.device)[None, :]
    causal_mask = row + (sk - sq) < col

    for head in range(num_heads):
        kv_head = head // gqa_ratio
        scores = torch.matmul(q[:, head, :].float(), k[:, kv_head, :].float().transpose(0, 1)) * softmax_scale
        if causal:
            scores.masked_fill_(causal_mask, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out[:, head] = torch.matmul(probs, v[:, kv_head, :].float()).to(dtype=q.dtype)
    return out


def flatten_decode_reference(q, k_dense, v_dense, seqlens_k, softmax_scale):
    parts = []
    for seq_idx, seqlen in enumerate(seqlens_k):
        parts.append(
            naive_attn_dense(
                q[seq_idx : seq_idx + 1],
                k_dense[seq_idx][:seqlen],
                v_dense[seq_idx][:seqlen],
                softmax_scale,
            )
        )
    return torch.cat(parts, dim=0)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton tests.")
class KernelTest(unittest.TestCase):

    def test_paged_flash_attn_varlen_matches_reference(self):
        torch.manual_seed(0)
        device = "cuda"
        dtype = torch.float16
        num_heads = 8
        num_kv_heads = 2
        head_dim = 64
        block_size = 16
        seqlens_q = [13, 7, 19]
        seqlens_k = [29, 7, 31]
        softmax_scale = 1.0 / math.sqrt(head_dim)

        k_cache, v_cache, block_table, k_dense, v_dense = make_paged_kvcache(
            seqlens_k, num_kv_heads, head_dim, block_size, dtype, device
        )
        total_q = sum(seqlens_q)
        q = torch.randn(total_q, num_heads, head_dim, dtype=dtype, device=device)
        cu_seqlens_q = torch.tensor([0, *torch.tensor(seqlens_q).cumsum(0).tolist()], dtype=torch.int32, device=device)
        cu_seqlens_k = torch.tensor([0, *torch.tensor(seqlens_k).cumsum(0).tolist()], dtype=torch.int32, device=device)

        out = paged_flash_attn_varlen(
            q,
            k_cache,
            v_cache,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            block_table=block_table,
            softmax_scale=softmax_scale,
            causal=True,
        )

        q_start = 0
        ref_parts = []
        for seq_idx, (sq, sk) in enumerate(zip(seqlens_q, seqlens_k)):
            ref_parts.append(
                naive_attn_dense(
                    q[q_start : q_start + sq],
                    k_dense[seq_idx][:sk],
                    v_dense[seq_idx][:sk],
                    softmax_scale,
                )
            )
            q_start += sq
        ref = torch.cat(ref_parts, dim=0)

        torch.testing.assert_close(out.float(), ref.float(), atol=2e-2, rtol=2e-2)

    def test_paged_flash_attn_decode_matches_reference(self):
        torch.manual_seed(1)
        device = "cuda"
        dtype = torch.float16
        num_heads = 8
        num_kv_heads = 2
        head_dim = 64
        block_size = 16
        seqlens_k = [45, 12, 33, 8]
        softmax_scale = 1.0 / math.sqrt(head_dim)

        k_cache, v_cache, block_table, k_dense, v_dense = make_paged_kvcache(
            seqlens_k, num_kv_heads, head_dim, block_size, dtype, device
        )
        q = torch.randn(len(seqlens_k), num_heads, head_dim, dtype=dtype, device=device)
        cache_seqlens = torch.tensor(seqlens_k, dtype=torch.int32, device=device)

        out = paged_flash_attn_decode(
            q,
            k_cache,
            v_cache,
            cache_seqlens=cache_seqlens,
            block_table=block_table,
            softmax_scale=softmax_scale,
            causal=True,
        )
        ref = flatten_decode_reference(q, k_dense, v_dense, seqlens_k, softmax_scale)

        torch.testing.assert_close(out.float(), ref.float(), atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    unittest.main()
