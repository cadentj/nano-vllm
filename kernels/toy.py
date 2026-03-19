"""
Toy naive attention with ragged sequences + paged KV cache.
Use as a correctness reference when building the Triton kernel.

Run:  python kernels/toy.py
"""

import torch
import math


# ── Paged KV cache setup ─────────────────────────────────────────────

def make_paged_kvcache(seqlens_k, num_kv_heads, head_dim, block_size,
                       dtype=torch.float16, device="cuda"):
    """
    Allocate a paged KV cache, fill it with random data, and return both the
    paged representation and the dense originals (for reference checking).

    Returns:
        k_cache, v_cache : (num_physical_blocks, block_size, num_kv_heads, head_dim)
        block_table       : (num_seqs, max_blocks_per_seq)  int32
        k_dense, v_dense  : list of (seqlen_k_i, num_kv_heads, head_dim) tensors
    """
    blocks_per_seq = [(s + block_size - 1) // block_size for s in seqlens_k]
    max_blocks = max(blocks_per_seq)
    total_blocks = sum(blocks_per_seq)
    num_seqs = len(seqlens_k)

    k_cache = torch.zeros(total_blocks, block_size, num_kv_heads, head_dim,
                          dtype=dtype, device=device)
    v_cache = torch.zeros_like(k_cache)
    block_table = torch.zeros(num_seqs, max_blocks, dtype=torch.int32, device=device)

    # shuffle physical blocks so they're non-contiguous (realistic)
    phys_ids = torch.randperm(total_blocks, device=device).int()

    k_dense, v_dense = [], []
    blk_idx = 0
    for seq, seqlen in enumerate(seqlens_k):
        k_seq = torch.randn(seqlen, num_kv_heads, head_dim, dtype=dtype, device=device)
        v_seq = torch.randn(seqlen, num_kv_heads, head_dim, dtype=dtype, device=device)
        k_dense.append(k_seq)
        v_dense.append(v_seq)

        for b in range(blocks_per_seq[seq]):
            phys = phys_ids[blk_idx].item()
            block_table[seq, b] = phys
            start = b * block_size
            end = min(start + block_size, seqlen)
            k_cache[phys, :end - start] = k_seq[start:end]
            v_cache[phys, :end - start] = v_seq[start:end]
            blk_idx += 1

    return k_cache, v_cache, block_table, k_dense, v_dense


def gather_kv(k_cache, v_cache, block_table, seq_idx, seqlen_k, block_size):
    """Reconstruct dense K/V for one sequence by walking the block table."""
    num_kv_heads, head_dim = k_cache.shape[2], k_cache.shape[3]
    k = torch.empty(seqlen_k, num_kv_heads, head_dim,
                    dtype=k_cache.dtype, device=k_cache.device)
    v = torch.empty_like(k)

    for b in range((seqlen_k + block_size - 1) // block_size):
        phys = block_table[seq_idx, b].item()
        start = b * block_size
        end = min(start + block_size, seqlen_k)
        k[start:end] = k_cache[phys, :end - start]
        v[start:end] = v_cache[phys, :end - start]
    return k, v


# ── Naive attention implementations ──────────────────────────────────

def naive_attn_varlen_paged(q, k_cache, v_cache,
                            cu_seqlens_q, cu_seqlens_k,
                            block_table, block_size,
                            softmax_scale, causal=True):
    """
    Variable-length attention that reads K/V from a paged cache.

    Args:
        q:            (total_q_tokens, num_heads, head_dim)   -- ragged / flattened
        k_cache:      (num_blocks, block_size, num_kv_heads, head_dim)
        v_cache:      same
        cu_seqlens_q: (num_seqs + 1,)  int32
        cu_seqlens_k: (num_seqs + 1,)  int32
        block_table:  (num_seqs, max_blocks)  int32
        block_size:   int
        softmax_scale: float
    Returns:
        o:            (total_q_tokens, num_heads, head_dim)

    *** This is the function your Triton kernel replaces. ***
    """
    num_seqs = cu_seqlens_q.shape[0] - 1
    num_heads, head_dim = q.shape[1], q.shape[2]
    num_kv_heads = k_cache.shape[2]
    gqa_ratio = num_heads // num_kv_heads

    o = torch.empty_like(q)

    for s in range(num_seqs):
        q_off = cu_seqlens_q[s].item()
        sq    = cu_seqlens_q[s + 1].item() - q_off
        sk    = (cu_seqlens_k[s + 1] - cu_seqlens_k[s]).item()

        q_seq = q[q_off : q_off + sq]                       # (sq, H, D)
        k_seq, v_seq = gather_kv(k_cache, v_cache,          # (sk, Hkv, D)
                                 block_table, s, sk, block_size)

        for h in range(num_heads):
            kv_h = h // gqa_ratio
            qh = q_seq[:, h, :].float()                     # (sq, D)
            kh = k_seq[:, kv_h, :].float()                  # (sk, D)
            vh = v_seq[:, kv_h, :].float()                  # (sk, D)

            scores = (qh @ kh.T) * softmax_scale            # (sq, sk)

            if causal:
                row = torch.arange(sq, device=q.device)[:, None]
                col = torch.arange(sk, device=q.device)[None, :]
                # q token at position i (0-indexed within the q window) corresponds
                # to absolute position (sk - sq + i) in the full sequence
                scores.masked_fill_(row + (sk - sq) < col, float("-inf"))

            attn = torch.softmax(scores, dim=-1)
            o[q_off : q_off + sq, h, :] = (attn @ vh).to(q.dtype)

    return o


def naive_attn_dense(q, k, v, softmax_scale, num_heads, num_kv_heads, causal=True):
    """Simplest possible single-sequence dense attention (no paging)."""
    sq, sk = q.shape[0], k.shape[0]
    gqa_ratio = num_heads // num_kv_heads
    o = torch.empty_like(q)

    for h in range(num_heads):
        kv_h = h // gqa_ratio
        qh = q[:, h, :].float()
        kh = k[:, kv_h, :].float()
        vh = v[:, kv_h, :].float()
        scores = (qh @ kh.T) * softmax_scale
        if causal:
            row = torch.arange(sq, device=q.device)[:, None]
            col = torch.arange(sk, device=q.device)[None, :]
            scores.masked_fill_(row + (sk - sq) < col, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        o[:, h, :] = (attn @ vh).to(q.dtype)
    return o


# ── Test harness ─────────────────────────────────────────────────────

def run_test(name, seqlens_q, seqlens_k, num_heads, num_kv_heads, head_dim,
             block_size, dtype=torch.float16, device="cuda"):
    softmax_scale = 1.0 / math.sqrt(head_dim)
    assert len(seqlens_q) == len(seqlens_k)

    # build paged cache from the K-side lengths
    k_cache, v_cache, block_table, k_dense, v_dense = make_paged_kvcache(
        seqlens_k, num_kv_heads, head_dim, block_size, dtype, device,
    )

    # build ragged Q
    total_q = sum(seqlens_q)
    q = torch.randn(total_q, num_heads, head_dim, dtype=dtype, device=device)

    cu_seqlens_q = torch.zeros(len(seqlens_q) + 1, dtype=torch.int32, device=device)
    cu_seqlens_k = torch.zeros(len(seqlens_k) + 1, dtype=torch.int32, device=device)
    for i, (sq, sk) in enumerate(zip(seqlens_q, seqlens_k)):
        cu_seqlens_q[i + 1] = cu_seqlens_q[i] + sq
        cu_seqlens_k[i + 1] = cu_seqlens_k[i] + sk

    # --- paged varlen attention ---
    o_paged = naive_attn_varlen_paged(
        q, k_cache, v_cache, cu_seqlens_q, cu_seqlens_k,
        block_table, block_size, softmax_scale, causal=True,
    )

    # --- dense per-sequence reference ---
    parts = []
    q_off = 0
    for i, (sq, sk) in enumerate(zip(seqlens_q, seqlens_k)):
        parts.append(naive_attn_dense(
            q[q_off:q_off + sq], k_dense[i], v_dense[i],
            softmax_scale, num_heads, num_kv_heads, causal=True,
        ))
        q_off += sq
    o_dense = torch.cat(parts, dim=0)

    diff = (o_paged.float() - o_dense.float()).abs().max().item()
    status = "PASS" if diff < 1e-2 else "FAIL"
    print(f"  [{status}] {name:30s}  max_diff={diff:.2e}")
    assert diff < 1e-2, f"MISMATCH in {name}"


def main():
    torch.manual_seed(42)
    device = "cuda"

    num_heads    = 8
    num_kv_heads = 2      # GQA ratio = 4
    head_dim     = 64
    block_size   = 16     # intentionally small for debugging

    print(f"num_heads={num_heads}  num_kv_heads={num_kv_heads}  "
          f"head_dim={head_dim}  block_size={block_size}\n")

    # ── Test 1: standard prefill (seqlen_q == seqlen_k, mixed lengths) ──
    run_test("prefill (mixed lengths)",
             seqlens_q=[13, 7, 21],
             seqlens_k=[13, 7, 21],
             num_heads=num_heads, num_kv_heads=num_kv_heads,
             head_dim=head_dim, block_size=block_size)

    # ── Test 2: prefix-cached prefill (seqlen_q < seqlen_k) ──
    run_test("prefix cache (sq < sk)",
             seqlens_q=[5, 3, 10],
             seqlens_k=[20, 16, 30],
             num_heads=num_heads, num_kv_heads=num_kv_heads,
             head_dim=head_dim, block_size=block_size)

    # ── Test 3: decode (seqlen_q == 1 per seq) ──
    run_test("decode (sq=1)",
             seqlens_q=[1, 1, 1, 1],
             seqlens_k=[45, 12, 33, 8],
             num_heads=num_heads, num_kv_heads=num_kv_heads,
             head_dim=head_dim, block_size=block_size)

    # ── Test 4: single long sequence ──
    run_test("single long seq",
             seqlens_q=[200],
             seqlens_k=[200],
             num_heads=num_heads, num_kv_heads=num_kv_heads,
             head_dim=head_dim, block_size=block_size)

    # ── Test 5: seqlens that exactly divide block_size ──
    run_test("block-aligned",
             seqlens_q=[16, 32, 48],
             seqlens_k=[16, 32, 48],
             num_heads=num_heads, num_kv_heads=num_kv_heads,
             head_dim=head_dim, block_size=block_size)

    # ── Test 6: head_dim = 128 ──
    run_test("head_dim=128",
             seqlens_q=[13, 7],
             seqlens_k=[13, 7],
             num_heads=num_heads, num_kv_heads=num_kv_heads,
             head_dim=128, block_size=block_size)

    print("\nAll tests passed.\n")

    # ── Print the target signature ──
    seqlens_q = [13, 7, 21]
    seqlens_k = [13, 7, 21]
    total_q = sum(seqlens_q)
    total_blocks = sum((s + block_size - 1) // block_size for s in seqlens_k)
    n = len(seqlens_q)

    print("=" * 64)
    print("TARGET KERNEL SIGNATURE")
    print("=" * 64)
    print(f"  q             : ({total_q}, {num_heads}, {head_dim})")
    print(f"  k_cache       : ({total_blocks}, {block_size}, {num_kv_heads}, {head_dim})")
    print(f"  v_cache       : ({total_blocks}, {block_size}, {num_kv_heads}, {head_dim})")
    print(f"  cu_seqlens_q  : ({n + 1},)   int32")
    print(f"  cu_seqlens_k  : ({n + 1},)   int32")
    print(f"  block_table   : ({n}, max_blocks_per_seq)   int32")
    print(f"  block_size    : {block_size}")
    print(f"  softmax_scale : 1/sqrt(head_dim)")
    print(f"  output        : ({total_q}, {num_heads}, {head_dim})")


if __name__ == "__main__":
    main()
