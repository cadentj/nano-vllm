import math

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_attn_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    o_ptr,
    q_seq_ids_ptr,
    q_positions_ptr,
    seq_lens_q_ptr,
    seq_lens_k_ptr,
    block_table_ptr,
    stride_qt,
    stride_qh,
    stride_kb,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_vb,
    stride_vt,
    stride_vh,
    stride_vd,
    stride_ot,
    stride_oh,
    stride_bt0,
    stride_bt1,
    softmax_scale,
    num_heads,
    num_kv_heads,
    cache_block_size,
    headdim,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    q_idx = tl.program_id(0)
    q_head = tl.program_id(1)

    seq_idx = tl.load(q_seq_ids_ptr + q_idx)
    q_pos = tl.load(q_positions_ptr + q_idx)
    seqlen_q = tl.load(seq_lens_q_ptr + seq_idx)
    seqlen_k = tl.load(seq_lens_k_ptr + seq_idx)

    gqa_group_size = num_heads // num_kv_heads
    kv_head = q_head // gqa_group_size
    q_abs_pos = q_pos + (seqlen_k - seqlen_q)

    offs_d = tl.arange(0, BLOCK_D)
    q_ptrs = q_ptr + q_idx * stride_qt + q_head * stride_qh + offs_d
    q = tl.load(q_ptrs, mask=offs_d < headdim, other=0.0).to(tl.float32)

    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    for start_n in range(0, seqlen_k, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        valid_n = offs_n < seqlen_k
        causal_n = offs_n <= q_abs_pos
        logical_block = offs_n // cache_block_size
        in_block_offset = offs_n % cache_block_size

        phys_block = tl.load(
            block_table_ptr + seq_idx * stride_bt0 + logical_block * stride_bt1,
            mask=valid_n,
            other=0,
        )

        k_ptrs = (
            k_cache_ptr
            + phys_block[:, None] * stride_kb
            + in_block_offset[:, None] * stride_kt
            + kv_head * stride_kh
            + offs_d[None, :] * stride_kd
        )
        mask = valid_n[:, None] & (offs_d[None, :] < headdim)
        k = tl.load(k_ptrs, mask=mask, other=0.0).to(tl.float32)

        scores = tl.sum(k * q[None, :], axis=1) * softmax_scale
        scores = tl.where(valid_n & causal_n, scores, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(scores, axis=0))
        p = tl.exp(scores - m_ij)
        alpha = tl.exp(m_i - m_ij)

        v_ptrs = (
            v_cache_ptr
            + phys_block[:, None] * stride_vb
            + in_block_offset[:, None] * stride_vt
            + kv_head * stride_vh
            + offs_d[None, :] * stride_vd
        )
        v = tl.load(v_ptrs, mask=mask, other=0.0).to(tl.float32)

        acc = acc * alpha + tl.sum(v * p[:, None], axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_ij

    out = acc / l_i
    o_ptrs = o_ptr + q_idx * stride_ot + q_head * stride_oh + offs_d
    tl.store(o_ptrs, out.to(q_ptr.dtype.element_ty), mask=offs_d < headdim)


def _check_paged_inputs(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    seq_lens_q: torch.Tensor,
    seq_lens_k: torch.Tensor,
    block_table: torch.Tensor,
    causal: bool,
) -> None:
    if not causal:
        raise NotImplementedError("Only causal attention is supported.")
    if q.ndim != 3:
        raise ValueError(f"Expected q to have shape (tokens, heads, dim), got {tuple(q.shape)}.")
    if k_cache.ndim != 4 or v_cache.ndim != 4:
        raise ValueError("Expected cache tensors to have shape (blocks, block_size, kv_heads, head_dim).")
    if q.device != k_cache.device or q.device != v_cache.device:
        raise ValueError("q, k_cache, and v_cache must be on the same device.")
    if q.device != block_table.device or q.device != seq_lens_q.device or q.device != seq_lens_k.device:
        raise ValueError("Metadata tensors must be on the same device as q.")
    if q.dtype != k_cache.dtype or q.dtype != v_cache.dtype:
        raise ValueError("q, k_cache, and v_cache must share the same dtype.")
    if q.shape[2] != k_cache.shape[3] or q.shape[2] != v_cache.shape[3]:
        raise ValueError("Head dimensions for q, k_cache, and v_cache must match.")
    if q.shape[1] % k_cache.shape[2] != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads for GQA/MQA.")
    if block_table.ndim != 2:
        raise ValueError(f"Expected block_table to have shape (num_seqs, max_blocks), got {tuple(block_table.shape)}.")
    if seq_lens_q.ndim != 1 or seq_lens_k.ndim != 1:
        raise ValueError("Sequence length metadata must be 1D.")
    if seq_lens_q.numel() != seq_lens_k.numel() or seq_lens_q.numel() != block_table.shape[0]:
        raise ValueError("Metadata tensors must agree on the number of sequences.")
    if q.stride(-1) != 1 or k_cache.stride(-1) != 1 or v_cache.stride(-1) != 1:
        raise ValueError("The last dimension must be contiguous for q, k_cache, and v_cache.")


def _build_prefill_query_metadata(cu_seqlens_q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seq_lens_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).to(dtype=torch.int32)
    seq_ids = torch.arange(seq_lens_q.numel(), device=cu_seqlens_q.device, dtype=torch.int32)
    counts = seq_lens_q.to(dtype=torch.int64)
    q_seq_ids = torch.repeat_interleave(seq_ids, counts)
    q_starts = torch.repeat_interleave(cu_seqlens_q[:-1].to(dtype=torch.int32), counts)
    q_positions = torch.arange(q_seq_ids.numel(), device=cu_seqlens_q.device, dtype=torch.int32) - q_starts
    return q_seq_ids, q_positions, seq_lens_q


def _paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    q_seq_ids: torch.Tensor,
    q_positions: torch.Tensor,
    seq_lens_q: torch.Tensor,
    seq_lens_k: torch.Tensor,
    block_table: torch.Tensor,
    softmax_scale: float,
    causal: bool,
) -> torch.Tensor:
    _check_paged_inputs(q, k_cache, v_cache, seq_lens_q, seq_lens_k, block_table, causal)
    if q.shape[0] == 0:
        return torch.empty_like(q)
    if q_seq_ids.shape != q_positions.shape or q_seq_ids.numel() != q.shape[0]:
        raise ValueError("Per-query sequence ids and positions must match the flattened q dimension.")

    head_dim = q.shape[-1]
    block_headdim = triton.next_power_of_2(head_dim)
    if block_headdim > 256:
        raise NotImplementedError(f"Head dimension {head_dim} is not supported yet.")

    out = torch.empty_like(q)
    num_warps = 4 if block_headdim <= 64 else 8
    grid = (q.shape[0], q.shape[1])
    _paged_attn_kernel[grid](
        q,
        k_cache,
        v_cache,
        out,
        q_seq_ids,
        q_positions,
        seq_lens_q,
        seq_lens_k,
        block_table,
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(3),
        out.stride(0),
        out.stride(1),
        block_table.stride(0),
        block_table.stride(1),
        softmax_scale,
        q.shape[1],
        k_cache.shape[2],
        k_cache.shape[1],
        head_dim,
        BLOCK_N=64,
        BLOCK_D=block_headdim,
        num_warps=num_warps,
    )
    return out


def dense_flash_attn_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    softmax_scale: float | None = None,
    causal: bool = True,
) -> torch.Tensor:
    if not causal:
        raise NotImplementedError("Only causal attention is supported.")
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("Expected q, k, and v to have shape (tokens, heads, dim).")
    if q.shape[2] != k.shape[2] or q.shape[2] != v.shape[2]:
        raise ValueError("Head dimensions for q, k, and v must match.")
    if k.shape[1] != v.shape[1]:
        raise ValueError("k and v must have the same number of KV heads.")
    if q.shape[1] % k.shape[1] != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads for GQA/MQA.")

    softmax_scale = softmax_scale or (1.0 / math.sqrt(q.shape[-1]))
    out = torch.empty_like(q)
    num_seqs = cu_seqlens_q.numel() - 1
    gqa_ratio = q.shape[1] // k.shape[1]

    for seq_idx in range(num_seqs):
        q_start = cu_seqlens_q[seq_idx].item()
        q_end = cu_seqlens_q[seq_idx + 1].item()
        k_start = cu_seqlens_k[seq_idx].item()
        k_end = cu_seqlens_k[seq_idx + 1].item()
        q_seq = q[q_start:q_end]
        k_seq = k[k_start:k_end]
        v_seq = v[k_start:k_end]

        row = torch.arange(q_seq.shape[0], device=q.device)[:, None]
        col = torch.arange(k_seq.shape[0], device=q.device)[None, :]
        causal_mask = row + (k_seq.shape[0] - q_seq.shape[0]) < col

        for head in range(q.shape[1]):
            kv_head = head // gqa_ratio
            scores = torch.matmul(
                q_seq[:, head, :].float(),
                k_seq[:, kv_head, :].float().transpose(0, 1),
            ) * softmax_scale
            scores.masked_fill_(causal_mask, float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            out[q_start:q_end, head] = torch.matmul(probs, v_seq[:, kv_head, :].float()).to(dtype=q.dtype)
    return out


def paged_flash_attn_varlen(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    block_table: torch.Tensor,
    softmax_scale: float | None = None,
    causal: bool = True,
) -> torch.Tensor:
    if cu_seqlens_q.ndim != 1 or cu_seqlens_k.ndim != 1:
        raise ValueError("cu_seqlens_q and cu_seqlens_k must be 1D.")
    if cu_seqlens_q.numel() != cu_seqlens_k.numel():
        raise ValueError("cu_seqlens_q and cu_seqlens_k must describe the same number of sequences.")
    if block_table is None:
        raise ValueError("block_table is required for paged attention.")

    q_seq_ids, q_positions, seq_lens_q = _build_prefill_query_metadata(cu_seqlens_q)
    seq_lens_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).to(dtype=torch.int32)
    scale = softmax_scale or (1.0 / math.sqrt(q.shape[-1]))
    return _paged_attention(
        q,
        k_cache,
        v_cache,
        q_seq_ids,
        q_positions,
        seq_lens_q,
        seq_lens_k,
        block_table.to(dtype=torch.int32),
        scale,
        causal,
    )


def paged_flash_attn_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    block_table: torch.Tensor,
    softmax_scale: float | None = None,
    causal: bool = True,
) -> torch.Tensor:
    if q.ndim != 3:
        raise ValueError(f"Expected q to have shape (batch, heads, dim), got {tuple(q.shape)}.")
    if cache_seqlens.ndim != 1:
        raise ValueError("cache_seqlens must be 1D.")
    if q.shape[0] != cache_seqlens.numel():
        raise ValueError("q and cache_seqlens must agree on batch size.")
    if block_table is None:
        raise ValueError("block_table is required for paged attention decode.")

    batch = q.shape[0]
    device = q.device
    q_seq_ids = torch.arange(batch, device=device, dtype=torch.int32)
    q_positions = torch.zeros(batch, device=device, dtype=torch.int32)
    seq_lens_q = torch.ones(batch, device=device, dtype=torch.int32)
    seq_lens_k = cache_seqlens.to(dtype=torch.int32)
    scale = softmax_scale or (1.0 / math.sqrt(q.shape[-1]))
    return _paged_attention(
        q,
        k_cache,
        v_cache,
        q_seq_ids,
        q_positions,
        seq_lens_q,
        seq_lens_k,
        block_table.to(dtype=torch.int32),
        scale,
        causal,
    )
