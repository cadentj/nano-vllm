import triton
import triton.language as tl

"""
TODO: Basic flash attention kernel for Qwen3.

Doesn't include a bias term, RoPE is applied before attention rather than an explicit attention bias.


M: M in the MxN attention matrix, the Q sequence dimension
N: N in the ...

HB: batch_idx * n_heads + head_idx
    is unpacked to H and B immediately
"""

@triton.jit 
def basic_kernel(
    # pointers to q,k,v,o tensors
    Q,
    K,
    V,
    Out,
    # input shapes
    seqlen_q,
    seqlen_k,
    nheads,
    headdim,
    # strides for q,k,v,o tensor dims
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_ob,
    stride_oh,
    stride_om,
    # constants
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
): 

    """
    basic flash attn forward kernel for non ragged sequences

    assumes causal attention
    """

    # Program IDs correspond to input shapes
    # 
    start_m = tl.program_id(0)  # Q sequence idx
    off_hb = tl.program_id(1)   # Packed head,batch offset

    off_b = off_hb // nheads    # batch offset
    off_h = off_hb % nheads     # head offset

    # Initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    # Initialize pointers to Q, K, V
    q_ptrs = (
        Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :])
    )
    k_ptrs = (
        K + off_b * stride_kb + off_h * stride_kh + (offs_n[:, None] * stride_kn + offs_d[None, :])
    )
    v_ptrs = (
        V + off_b * stride_vb + off_h * stride_vh + (offs_n[:, None] * stride_vn + offs_d[None, :])
    )

    # initialize pointer to m and l
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)

    # loop over k, v and update accumulator
    end_n = tl.minimum((start_m + 1) * BLOCK_M, seqlen_k) # would just be seqlen_k if not causal?

    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # -- compute qk ----
        if EVEN_N & EVEN_M:  # If we just do "if EVEN_N", there seems to be some race condition
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs + start_n * stride_kn)
            else:
                k = tl.load(k_ptrs + start_n * stride_kn, mask=offs_d[None, :] < headdim, other=0.0)
        else:
            if EVEN_HEADDIM:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
            else:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                    other=0.0,
                )