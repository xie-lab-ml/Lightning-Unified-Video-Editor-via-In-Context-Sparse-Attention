"""
TileLang rewrite of sparse piecewise attention (forward) matching Triton logic,
with Chunked Interval optimization (KC/VC batch processing).

Stages:
  1) Reduce Q/K/V into block sums (qc, kc, vc)
  2) Block relevance (qc @ kc^T) → TopK indices (can be host or GPU)
  3) Attention compute:
     - split kernels: sparse_topk_kernel + sparse_interval_kernel
     - fused fast-path (VBlocks==1): sparse_fused_vblocks1_kernel
"""

import math
import itertools
import tilelang as tl
import tilelang.language as T

try:
    # Optional: autotune fused kernel parameters (threads/num_stages).
    # Keep this optional so normal runs don't pay tuning overhead.
    from tilelang.autotuner import autotune  # type: ignore
except Exception:  # pragma: no cover
    autotune = None


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


# ---------------------------------------------------------
# Block reductions (TileLang, GPU) to mimic Triton chunk_reduce
# ---------------------------------------------------------
@tl.jit(out_idx=[1], pass_configs={tl.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def reduce_q_kernel(
    BH: int,
    NQ: int,
    BT: int,
    BK: int,
    TQ: int,
    threads: int = 128,
):
    dtype = "bfloat16"
    accum_dtype = "float"
    q_shape = [BH, TQ, BK]
    qc_shape = [BH, NQ, BK]

    @T.prim_func
    def main(Q: T.Tensor(q_shape, dtype), QC: T.Tensor(qc_shape, dtype)):
        with T.Kernel(NQ, BH, threads=threads) as (i_nq, i_bh):
            Q_shared = T.alloc_shared([BT, BK], dtype)
            acc_q = T.alloc_fragment([BK], accum_dtype)

            # load block
            q_start = i_nq * BT
            T.copy(Q[i_bh, q_start:q_start + BT, :], Q_shared)

            # reduce over BT
            T.reduce_sum(Q_shared, acc_q, dim=0, clear=True)

            # write
            for k in T.Parallel(BK):
                QC[i_bh, i_nq, k] = T.cast(acc_q[k], dtype)

    return main


@tl.jit(out_idx=[2, 3], pass_configs={tl.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def reduce_kv_kernel(
    BH: int,
    NKV: int,
    BT: int,
    BK: int,
    BV: int,
    VBlocks: int,
    TKV: int,
    Vdim: int,
    threads: int = 128,
):
    dtype = "bfloat16"
    accum_dtype = "float"
    k_shape = [BH, TKV, BK]
    v_shape = [BH, TKV, Vdim]
    kc_shape = [BH, NKV, BK]
    vc_shape = [BH, NKV, Vdim]
    # NOTE: KVC ([BH, NKV, BK, Vdim]) used to be computed here (K^T @ V per block),
    # but it's currently unused by downstream kernels. We skip it for performance.

    @T.prim_func
    def main(
        K: T.Tensor(k_shape, dtype),
        V: T.Tensor(v_shape, dtype),
        KC: T.Tensor(kc_shape, dtype),
        VC: T.Tensor(vc_shape, dtype),
    ):
        with T.Kernel(NKV, BH, threads=threads) as (i_nkv, i_bh):
            K_shared = T.alloc_shared([BT, BK], dtype)
            V_shared = T.alloc_shared([BT, BV], dtype)
            acc_kc = T.alloc_fragment([BK], accum_dtype)
            acc_vc = T.alloc_fragment([BV], accum_dtype)

            k_start = i_nkv * BT

            # load K block
            T.copy(K[i_bh, k_start:k_start + BT, :], K_shared)
            # KC sum over rows
            T.reduce_sum(K_shared, acc_kc, dim=0, clear=True)
            for k in T.Parallel(BK):
                KC[i_bh, i_nkv, k] = T.cast(acc_kc[k], dtype)

            # VC and KVC in BV tiles
            for vblk in T.serial(VBlocks):
                v_col = vblk * BV
                # load V tile
                T.copy(V[i_bh, k_start:k_start + BT, v_col:v_col + BV], V_shared)
                # VC sum
                T.reduce_sum(V_shared, acc_vc, dim=0, clear=True)
                for j in T.Parallel(BV):
                    VC[i_bh, i_nkv, v_col + j] = T.cast(acc_vc[j], dtype)

    return main


# ---------------------------------------------------------
# Block Relevance + TopK Kernel (TileLang, GPU)
# Replaces PyTorch einsum('bhik,bhjk->bhij') + topk
# ---------------------------------------------------------
@tl.jit(out_idx=[2], pass_configs={tl.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def block_relevance_topk_kernel(
    BH: int,
    NQ: int,
    NKV: int,
    Kdim: int,
    TopK: int,
    BT: int, # Block size for NQ (usually 64)
    BK: int, # Kdim (usually 128)
    threads: int = 128,
):
    """
    Computes scores = QC @ KC^T, then selects TopK indices per row.
    Input:
      QC: [BH, NQ, Kdim]
      KC: [BH, NKV, Kdim]
    Output:
      Indices: [BH, NQ, TopK] (int32)
    """
    dtype = "bfloat16"
    accum_dtype = "float"
    
    # Grid: (NQ, BH)
    # Each CTA handles one query block row, computes scores against all NKV, selects TopK.
    
    qc_shape = [BH, NQ, BK]
    kc_shape = [BH, NKV, BK]
    indices_shape = [BH, NQ, TopK]

    @T.prim_func
    def main(
        QC: T.Tensor(qc_shape, dtype),
        KC: T.Tensor(kc_shape, dtype),
        Indices: T.Tensor(indices_shape, "int32")
    ):
        with T.Kernel(NQ, BH, threads=threads) as (i_nq, i_bh):
            # Shared memory for QC block
            QC_shared = T.alloc_shared([BK], dtype)
            
            # Fragment for Scores [NKV] - assumes NKV fits in registers (e.g. < 1024)
            Scores_frag = T.alloc_fragment([NKV], accum_dtype)
            
            # 1. Load QC vector to Shared
            for k in T.Parallel(BK):
                QC_shared[k] = QC[i_bh, i_nq, k]
            
            # 2. Compute Scores: Loop over NKV blocks
            # We compute all scores into the fragment
            for j in T.Parallel(NKV):
                acc = T.alloc_fragment([1], accum_dtype)
                acc[0] = 0.0
                for k in T.serial(BK):
                    acc[0] += T.cast(QC_shared[k], accum_dtype) * T.cast(KC[i_bh, j, k], accum_dtype)
                Scores_frag[j] = acc[0]
            
            # 3. Select TopK (Iterative selection)
            max_val = T.alloc_fragment([1], accum_dtype)
            max_idx = T.alloc_fragment([1], "int32")
            current_max_idx_frag = T.alloc_fragment([NKV], "int32")

            for k_idx in T.serial(TopK):
                # Find max value
                T.reduce_max(Scores_frag, max_val, dim=0, clear=True)
                
                # Identify index of max value
                # We set matching indices to their index, others to -1
                for j in T.Parallel(NKV):
                    current_max_idx_frag[j] = T.if_then_else(Scores_frag[j] == max_val[0], j, -1)
                
                # Reduce to find the largest index among the winners (resolves ties deterministically)
                T.reduce_max(current_max_idx_frag, max_idx, dim=0, clear=True)
                
                # Write result
                Indices[i_bh, i_nq, k_idx] = max_idx[0]
                
                # Mask out the selected index
                for j in T.Parallel(NKV):
                    if j == max_idx[0]:
                        Scores_frag[j] = -T.infinity(accum_dtype)

    return main


# ---------------------------------------------------------
# Sparse TopK kernel (no interval)
# ---------------------------------------------------------
@tl.jit(
    out_idx=[6, 7, 8],
    pass_configs={
        tl.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tl.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    })
def sparse_topk_kernel(
    BH: int,
    NQ: int,
    NKV: int,
    TopK: int,
    BT: int,
    BK: int,
    BV: int,
    Kdim: int,
    Vdim: int,
    TQ: int,
    TKV: int,
    VBlocks: int,
    num_stages: int = 2,
    threads: int = 128,
):
    """
    Sparse TopK FlashAttention only.
    Produces partial output + running logsum (L) and max (M) for later interval kernel.
    Grid: (VBlocks, NQ, BH)
    """
    dtype = "bfloat16"
    accum_dtype = "float"
    scale = (1.0 / Kdim) ** 0.5 * 1.44269504  # log2(e)

    q_shape = [BH, TQ, Kdim]
    k_shape = [BH, TKV, Kdim]
    v_shape = [BH, TKV, Vdim]
    kc_shape = [BH, NKV, Kdim]          # block-sum of K
    vc_shape = [BH, NKV, Vdim]          # block-sum of V
    indices_shape = [BH, NQ, TopK]
    out_shape = [BH, NQ, BT, Vdim]
    # IMPORTANT: keep running softmax state in fp32.
    # Triton fused kernel keeps l_i/m_i in fp32 across the whole computation.
    # If we store them in bf16 between split kernels, numerical error can be large.
    l_shape = [BH, NQ, BT]
    m_shape = [BH, NQ, BT]

    # A1 fast path: if VBlocks == 1 (i.e. we compute full Vdim in one block),
    # we can safely re-enable the aggressive swizzle + async T.copy path that was fast before.
    # The previous correctness issue was vblk!=0; with VBlocks==1, i_vblk is always 0.
    _single_vblock = (VBlocks == 1)

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(k_shape, dtype),
        V: T.Tensor(v_shape, dtype),
        KC: T.Tensor(kc_shape, dtype),
        VC: T.Tensor(vc_shape, dtype),
        Indices: T.Tensor(indices_shape, "int32"),
        Output: T.Tensor(out_shape, dtype),
        L: T.Tensor(l_shape, accum_dtype),
        M: T.Tensor(m_shape, accum_dtype),
    ):
        with T.Kernel(VBlocks, NQ, BH, threads=threads) as (i_vblk, i_nq, i_bh):
            # Shared buffers
            Q_shared = T.alloc_shared([BT, BK], dtype)
            K0 = T.alloc_shared([BT, BK], dtype)
            V0 = T.alloc_shared([BT, BV], dtype)
            O_shared = T.alloc_shared([BT, BV], dtype)
            KC_shared = T.alloc_shared([1, BK], dtype)

            acc_s = T.alloc_fragment([BT, BT], accum_dtype)
            acc_s_cast = T.alloc_fragment([BT, BT], dtype)
            acc_o = T.alloc_fragment([BT, BV], accum_dtype)
            scores_max = T.alloc_fragment([BT], accum_dtype)
            scores_max_prev = T.alloc_fragment([BT], accum_dtype)
            scores_scale = T.alloc_fragment([BT], accum_dtype)
            scores_sum = T.alloc_fragment([BT], accum_dtype)
            logsum = T.alloc_fragment([BT], accum_dtype)

            # Explicitly cast vblk to int32 for address arithmetic (avoid PrimExpr typing quirks).
            v_col = T.Cast("int32", i_vblk) * BV

            # Load indices for this query block
            block_indices = T.alloc_local([TopK], "int32")
            for t in T.serial(TopK):
                block_indices[t] = Indices[i_bh, i_nq, t]

            if _single_vblock:
                # Fast path (VBlocks==1): swizzle Q/K/V/O and enable swizzle addressing.
                # Also switch to `T.Pipelined` over the TopK loop (mirrors TileLang flash-attn examples)
                # to improve overlap vs the manual async ping-pong version.
                T.annotate_layout({
                    Q_shared: tl.layout.make_swizzled_layout(Q_shared),
                    K0: tl.layout.make_swizzled_layout(K0),
                    V0: tl.layout.make_swizzled_layout(V0),
                    O_shared: tl.layout.make_swizzled_layout(O_shared),
                })
                T.use_swizzle(10)
            else:
                # Safe path (VBlocks>1): swizzle Q/K only, but do NOT enable swizzle addressing.
                # This preserves the vblk!=0 correctness we validated.
                T.annotate_layout({
                    Q_shared: tl.layout.make_swizzled_layout(Q_shared),
                    K0: tl.layout.make_swizzled_layout(K0),
                })

            # Load Q block
            q_start = i_nq * BT
            T.copy(Q[i_bh, q_start:q_start + BT, :], Q_shared)

            # Init accumulators
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            if _single_vblock:
                # ---- Fast path: `T.Pipelined` (no manual ping-pong) ----
                # NOTE: with `threads=256`, TileLang may hit a layout-infer conflict when converting
                # `acc_s` -> `acc_s_cast` as two fragments in one parallel loop. To enable 256 threads,
                # we optionally cast probabilities into shared memory and use shared@shared gemm.
                if threads == 256:
                    P_shared = T.alloc_shared([BT, BT], dtype)
                for i in T.Pipelined(TopK, num_stages=num_stages):
                    blk = block_indices[i]
                    k0 = blk * BT
                    T.copy(K[i_bh, k0:k0 + BT, :], K0)
                    T.copy(V[i_bh, k0:k0 + BT, v_col:v_col + BV], V0)
                    # GEMM QK^T
                    T.clear(acc_s)
                    T.gemm(Q_shared, K0, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                    # Online softmax (same as FlashAttention example)
                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    for j in T.Parallel(BT):
                        scores_max[j] = T.max(scores_max[j], scores_max_prev[j])
                    for j in T.Parallel(BT):
                        scores_scale[j] = T.exp2(scores_max_prev[j] * scale - scores_max[j] * scale)
                    for j, k in T.Parallel(BT, BT):
                        acc_s[j, k] = T.exp2(acc_s[j, k] * scale - scores_max[j] * scale)
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for j in T.Parallel(BT):
                        logsum[j] = logsum[j] * scores_scale[j] + scores_sum[j]

                    if threads == 256:
                        # Cast probs into shared to avoid fragment layout inference conflict.
                        for r, c in T.Parallel(BT, BT):
                            P_shared[r, c] = T.cast(acc_s[r, c], dtype)
                    else:
                        T.copy(acc_s, acc_s_cast)

                    # Rescale acc_o
                    for j, k in T.Parallel(BT, BV):
                        acc_o[j, k] *= scores_scale[j]

                    # GEMM PV
                    if threads == 256:
                        T.gemm(P_shared, V0, acc_o, policy=T.GemmWarpPolicy.FullRow)
                    else:
                        T.gemm(acc_s_cast, V0, acc_o, policy=T.GemmWarpPolicy.FullRow)
            else:
                # ---- Safe path: simple serial (no async ping-pong) ----
                # This path is correctness-oriented for VBlocks>1 and keeps the code simple.
                for i in T.serial(TopK):
                    blk = block_indices[i]
                    k0 = blk * BT
                    T.copy(K[i_bh, k0:k0 + BT, :], K0)
                    # Explicit V loads to avoid layout inference pitfalls when v_col != 0.
                    for r, c in T.Parallel(BT, BV):
                        V0[r, c] = V[i_bh, k0 + r, v_col + c]

                    T.clear(acc_s)
                    T.gemm(Q_shared, K0, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    for j in T.Parallel(BT):
                        scores_max[j] = T.max(scores_max[j], scores_max_prev[j])
                    for j in T.Parallel(BT):
                        scores_scale[j] = T.exp2(scores_max_prev[j] * scale - scores_max[j] * scale)
                    for j, k in T.Parallel(BT, BT):
                        acc_s[j, k] = T.exp2(acc_s[j, k] * scale - scores_max[j] * scale)
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for j in T.Parallel(BT):
                        logsum[j] = logsum[j] * scores_scale[j] + scores_sum[j]
                    T.copy(acc_s, acc_s_cast)

                    for j, k in T.Parallel(BT, BV):
                        acc_o[j, k] *= scores_scale[j]

                    T.gemm(acc_s_cast, V0, acc_o, policy=T.GemmWarpPolicy.FullRow)

            # Normalize and write output (store bf16 output, keep L/M as fp32 state).
            for j, k in T.Parallel(BT, BV):
                acc_o[j, k] /= logsum[j]
            T.copy(acc_o, O_shared)

            # Store with bound check on Vdim
            for j, k in T.Parallel(BT, BV):
                if v_col + k < Vdim:
                    Output[i_bh, i_nq, j, v_col + k] = O_shared[j, k]

            # IMPORTANT: L/M are independent of VBlocks (they depend only on Q/K).
            # Since the grid includes i_vblk, writing L/M from every VBlock causes a race
            # (last writer wins). Write them only once.
            if i_vblk == 0:
                for j in T.Parallel(BT):
                    L[i_bh, i_nq, j] = logsum[j]
                    M[i_bh, i_nq, j] = scores_max[j]

    return main


# ---------------------------------------------------------
# Interval-only kernel: Chunked processing (BN=64) matching Triton
# ---------------------------------------------------------
@tl.jit(
            # UPDATE: TileLang runtime scalar args are unreliable here; make `NKV_REAL` a compile-time constant
    # captured by the kernel specialization instead. Output/LSE go back to indices [7, 8].
    out_idx=[7, 8],
    pass_configs={
        tl.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tl.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    })
def sparse_interval_kernel(
    BH: int,
    NQ: int,
    NKV: int,
    TopK: int,
    BT: int,
    BK: int,
    BV: int,
    Kdim: int,
    Vdim: int,
    TQ: int,
    TKV: int,
    VBlocks: int,
    NKV_REAL: int = -1,
    BN: int = 64,  # Chunk Size
    num_stages: int = 2,
    threads: int = 128,
):
    """
    Interval approximation using Chunked processing (size BN).
    Computes [BT, BN] scores via Q @ KC_chunk^T, then [BT, BV] via Prob @ VC_chunk.
    Grid: (VBlocks, NQ, BH)
    """
    dtype = "bfloat16"
    accum_dtype = "float"
    scale = (1.0 / Kdim) ** 0.5 * 1.44269504  # log2(e)

    q_shape = [BH, TQ, Kdim]
    kc_shape = [BH, NKV, Kdim]
    vc_shape = [BH, NKV, Vdim]
    indices_shape = [BH, NQ, TopK]
    out_shape = [BH, NQ, BT, Vdim]
    l_shape = [BH, NQ, BT]
    m_shape = [BH, NQ, BT]
    lse_shape = [BH, NQ, BT]

    num_chunks = ceil_div(NKV, BN)
    NKV_REAL_CONST = NKV if NKV_REAL < 0 else NKV_REAL

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        KC: T.Tensor(kc_shape, dtype),
        VC: T.Tensor(vc_shape, dtype),
        Indices: T.Tensor(indices_shape, "int32"),
        OutIn: T.Tensor(out_shape, dtype),
        Lin: T.Tensor(l_shape, accum_dtype),
        Min: T.Tensor(m_shape, accum_dtype),
        Output: T.Tensor(out_shape, dtype),
        LSE: T.Tensor(lse_shape, accum_dtype),
    ):
        with T.Kernel(VBlocks, NQ, BH, threads=threads) as (i_vblk, i_nq, i_bh):
            Q_shared = T.alloc_shared([BT, BK], dtype)
            
            # Chunk buffers (ping-pong): [BN, BK] for KC, [BN, BV] for VC
            # Use async prefetch to overlap global->shared copies with compute (see TileLang examples).
            KC0 = T.alloc_shared([BN, BK], dtype)
            KC1 = T.alloc_shared([BN, BK], dtype)
            VC0 = T.alloc_shared([BN, BV], dtype)
            VC1 = T.alloc_shared([BN, BV], dtype)
            O_shared = T.alloc_shared([BT, BV], dtype)

            # Scores: [BT, BN]
            scores_chunk = T.alloc_fragment([BT, BN], accum_dtype)
            scores_chunk_cast = T.alloc_fragment([BT, BN], dtype)
            
            acc_o = T.alloc_fragment([BT, BV], accum_dtype)
            scores_max = T.alloc_fragment([BT], accum_dtype)
            scores_max_prev = T.alloc_fragment([BT], accum_dtype)
            scores_scale = T.alloc_fragment([BT], accum_dtype)
            logsum = T.alloc_fragment([BT], accum_dtype)
            
            # Helper for mask
            indices_local = T.alloc_local([TopK], "int32")
            sel_mask = T.alloc_local([BN], "int8")

            # Try explicit swizzled shared layouts (TileLang examples show this often helps GEMM speed).
            T.annotate_layout({
                Q_shared: tl.layout.make_swizzled_layout(Q_shared),
                KC0: tl.layout.make_swizzled_layout(KC0),
                KC1: tl.layout.make_swizzled_layout(KC1),
                VC0: tl.layout.make_swizzled_layout(VC0),
                VC1: tl.layout.make_swizzled_layout(VC1),
                O_shared: tl.layout.make_swizzled_layout(O_shared),
            })
            T.use_swizzle(10)

            # Load Q
            q_start = i_nq * BT
            T.copy(Q[i_bh, q_start:q_start + BT, :], Q_shared)

            # Load initial state
            v_col = i_vblk * BV
            T.copy(OutIn[i_bh, i_nq, :, v_col:v_col + BV], O_shared)
            T.copy(Lin[i_bh, i_nq, :], logsum)
            T.copy(Min[i_bh, i_nq, :], scores_max)
            
            # Initialize acc_o from O_shared
            for j, k in T.Parallel(BT, BV):
                # OutIn is normalized output from sparse_topk_kernel; reconstruct unnormalized acc via *L.
                acc_o[j, k] = T.cast(O_shared[j, k], accum_dtype) * logsum[j]

            # Triton reference accumulates interval contributions separately (g_acc/g_l) without rescaling.
            # We mirror that to match it numerically.
            g_acc = T.alloc_fragment([BT, BV], accum_dtype)
            g_l = T.alloc_fragment([BT], accum_dtype)
            T.fill(g_acc, 0)
            T.fill(g_l, 0)

            # Load TopK indices for masking
            for t in T.serial(TopK):
                indices_local[t] = Indices[i_bh, i_nq, t]

            # ---- Prefetch chunk 0 into ping buffer 0 ----
            if num_chunks > 0:
                start0 = 0
                with T.attr("default", "async_scope", 1):
                    T.copy(KC[i_bh, start0:start0 + BN, :], KC0)
                with T.attr("default", "async_scope", 1):
                    T.copy(VC[i_bh, start0:start0 + BN, v_col:v_col + BV], VC0)
                T.ptx_commit_group()
                T.ptx_wait_group(0)

            # Iterate over chunks with ping-pong prefetch
            for chunk_idx in T.serial(num_chunks):
                start_n = chunk_idx * BN
                
                # Prefetch next chunk into the other buffer
                if chunk_idx + 1 < num_chunks:
                    start_next = (chunk_idx + 1) * BN
                    if (chunk_idx + 1) % 2 == 0:
                        with T.attr("default", "async_scope", 1):
                            T.copy(KC[i_bh, start_next:start_next + BN, :], KC0)
                        with T.attr("default", "async_scope", 1):
                            T.copy(VC[i_bh, start_next:start_next + BN, v_col:v_col + BV], VC0)
                    else:
                        with T.attr("default", "async_scope", 1):
                            T.copy(KC[i_bh, start_next:start_next + BN, :], KC1)
                        with T.attr("default", "async_scope", 1):
                            T.copy(VC[i_bh, start_next:start_next + BN, v_col:v_col + BV], VC1)
                    T.ptx_commit_group()

                # Compute Scores: Q [BT, BK] @ KC_chunk.T [BK, BN] -> [BT, BN]
                T.clear(scores_chunk)
                if chunk_idx % 2 == 0:
                    T.gemm(Q_shared, KC0, scores_chunk, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                else:
                    T.gemm(Q_shared, KC1, scores_chunk, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                # Build selection mask for this chunk: sel_mask[j]=1 if (start_n+j) is one of TopK indices.
                for j in T.Parallel(BN):
                    sel_mask[j] = 0
                for t in T.serial(TopK):
                    idx = indices_local[t] - start_n
                    if idx >= 0:
                        if idx < BN:
                            sel_mask[idx] = 1
                
                # Apply mask and scale.
                # IMPORTANT: avoid python `if` on symbolic values and avoid mutable temporaries across
                # nested loop frames. We update `scores_chunk[i,j]` in-place using `T.if_then_else`.
                for i, j in T.Parallel(BT, BN):
                    blk = start_n + j
                    # scale (unscaled dot -> /BT), then mask
                    scores_chunk[i, j] = scores_chunk[i, j] / BT
                    scores_chunk[i, j] = T.if_then_else(
                        blk >= NKV_REAL_CONST, -T.infinity(accum_dtype), scores_chunk[i, j]
                    )
                    # Mask TopK blocks: skip them in interval approximation.
                    scores_chunk[i, j] = T.if_then_else(
                        sel_mask[j] != 0, -T.infinity(accum_dtype), scores_chunk[i, j]
                    )

                # Online Softmax update
                T.copy(scores_max, scores_max_prev)
                # Local max in this chunk
                local_max = T.alloc_fragment([BT], accum_dtype)
                T.fill(local_max, -T.infinity(accum_dtype))
                T.reduce_max(scores_chunk, local_max, dim=1, clear=False)
                
                for i in T.Parallel(BT):
                    scores_max[i] = T.max(scores_max[i], local_max[i])
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                
                # Rescale accumulated output
                for j, k in T.Parallel(BT, BV):
                    acc_o[j, k] *= scores_scale[j]
                
                for i in T.Parallel(BT):
                    logsum[i] = logsum[i] * scores_scale[i]

                # Compute exp scores for this chunk
                for i, j in T.Parallel(BT, BN):
                    scores_chunk[i, j] = T.exp2(scores_chunk[i, j] * scale - scores_max[i] * scale)
                
                # Update logsum with this chunk's sum
                chunk_sum = T.alloc_fragment([BT], accum_dtype)
                T.reduce_sum(scores_chunk, chunk_sum, dim=1, clear=True)
                for i in T.Parallel(BT):
                    g_l[i] += chunk_sum[i] * BT

                # Load VC chunk: [BN, BV]
                # Accumulate values into g_acc: Prob [BT, BN] @ VC [BN, BV] -> [BT, BV]
                T.copy(scores_chunk, scores_chunk_cast)
                if chunk_idx % 2 == 0:
                    T.gemm(scores_chunk_cast, VC0, g_acc, policy=T.GemmWarpPolicy.FullRow)
                else:
                    T.gemm(scores_chunk_cast, VC1, g_acc, policy=T.GemmWarpPolicy.FullRow)

                # Ensure the next prefetch is ready before the next iteration uses it
                if chunk_idx + 1 < num_chunks:
                    T.ptx_wait_group(0)

            # Combine sparse + interval contributions and normalize once.
            for i in T.Parallel(BT):
                logsum[i] = logsum[i] + g_l[i]
            for j, k in T.Parallel(BT, BV):
                acc_o[j, k] = acc_o[j, k] + g_acc[j, k]

            # Final Normalize
            for j, k in T.Parallel(BT, BV):
                denom = logsum[j]
                O_shared[j, k] = T.cast(
                    T.if_then_else(denom == 0, 0.0, acc_o[j, k] / denom),
                    dtype,
                )
            
            # Write Output
            T.copy(O_shared, Output[i_bh, i_nq, :, v_col:v_col + BV])
            
            # Write LSE
            # LSE is also independent of VBlocks -> avoid races
            if i_vblk == 0:
                for j in T.Parallel(BT):
                    LSE[i_bh, i_nq, j] = scores_max[j] * scale + T.log2(logsum[j])

    return main


# ---------------------------------------------------------
# Fused kernel (VBlocks==1 only): Sparse TopK + Interval in one launch
# ---------------------------------------------------------

def _build_sparse_fused_vblocks1_kernel(
    BH: int,
    NQ: int,
    NKV: int,   # padded NKV for KC/VC (multiple of BN)
    TopK: int,
    BT: int,
    BK: int,
    BV: int,    # must equal Vdim
    Kdim: int,
    Vdim: int,
    TQ: int,
    TKV: int,
    BN: int = 64,
    NKV_REAL: int = -1,
    num_stages: int = 2,
    threads: int = 128,
    use_swizzle_addr: bool = False,
    use_wgmma: bool = False,
):
    """
    Fused TopK exact attention + interval approximation in one kernel.
    This is a performance fast-path for VBlocks==1 (BV == Vdim), so we avoid
    the intermediate global writes/reads (Out/L/M) between split kernels.
    Grid: (NQ, BH)
    """
    dtype = "bfloat16"
    accum_dtype = "float"
    scale = (1.0 / Kdim) ** 0.5 * 1.44269504  # log2(e)

    q_shape = [BH, TQ, Kdim]
    k_shape = [BH, TKV, Kdim]
    v_shape = [BH, TKV, Vdim]
    kc_shape = [BH, NKV, Kdim]
    vc_shape = [BH, NKV, Vdim]
    indices_shape = [BH, NQ, TopK]
    out_shape = [BH, NQ, BT, Vdim]
    lse_shape = [BH, NQ, BT]

    num_chunks = ceil_div(NKV, BN)
    NKV_REAL_CONST = NKV if NKV_REAL < 0 else NKV_REAL

    # ---- Sparse TopK exact blocks (FlashAttention-style macros + explicit pipeline schedule) ----
    # NOTE: TileLang requires `@T.macro` definitions to live outside the `@T.prim_func` body.
    @T.macro
    def MMA0(
        K: T.Tensor(k_shape, dtype),
        Qs: T.SharedBuffer([BT, BK], dtype),
        Ks: T.SharedBuffer([BT, BK], dtype),
        acc_s: T.FragmentBuffer([BT, BT], accum_dtype),
        blk: T.int32,
        i_bh: T.int32,
    ):
        k0 = blk * BT
        T.copy(K[i_bh, k0:k0 + BT, :], Ks)
        T.clear(acc_s)
        if use_wgmma:
            T.gemm(Qs, Ks, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
            T.wait_wgmma(0)
        else:
            T.gemm(Qs, Ks, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

    @T.macro
    def Softmax(
        acc_s: T.FragmentBuffer([BT, BT], accum_dtype),
        acc_s_cast: T.FragmentBuffer([BT, BT], dtype),
        scores_max: T.FragmentBuffer([BT], accum_dtype),
        scores_max_prev: T.FragmentBuffer([BT], accum_dtype),
        scores_scale: T.FragmentBuffer([BT], accum_dtype),
        scores_sum: T.FragmentBuffer([BT], accum_dtype),
        logsum: T.FragmentBuffer([BT], accum_dtype),
    ):
        T.copy(scores_max, scores_max_prev)
        T.fill(scores_max, -T.infinity(accum_dtype))
        T.reduce_max(acc_s, scores_max, dim=1, clear=False)
        for j in T.Parallel(BT):
            scores_max[j] = T.max(scores_max[j], scores_max_prev[j])
        for j in T.Parallel(BT):
            scores_scale[j] = T.exp2(scores_max_prev[j] * scale - scores_max[j] * scale)
        for j, k in T.Parallel(BT, BT):
            acc_s[j, k] = T.exp2(acc_s[j, k] * scale - scores_max[j] * scale)
        T.reduce_sum(acc_s, scores_sum, dim=1)
        for j in T.Parallel(BT):
            logsum[j] = logsum[j] * scores_scale[j] + scores_sum[j]
        T.copy(acc_s, acc_s_cast)

    @T.macro
    def Rescale(
        acc_o: T.FragmentBuffer([BT, BV], accum_dtype),
        scores_scale: T.FragmentBuffer([BT], accum_dtype),
    ):
        for j, k in T.Parallel(BT, BV):
            acc_o[j, k] *= scores_scale[j]

    @T.macro
    def MMA1(
        V: T.Tensor(v_shape, dtype),
        Vs: T.SharedBuffer([BT, BV], dtype),
        acc_s_cast: T.FragmentBuffer([BT, BT], dtype),
        acc_o: T.FragmentBuffer([BT, BV], accum_dtype),
        blk: T.int32,
        i_bh: T.int32,
    ):
        k0 = blk * BT
        T.copy(V[i_bh, k0:k0 + BT, 0:BV], Vs)
        if use_wgmma:
            T.gemm(acc_s_cast, Vs, acc_o, policy=T.GemmWarpPolicy.FullRow)
            T.wait_wgmma(0)
        else:
            T.gemm(acc_s_cast, Vs, acc_o, policy=T.GemmWarpPolicy.FullRow)

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(k_shape, dtype),
        V: T.Tensor(v_shape, dtype),
        KC: T.Tensor(kc_shape, dtype),
        VC: T.Tensor(vc_shape, dtype),
        Indices: T.Tensor(indices_shape, "int32"),
        Output: T.Tensor(out_shape, dtype),
        LSE: T.Tensor(lse_shape, accum_dtype),
    ):
        with T.Kernel(NQ, BH, threads=threads) as (i_nq, i_bh):
            # Shared buffers
            Q_shared = T.alloc_shared([BT, BK], dtype)
            K_shared = T.alloc_shared([BT, BK], dtype)
            V_shared = T.alloc_shared([BT, BV], dtype)

            KC0 = T.alloc_shared([BN, BK], dtype)
            KC1 = T.alloc_shared([BN, BK], dtype)
            VC0 = T.alloc_shared([BN, BV], dtype)
            VC1 = T.alloc_shared([BN, BV], dtype)

            O_shared = T.alloc_shared([BT, BV], dtype)

            # Fragments
            acc_s = T.alloc_fragment([BT, BT], accum_dtype)
            acc_s_cast = T.alloc_fragment([BT, BT], dtype)
            acc_o = T.alloc_fragment([BT, BV], accum_dtype)
            scores_max = T.alloc_fragment([BT], accum_dtype)
            scores_max_prev = T.alloc_fragment([BT], accum_dtype)
            scores_scale = T.alloc_fragment([BT], accum_dtype)
            scores_sum = T.alloc_fragment([BT], accum_dtype)
            logsum = T.alloc_fragment([BT], accum_dtype)

            scores_chunk = T.alloc_fragment([BT, BN], accum_dtype)
            scores_chunk_cast = T.alloc_fragment([BT, BN], dtype)

            # indices
            indices_local = T.alloc_local([TopK], "int32")
            # For interval masking: mark which j in [0, BN) correspond to TopK blocks in this chunk.
            # This avoids the expensive inner loop (BT*BN*TopK comparisons) when masking scores_chunk.
            sel_mask = T.alloc_local([BN], "int8")
            for t in T.serial(TopK):
                indices_local[t] = Indices[i_bh, i_nq, t]

            # Fused kernel: keep correctness first.
            # We can still ask TileLang to use swizzled shared layouts (often improves GEMM).
            # By default we intentionally do NOT enable `T.use_swizzle(10)` because it has
            # produced observable numerical mismatches in this fused kernel on some revisions.
            T.annotate_layout({
                Q_shared: tl.layout.make_swizzled_layout(Q_shared),
                K_shared: tl.layout.make_swizzled_layout(K_shared),
                V_shared: tl.layout.make_swizzled_layout(V_shared),
                KC0: tl.layout.make_swizzled_layout(KC0),
                KC1: tl.layout.make_swizzled_layout(KC1),
                VC0: tl.layout.make_swizzled_layout(VC0),
                VC1: tl.layout.make_swizzled_layout(VC1),
                O_shared: tl.layout.make_swizzled_layout(O_shared),
            })
            if use_swizzle_addr:
                T.use_swizzle(10)

            # Load Q
            q_start = i_nq * BT
            T.copy(Q[i_bh, q_start:q_start + BT, :], Q_shared)

            # Init online softmax state
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            # ---- Sparse TopK exact blocks ----
            for i in T.Pipelined(TopK, num_stages=num_stages):
                blk = indices_local[i]
                MMA0(K, Q_shared, K_shared, acc_s, blk, i_bh)
                Softmax(acc_s, acc_s_cast, scores_max, scores_max_prev, scores_scale, scores_sum, logsum)
                Rescale(acc_o, scores_scale)
                MMA1(V, V_shared, acc_s_cast, acc_o, blk, i_bh)

            # ---- Interval approximation over KC/VC ----
            g_acc = T.alloc_fragment([BT, BV], accum_dtype)
            g_l = T.alloc_fragment([BT], accum_dtype)
            T.fill(g_acc, 0)
            T.fill(g_l, 0)

            # Prefetch chunk0
            if num_chunks > 0:
                start0 = 0
                with T.attr("default", "async_scope", 1):
                    T.copy(KC[i_bh, start0:start0 + BN, :], KC0)
                with T.attr("default", "async_scope", 1):
                    T.copy(VC[i_bh, start0:start0 + BN, 0:BV], VC0)
                T.ptx_commit_group()
                T.ptx_wait_group(0)

            for chunk_idx in T.serial(num_chunks):
                start_n = chunk_idx * BN

                # Prefetch next
                if chunk_idx + 1 < num_chunks:
                    start_next = (chunk_idx + 1) * BN
                    if (chunk_idx + 1) % 2 == 0:
                        with T.attr("default", "async_scope", 1):
                            T.copy(KC[i_bh, start_next:start_next + BN, :], KC0)
                        with T.attr("default", "async_scope", 1):
                            T.copy(VC[i_bh, start_next:start_next + BN, 0:BV], VC0)
                    else:
                        with T.attr("default", "async_scope", 1):
                            T.copy(KC[i_bh, start_next:start_next + BN, :], KC1)
                        with T.attr("default", "async_scope", 1):
                            T.copy(VC[i_bh, start_next:start_next + BN, 0:BV], VC1)
                    T.ptx_commit_group()

                # scores_chunk = Q @ KC_chunk^T
                T.clear(scores_chunk)
                if chunk_idx % 2 == 0:
                    if use_wgmma:
                        T.gemm(Q_shared, KC0, scores_chunk, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                        T.wait_wgmma(0)
                    else:
                        T.gemm(Q_shared, KC0, scores_chunk, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                else:
                    if use_wgmma:
                        T.gemm(Q_shared, KC1, scores_chunk, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                        T.wait_wgmma(0)
                    else:
                        T.gemm(Q_shared, KC1, scores_chunk, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                # Build selection mask for this chunk: sel_mask[j]=1 if (start_n+j) is one of TopK indices.
                for j in T.Parallel(BN):
                    sel_mask[j] = 0
                for t in T.serial(TopK):
                    idx = indices_local[t] - start_n
                    if idx >= 0:
                        if idx < BN:
                            sel_mask[idx] = 1

                # scale to mean and apply mask
                for i, j in T.Parallel(BT, BN):
                    blk = start_n + j
                    scores_chunk[i, j] = scores_chunk[i, j] / BT
                    scores_chunk[i, j] = T.if_then_else(
                        blk >= NKV_REAL_CONST, -T.infinity(accum_dtype), scores_chunk[i, j]
                    )
                    # Mask TopK blocks: skip them in interval approximation.
                    scores_chunk[i, j] = T.if_then_else(
                        sel_mask[j] != 0, -T.infinity(accum_dtype), scores_chunk[i, j]
                    )

                # online softmax update (interval)
                T.copy(scores_max, scores_max_prev)
                local_max = T.alloc_fragment([BT], accum_dtype)
                T.fill(local_max, -T.infinity(accum_dtype))
                T.reduce_max(scores_chunk, local_max, dim=1, clear=False)

                for i in T.Parallel(BT):
                    scores_max[i] = T.max(scores_max[i], local_max[i])
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for j, k in T.Parallel(BT, BV):
                    acc_o[j, k] *= scores_scale[j]
                for i in T.Parallel(BT):
                    logsum[i] = logsum[i] * scores_scale[i]

                # prob = exp2(scores_chunk*scale - m*scale)
                for i, j in T.Parallel(BT, BN):
                    scores_chunk[i, j] = T.exp2(scores_chunk[i, j] * scale - scores_max[i] * scale)

                chunk_sum = T.alloc_fragment([BT], accum_dtype)
                T.reduce_sum(scores_chunk, chunk_sum, dim=1, clear=True)
                for i in T.Parallel(BT):
                    g_l[i] += chunk_sum[i] * BT

                T.copy(scores_chunk, scores_chunk_cast)
                if chunk_idx % 2 == 0:
                    if use_wgmma:
                        T.gemm(scores_chunk_cast, VC0, g_acc, policy=T.GemmWarpPolicy.FullRow)
                        T.wait_wgmma(0)
                    else:
                        T.gemm(scores_chunk_cast, VC0, g_acc, policy=T.GemmWarpPolicy.FullRow)
                else:
                    if use_wgmma:
                        T.gemm(scores_chunk_cast, VC1, g_acc, policy=T.GemmWarpPolicy.FullRow)
                        T.wait_wgmma(0)
                    else:
                        T.gemm(scores_chunk_cast, VC1, g_acc, policy=T.GemmWarpPolicy.FullRow)

                if chunk_idx + 1 < num_chunks:
                    T.ptx_wait_group(0)

            # combine + normalize
            for i in T.Parallel(BT):
                logsum[i] = logsum[i] + g_l[i]
            for j, k in T.Parallel(BT, BV):
                acc_o[j, k] = acc_o[j, k] + g_acc[j, k]

            for j, k in T.Parallel(BT, BV):
                denom = logsum[j]
                O_shared[j, k] = T.cast(
                    T.if_then_else(denom == 0, 0.0, acc_o[j, k] / denom),
                    dtype,
                )
            T.copy(O_shared, Output[i_bh, i_nq, :, 0:BV])

            for j in T.Parallel(BT):
                LSE[i_bh, i_nq, j] = scores_max[j] * scale + T.log2(logsum[j])

    return main


@tl.jit(
    out_idx=[6, 7],
    pass_configs={
        tl.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tl.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
)
def sparse_fused_vblocks1_kernel(
    BH: int,
    NQ: int,
    NKV: int,   # padded NKV for KC/VC (multiple of BN)
    TopK: int,
    BT: int,
    BK: int,
    BV: int,    # must equal Vdim
    Kdim: int,
    Vdim: int,
    TQ: int,
    TKV: int,
    BN: int = 64,
    NKV_REAL: int = -1,
    num_stages: int = 2,
    threads: int = 128,
    use_swizzle_addr: bool = False,
    use_wgmma: bool = False,
):
    return _build_sparse_fused_vblocks1_kernel(
        BH,
        NQ,
        NKV,
        TopK,
        BT,
        BK,
        BV,
        Kdim,
        Vdim,
        TQ,
        TKV,
        BN=BN,
        NKV_REAL=NKV_REAL,
        num_stages=num_stages,
        threads=threads,
        use_swizzle_addr=use_swizzle_addr,
        use_wgmma=use_wgmma,
    )


def _fused_autotune_configs():
    # Conservative configs: keep BN fixed (host code pads by BN),
    # tune only num_stages and threads (must be a multiple of 32).
    iter_params = dict(
        num_stages=[2, 3, 4, 5],
        threads=[64, 96, 128, 160, 192],
    )
    return [dict(zip(iter_params, values)) for values in itertools.product(*iter_params.values())]


if autotune is not None:
    @autotune(configs=_fused_autotune_configs(), warmup=3, rep=10)
    @tl.jit(
        out_idx=[6, 7],
        pass_configs={
            tl.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tl.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        },
    )
    def sparse_fused_vblocks1_kernel_autotune(
        BH: int,
        NQ: int,
        NKV: int,
        TopK: int,
        BT: int,
        BK: int,
        BV: int,
        Kdim: int,
        Vdim: int,
        TQ: int,
        TKV: int,
        BN: int = 64,
        NKV_REAL: int = -1,
        num_stages: int = 2,
        threads: int = 128,
        use_swizzle_addr: bool = False,
        use_wgmma: bool = False,
    ):
        return _build_sparse_fused_vblocks1_kernel(
            BH,
            NQ,
            NKV,
            TopK,
            BT,
            BK,
            BV,
            Kdim,
            Vdim,
            TQ,
            TKV,
            BN=BN,
            NKV_REAL=NKV_REAL,
            num_stages=num_stages,
            threads=threads,
            use_swizzle_addr=use_swizzle_addr,
            use_wgmma=use_wgmma,
        )

else:  # pragma: no cover
    # Fallback: no autotuner available; keep API but don't tune.
    sparse_fused_vblocks1_kernel_autotune = sparse_fused_vblocks1_kernel


# Alias: preferred entrypoint when you want autotune if available.
sparse_fused_vblocks1_kernel_auto = sparse_fused_vblocks1_kernel_autotune

## NOTE:
# - This file intentionally contains *only* kernel definitions + small helpers.
# - All correctness/benchmark entrypoints live in:
#   - `verify_vs_triton.py`
#   - `bench_sparse_tilelang.py`
