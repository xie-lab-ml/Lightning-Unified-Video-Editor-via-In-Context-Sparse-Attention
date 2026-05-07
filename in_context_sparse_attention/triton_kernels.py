import torch
import triton
import triton.language as tl
import functools
import torch.nn.functional as F
import torch.distributed as dist

def contiguous(fn):
    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        return fn(ctx,
                  *(i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args),
                  **{k: (v if not isinstance(v, torch.Tensor) else v.contiguous()) for k, v in kwargs.items()})
    return wrapper

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [4, 8]
        for num_stages in [3, 4]
    ],
    key=["T"]
)
@triton.jit
def chunk_reduce_kv_kernel(
    k, v, 
    kc, vc, kvc,
    T, N: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr
):
    i_kv, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    
    p_k = tl.make_block_ptr(k + i_bh * T * K, (T, K), (K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * T * V, (T, V), (V, 1), (i_t * BT, 0), (BT, BV), (1, 0))
    
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_v = tl.load(p_v, boundary_check=(0, 1))

    b_kc = tl.sum(b_k, axis=0)
    b_vc = tl.sum(b_v, axis=0)
    
    # Normalize to prevent overflow
    max_k = tl.max(tl.abs(b_k))
    max_v = tl.max(tl.abs(b_v))
    
    # Avoid division by zero
    max_k = tl.where(max_k == 0, 1.0, max_k)
    max_v = tl.where(max_v == 0, 1.0, max_v)
    
    b_k_norm = b_k / max_k
    b_v_norm = b_v / max_v
    
    # kvc = k.T @ v
    b_kvc = tl.dot(tl.trans(b_k_norm.to(b_v.dtype)), b_v_norm.to(b_v.dtype))
    b_kvc = b_kvc * (max_k * max_v)
    p_kc = tl.make_block_ptr(kc + i_bh * N * K + i_t * K, (K,), (1,), (0,), (BK,), (0,))
    p_vc = tl.make_block_ptr(vc + i_bh * N * V + i_t * V, (V,), (1,), (0,), (BV,), (0,))
    
    p_kvc = tl.make_block_ptr(kvc + i_bh * N * K * V + i_t * K * V, (K, V), (V, 1), (0, 0), (BK, BV), (1, 0))
    
    tl.store(p_kc, b_kc.to(p_kc.dtype.element_ty), boundary_check=(0,))
    tl.store(p_vc, b_vc.to(p_vc.dtype.element_ty), boundary_check=(0,))
    tl.store(p_kvc, b_kvc.to(p_kvc.dtype.element_ty), boundary_check=(0, 1))

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [4, 8]
        for num_stages in [3, 4]
    ],
    key=["T"]
)
@triton.jit
def chunk_reduce_q_kernel(
    q, 
    qc,
    T, N: tl.constexpr, K: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr
):
    i_kv, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    
    p_q = tl.make_block_ptr(q + i_bh * T * K, (T, K), (K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_qc = tl.sum(b_q, axis=0)

    p_qc = tl.make_block_ptr(qc + i_bh * N * K + i_t * K, (K,), (1,), (0,), (BK,), (0,))
    tl.store(p_qc, b_qc.to(p_qc.dtype.element_ty), boundary_check=(0,))

def chunk_reduce_1st(q, k, v, block_size):
    B, H, T_Q, K = q.shape
    T_KV = k.shape[2]
    V = v.shape[-1]
    
    N_Q = triton.cdiv(T_Q, block_size)
    N_KV = triton.cdiv(T_KV, block_size)
    
    BK = triton.next_power_of_2(K)
    BV = triton.next_power_of_2(V)
    
    qc = torch.empty(B, H, N_Q, K, device=q.device, dtype=torch.bfloat16)
    kc = torch.empty(B, H, N_KV, K, device=k.device, dtype=torch.bfloat16)
    vc = torch.empty(B, H, N_KV, V, device=v.device, dtype=torch.bfloat16)
    kvc = torch.empty(B, H, N_KV, K, V, device=k.device, dtype=torch.bfloat16)
    
    # 1. Reduce Q
    grid_q = (1, N_Q, B * H)
    chunk_reduce_q_kernel[grid_q](
        q, qc,
        T_Q, N_Q, K, block_size, BK
    )
    
    # 2. Reduce KV
    grid_kv = (1, N_KV, B * H)
    chunk_reduce_kv_kernel[grid_kv](
        k, v, kc, vc, kvc,
        T_KV, N_KV, K, V, block_size, BK, BV
    )
    
    return qc.to(torch.bfloat16), kc.to(torch.bfloat16), vc.to(torch.bfloat16), kvc.to(torch.bfloat16)


@triton.autotune(
    configs=[
        triton.Config({'BN': BN}, num_warps=num_warps, num_stages=num_stages)
        for BN in [16, 32, 64] # Smaller BN might be better for 1st order loop
        for num_warps in [4, 8]
        for num_stages in [1, 2, 3, 4]
    ],
    key=["T"]
)
@triton.jit
def sparse_piecewise_attn_1st_intervals_kernel(
    q, k, v, 
    kc, vc, kvc,
    o, lse, 
    indices,        
    scale, 
    T_Q, T_KV,
    K: tl.constexpr, V: tl.constexpr, 
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    NT_Q: tl.constexpr, NT_KV: tl.constexpr, NS: tl.constexpr,
    BN: tl.constexpr
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    
    p_q = tl.make_tensor_descriptor(
     base = q + i_bh * T_Q * K,
     shape = (T_Q, K),
     strides = (K, 1),
     block_shape = (BT, BK),
    )

    b_q = p_q.load([i_t * BT, 0])
    
    sm_scale = scale * 1.44269504 # log2(e)
    acc = tl.zeros([BT, BV], dtype=tl.float32)
    l_i = tl.zeros((BT,), dtype=tl.float32)
    m_i = tl.zeros((BT,), dtype=tl.float32) - float('inf')
    
    for i in range(NS):
        i_n = tl.load(indices + i_bh * NT_Q * NS + i_t * NS + i).to(tl.int32)
        bos = i_n * BT
        
        p_k = tl.make_tensor_descriptor(
        base = k + i_bh * T_KV * K,
        shape = (T_KV, K),
        strides = (K, 1),
        block_shape = (BT, BK),
        )

        p_v = tl.make_tensor_descriptor(
        base = v + i_bh * T_KV * V,
        shape = (T_KV, V),
        strides = (V, 1),
        block_shape = (BT, BV),
        )

        b_k = p_k.load([bos, 0])
        b_v = p_v.load([bos, i_v * BV])
        
        b_s = tl.dot(b_q, tl.trans(b_k))
        b_s *= sm_scale
        # Mask out if out of bounds (should be handled by padding or boundary check, but explicit mask for safety)
        # Actually T_KV check:
        b_s += tl.where((bos + tl.arange(0, BT))[None, :] < T_KV, 0, float("-inf"))
        
        new_m_i = tl.maximum(m_i, tl.max(b_s, -1))
        alpha = tl.math.exp2(m_i - new_m_i)
        score = tl.math.exp2(b_s - new_m_i[:, None])
        
        l_i = l_i * alpha + tl.sum(score, -1)
        acc = acc * alpha[:, None] + tl.dot(score.to(b_v.dtype), b_v)
        m_i = new_m_i

    B_NS: tl.constexpr = triton.next_power_of_2(NS)
    offs_n_idx = tl.arange(0, B_NS)
    loaded_indices = tl.load(
        indices + i_bh * NT_Q * NS + i_t * NS + offs_n_idx, 
        mask=offs_n_idx < NS, 
        other=-1
    )

    g_acc = tl.zeros([BT, BV], dtype=tl.float32)
    g_l = tl.zeros([BT], dtype=tl.float32)

    for start_n in range(0, NT_KV, BN):

        p_kc = tl.make_tensor_descriptor(
        base = kc + i_bh * NT_KV * K,
        shape = (NT_KV, K),
        strides = (K, 1),
        block_shape = (BN, BK),
        )
        b_kc = p_kc.load([start_n, 0])
        
        b_s_mean = tl.dot(b_q, tl.trans(b_kc))
        b_s_mean = b_s_mean * sm_scale / BT
        
        chunk_indices = start_n + tl.arange(0, BN)
        is_in_indices = chunk_indices[:, None] == loaded_indices[None, :]
        mask_is_selected = tl.max(is_in_indices, axis=1)
        valid_mask = (chunk_indices < NT_KV) & (mask_is_selected == 0)
        
        b_s_mean = tl.where(valid_mask[None, :], b_s_mean, float("-inf"))
        
        new_m_i = tl.maximum(m_i, tl.max(b_s_mean, 1))
        alpha = tl.math.exp2(m_i - new_m_i)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha
        m_i = new_m_i

        prob_chunk = tl.math.exp2(b_s_mean - m_i[:, None]) # [BT, BN]

        p_vc = tl.make_tensor_descriptor(
        base = vc + i_bh * NT_KV * V,
        shape = (NT_KV, V),
        strides = (V, 1),
        block_shape = (BN, BV),
        )
        b_vc = p_vc.load([start_n, i_v * BV])
        g_acc += tl.dot(prob_chunk.to(b_vc.dtype), b_vc)
        g_l += tl.sum(prob_chunk, axis=1) * BT

    acc += g_acc
    l_i += g_l
    acc = acc / l_i[:, None]
    m_i += tl.math.log2(l_i)
    

    p_o = tl.make_tensor_descriptor(
        base = o + i_bh * T_Q * V,
        shape = (T_Q, V),
        strides = (V, 1),
        block_shape = (BT, BV),
        )
    p_lse = tl.make_tensor_descriptor(
        base = lse + i_bh * T_Q,
        shape = (T_Q,),
        strides = (1,),
        block_shape = (BT,),
        )
    p_o.store([i_t * BT, i_v * BV], acc.to(p_o.dtype))
    p_lse.store([i_t * BT], m_i.to(p_lse.dtype))

@triton.jit
def sparse_piecewise_attn_1st_intervals_bwd_kernel(
    do, q, k, v, 
    kc, vc,
    o, lse, 
    dq, dk, dv,
    indices,        
    scale, 
    T_Q, T_KV,
    K: tl.constexpr, V: tl.constexpr, 
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    NT_Q, NT_KV, 
    NT_KV_PADDED: tl.constexpr,
    NS: tl.constexpr,
    BN: tl.constexpr = 64 # Default fixed value
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    
    # Load Q, DO, O, LSE
    p_q = tl.make_tensor_descriptor(
        base = q + i_bh * T_Q * K,
        shape = (T_Q, K),
        strides = (K, 1),
        block_shape = (BT, BK),
    )
    b_q = p_q.load([i_t * BT, 0])
    
    p_do = tl.make_tensor_descriptor(
        base = do + i_bh * T_Q * V,
        shape = (T_Q, V),
        strides = (V, 1),
        block_shape = (BT, BV),
    )
    b_do = p_do.load([i_t * BT, i_v * BV])
    
    p_o = tl.make_tensor_descriptor(
        base = o + i_bh * T_Q * V,
        shape = (T_Q, V),
        strides = (V, 1),
        block_shape = (BT, BV),
    )
    b_o = p_o.load([i_t * BT, i_v * BV])
    
    p_lse = tl.make_tensor_descriptor(
        base = lse + i_bh * T_Q,
        shape = (T_Q,),
        strides = (1,),
        block_shape = (BT,),
    )
    b_lse = p_lse.load([i_t * BT]) # [BT]
    
    # D = (do * o).sum(-1)
    # Use float32 for D calculation to avoid precision loss
    b_D = tl.sum(b_do.to(tl.float32) * b_o.to(tl.float32), axis=1) # [BT]
    
    sm_scale = scale * 1.44269504 # log2(e)
    
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    
    # --- Sparse Part ---
    for i in range(NS):
        i_n = tl.load(indices + i_bh * NT_Q * NS + i_t * NS + i).to(tl.int32)
        bos = i_n * BT
        
        p_k = tl.make_tensor_descriptor(
            base = k + i_bh * T_KV * K,
            shape = (T_KV, K),
            strides = (K, 1),
            block_shape = (BT, BK),
        )
        p_v = tl.make_tensor_descriptor(
            base = v + i_bh * T_KV * V,
            shape = (T_KV, V),
            strides = (V, 1),
            block_shape = (BT, BV),
        )
        
        b_k = p_k.load([bos, 0])
        b_v = p_v.load([bos, i_v * BV])
        
        # S = Q K^T
        b_s = tl.dot(b_q, tl.trans(b_k))
        b_s *= sm_scale
        b_s += tl.where((bos + tl.arange(0, BT))[None, :] < T_KV, 0, float("-inf"))
        
        # P = exp2(S - lse)
        b_p = tl.math.exp2(b_s - b_lse[:, None]) # [BT, BT]
        
        # dS = P * (do @ v.T - D)
        # term1 = do @ v.T
        b_term1 = tl.dot(b_do, tl.trans(b_v))
        b_ds = b_p * (b_term1 - b_D[:, None])
        
        # dq += dS @ k * scale
        # Use float32 for accumulation
        b_dq += tl.dot(b_ds.to(tl.float32), b_k.to(tl.float32)) * scale
        
        # dk += dS.T @ q * scale
        b_dk_update = tl.dot(tl.trans(b_ds).to(tl.float32), b_q.to(tl.float32)) * scale
        
        # dv += P.T @ do
        b_dv_update = tl.dot(tl.trans(b_p).to(b_do.dtype), b_do)
        
        # Atomic Add to global memory
        offs_k_m = bos + tl.arange(0, BT)
        offs_k_n = tl.arange(0, BK)
        ptr_dk = dk + (i_bh * T_KV * K + offs_k_m[:, None] * K + offs_k_n[None, :])
        tl.atomic_add(ptr_dk, b_dk_update)
        
        offs_v_m = bos + tl.arange(0, BT)
        offs_v_n = i_v * BV + tl.arange(0, BV)
        ptr_dv = dv + (i_bh * T_KV * V + offs_v_m[:, None] * V + offs_v_n[None, :])
        tl.atomic_add(ptr_dv, b_dv_update)

    # --- Interval Part ---
    B_NS: tl.constexpr = triton.next_power_of_2(NS)
    offs_n_idx = tl.arange(0, B_NS)
    loaded_indices = tl.load(
        indices + i_bh * NT_Q * NS + i_t * NS + offs_n_idx, 
        mask=offs_n_idx < NS, 
        other=-1
    )
    
    for start_n in range(0, NT_KV, BN):
        p_kc = tl.make_tensor_descriptor(
            base = kc + i_bh * NT_KV_PADDED * K,
            shape = (NT_KV_PADDED, K),
            strides = (K, 1),
            block_shape = (BN, BK),
        )
        b_kc = p_kc.load([start_n, 0])
        
        # S_mean = Q * kc^T * scale / BT
        b_s_mean = tl.dot(b_q, tl.trans(b_kc))
        b_s_mean = b_s_mean * sm_scale / BT
        
        chunk_indices = start_n + tl.arange(0, BN)
        is_in_indices = chunk_indices[:, None] == loaded_indices[None, :]
        mask_is_selected = tl.max(is_in_indices, axis=1)
        valid_mask = (chunk_indices < NT_KV) & (mask_is_selected == 0)
        
        b_s_mean = tl.where(valid_mask[None, :], b_s_mean, float("-inf"))
        
        # P = exp2(S_mean - lse)
        b_p = tl.math.exp2(b_s_mean - b_lse[:, None]) # [BT, BN]
        
        p_vc = tl.make_tensor_descriptor(
            base = vc + i_bh * NT_KV_PADDED * V,
            shape = (NT_KV_PADDED, V),
            strides = (V, 1),
            block_shape = (BN, BV),
        )
        b_vc = p_vc.load([start_n, i_v * BV])
        
        # dS = P * (do @ vc.T - D * BT)
        b_term1 = tl.dot(b_do, tl.trans(b_vc))
        b_term2 = b_D[:, None] * BT
        b_ds = b_p * (b_term1 - b_term2)
        
        # dq += dS @ kc * (scale / BT)
        # Use float32 for accumulation
        b_dq += tl.dot(b_ds.to(tl.float32), b_kc.to(tl.float32)) * (scale / BT)
        
        # dKc = dS.T @ q * scale
        b_dkc = tl.dot(tl.trans(b_ds).to(tl.float32), b_q.to(tl.float32)) * scale # [BN, BK]
        
        # dVc = P.T @ do
        b_dvc = tl.dot(tl.trans(b_p).to(b_do.dtype), b_do) # [BN, BV]
        
        # Distribute to dk, dv
        # dk_j += dKc_c / BT
        # dv_j += dVc_c
        
        for i in range(BN):
            chunk_idx = start_n + i
            if chunk_idx < NT_KV:
                # Check if this chunk was actually valid (not selected)
                # We computed dS based on valid_mask. If invalid, dS is 0 (due to -inf in S -> P=0).
                # So b_dkc/b_dvc will be 0 for invalid chunks.
                # We can just add.
                
                bos = chunk_idx * BT
                
                # dk
                offs_k_m = bos + tl.arange(0, BT)
                offs_k_n = tl.arange(0, BK)
                ptr_dk = dk + (i_bh * T_KV * K + offs_k_m[:, None] * K + offs_k_n[None, :])
                
                # val to add: b_dkc[i] / BT
                mask_i = (tl.arange(0, BN) == i)[:, None]
                row_dkc = tl.sum(b_dkc * mask_i, axis=0)
                
                # Broadcast to [BT, BK]
                val_dk = row_dkc[None, :] / BT
                val_dk = val_dk + tl.zeros([BT, BK], dtype=val_dk.dtype)
                tl.atomic_add(ptr_dk, val_dk)
                
                # dv
                offs_v_m = bos + tl.arange(0, BT)
                offs_v_n = i_v * BV + tl.arange(0, BV)
                ptr_dv = dv + (i_bh * T_KV * V + offs_v_m[:, None] * V + offs_v_n[None, :])
                
                # val to add: b_dvc[i]
                # mask_i is same
                row_dvc = tl.sum(b_dvc * mask_i, axis=0)
                # Broadcast to [BT, BV]
                val_dv = row_dvc[None, :]
                val_dv = val_dv + tl.zeros([BT, BV], dtype=val_dv.dtype)
                tl.atomic_add(ptr_dv, val_dv)

    # Store dq
    # We can use TMA store or standard store.
    # But other warps might be processing same Q block?
    # No, grid is (V/BV, NT_Q, B*H).
    # Different i_v will process same Q block.
    # So we need atomic add for dq as well!
    # So we MUST atomic add `b_dq`
    offs_q_m = i_t * BT + tl.arange(0, BT)
    offs_q_n = tl.arange(0, BK)
    ptr_dq = dq + (i_bh * T_Q * K + offs_q_m[:, None] * K + offs_q_n[None, :])
    
    tl.atomic_add(ptr_dq, b_dq)


class SparsePiecewiseAttn1stIntervalsFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sparsity, block_size):
        # Forward logic
        B, H, T_Q, K = q.shape
        T_KV = k.shape[2]
        V = v.shape[-1]
        
        qc, kc, vc, kvc = chunk_reduce_1st(q, k, v, block_size)
        
        # Top-K
        p = torch.einsum('bhik,bhjk->bhij', qc.to(kc.dtype), kc)
        top_k = max(1, int(sparsity * (T_KV // block_size)))
        indices = torch.topk(p, k=top_k, dim=-1).indices.to(torch.int32)
        
        o = torch.empty(B, H, T_Q, V, device=q.device, dtype=v.dtype)
        lse = torch.empty(B, H, T_Q, device=q.device, dtype=torch.float32)
        
        BT = block_size
        BK = triton.next_power_of_2(K)
        BV = triton.next_power_of_2(V)
        NT_Q = triton.cdiv(T_Q, BT)
        NT_KV = triton.cdiv(T_KV, BT)
        NS = indices.shape[-1]
        
        grid = (triton.cdiv(V, BV), NT_Q, B * H)
        
        # TMA descriptors require a global memory allocation
        def alloc_fn(size: int, alignment: int, stream):
            return torch.empty(size, device="cuda", dtype=torch.int8)

        triton.set_allocator(alloc_fn)

        sparse_piecewise_attn_1st_intervals_kernel[grid](
            q, k, v,
            kc, vc, kvc,
            o, lse,
            indices,
            K ** -0.5,
            T_Q, T_KV, K, V,
            BT, BK, BV,
            NT_Q, NT_KV, NS,
        )
        
        ctx.save_for_backward(q, k, v, kc, vc, o, lse, indices)
        ctx.scale = K ** -0.5
        ctx.block_size = block_size
        ctx.T_Q = T_Q
        ctx.T_KV = T_KV
        
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, kc, vc, o, lse, indices = ctx.saved_tensors
        scale = ctx.scale
        block_size = ctx.block_size
        T_Q = ctx.T_Q
        T_KV = ctx.T_KV
        
        B, H, _, K = q.shape
        V = v.shape[-1]
        
        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.zeros_like(k, dtype=torch.float32)
        dv = torch.zeros_like(v, dtype=torch.float32)
        
        BT = block_size
        BK = triton.next_power_of_2(K)
        BV = triton.next_power_of_2(V)
        NT_Q = triton.cdiv(T_Q, BT)
        NT_KV = triton.cdiv(T_KV, BT)
        NS = indices.shape[-1]
        
        grid = (triton.cdiv(V, BV), NT_Q, B * H)

        # TMA descriptors require a global memory allocation
        def alloc_fn(size: int, alignment: int, stream):
            return torch.empty(size, device="cuda", dtype=torch.int8)

        triton.set_allocator(alloc_fn)
        
        # Pad kc and vc to ensure TMA loads are safe (multiple of BN)
        # Max BN is 64 in config, but let's be safe with 128
        pad_len = (128 - (NT_KV % 128)) % 128
        if pad_len > 0:
            kc_padded = F.pad(kc, (0, 0, 0, pad_len))
            vc_padded = F.pad(vc, (0, 0, 0, pad_len))
        else:
            kc_padded = kc
            vc_padded = vc
            
        sparse_piecewise_attn_1st_intervals_bwd_kernel[grid](
            do, q, k, v,
            kc_padded, vc_padded,
            o, lse,
            dq, dk, dv,
            indices,
            scale,
            T_Q, T_KV, K, V,
            BT, BK, BV,
            NT_Q, NT_KV, 
            kc_padded.shape[2], # NT_KV_PADDED
            NS,
            BN=64,
            num_warps=1,
            num_stages=1
        )
        
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None

