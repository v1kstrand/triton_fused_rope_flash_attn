import os
os.environ["TORCHDYNAMO_VERBOSE"]="1"
os.environ["TRITON_PRINT_AUTOTUNING"]="1"
import torch
from torch import Tensor
import triton
import triton.language as tl


import math
from torch import nn

torch.backends.cuda.matmul.allow_tf32 = True

GROUP_NM_SWEEP = [8]
NUM_STAGES_SWEEP = [3, 4]
NUM_WARPS_SWEEP = [4, 8]
KEY_CACHE = ["BATCH_SIZE", "NUM_HEADS", "SEQ_LEN", "HEAD_DIM"]

def _sdpa_comp_dtype(x: torch.Tensor) -> torch.dtype:
    return torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled() else x.dtype

def _triton_compute_dtype(dtype: torch.dtype):
    if dtype is torch.float16:
        return tl.float16
    if dtype is torch.bfloat16:
        return tl.bfloat16
    if dtype is torch.float32:
        return tl.float32
    raise ValueError(f"Unsupported compute dtype for Triton SDPA: {dtype}")

@triton.autotune(
    [
        triton.Config(
            {"BLOCK_Q": BLOCK_Q, "BLOCK_KV": BLOCK_KV, "GROUP_M": GROUP_M},
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for BLOCK_Q in [64, 128]
        for BLOCK_KV in [32, 64]
        for GROUP_M in GROUP_NM_SWEEP
        for num_stages in NUM_STAGES_SWEEP
        for num_warps in NUM_WARPS_SWEEP
    ],
    key=KEY_CACHE,
)
@triton.jit
def _attn_fwd(
    Q, K, V, M, O,
    sqb, sqh, sqs, sqd,
    skb, skh, sks, skd,
    svb, svh, svs, svd,
    sob, soh, sos, sod,
    NUM_HEADS: tl.constexpr, SEQ_LEN: tl.constexpr, HEAD_DIM: tl.constexpr,
    softmax_scale:tl.constexpr, BLOCK_Q: tl.constexpr, BLOCK_KV: tl.constexpr, 
    DTYPE: tl.constexpr, GROUP_M: tl.constexpr,
):
    tl.static_assert(BLOCK_KV <= HEAD_DIM)
    pid_m  = tl.program_id(0)
    pid_bh = tl.program_id(1)

    num_tiles_m   = tl.cdiv(SEQ_LEN, BLOCK_Q)
    group_id      = pid_m // GROUP_M
    tiles_in_this = tl.minimum(GROUP_M, num_tiles_m - group_id*GROUP_M)

    m_in_grp      = pid_m - group_id*GROUP_M                        # 0..GROUP_M-1
    m_in_grp_eff  = m_in_grp % tiles_in_this                        # clamp to tail size
    rot           = pid_bh % tiles_in_this
    m_swizzled    = group_id*GROUP_M + ((m_in_grp_eff + rot) % tiles_in_this)

    start_q       = m_swizzled * BLOCK_Q
    if start_q >= SEQ_LEN:
        return

    b = pid_bh // NUM_HEADS
    h  = pid_bh %  NUM_HEADS
    off_bh_k  = (b * skb   + h * skh ).to(tl.int64)
    off_bh_v  = (b * svb   + h * svh ).to(tl.int64)
    off_bh_q  = (b * sqb   + h * sqh ).to(tl.int64)
    off_bh_o  = (b * sob   + h * soh ).to(tl.int64)
    
    # --- block pointers ---
    Q_block_ptr = tl.make_block_ptr(
        Q + off_bh_q, (SEQ_LEN, HEAD_DIM), (sqs, sqd), (start_q, 0), (BLOCK_Q, HEAD_DIM), (1, 0)
    )
    V_block_ptr = tl.make_block_ptr(
        V + off_bh_v, (SEQ_LEN, HEAD_DIM), (svs, svd), (0, 0), (BLOCK_KV, HEAD_DIM), (1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        K + off_bh_k, (HEAD_DIM, SEQ_LEN), (skd, sks), (0, 0), (HEAD_DIM, BLOCK_KV), (0, 1)
    )
    O_block_ptr = tl.make_block_ptr(
        O + off_bh_o, (SEQ_LEN, HEAD_DIM), (sos, sod), (start_q, 0), (BLOCK_Q, HEAD_DIM), (1, 0)
    )

    # --- per-row running stats + output tile ---
    m_i = tl.full((BLOCK_Q,), -float("inf"), dtype=tl.float32)
    l_i = tl.full((BLOCK_Q,),  0.0,          dtype=tl.float32)
    O_block = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)

    # --- inner loop over KV tiles (online softmax) ---
    s = tl.full([1], softmax_scale, dtype=DTYPE)
    Q_block = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero").to(DTYPE) * s
    offs_kv = tl.arange(0, BLOCK_KV)
    for start_kv in range(0, SEQ_LEN, BLOCK_KV):
        K_block = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero").to(DTYPE)
        S = tl.dot(Q_block, K_block, tl.zeros((BLOCK_Q, BLOCK_KV), tl.float32))

        kv_valid = start_kv + offs_kv < SEQ_LEN
        S = tl.where(kv_valid[None, :], S, -float("inf"))

        m_ij    = tl.maximum(m_i, tl.max(S, axis=1))
        P_block = tl.exp(S - m_ij[:, None])
        l_ij = tl.sum(P_block, axis=1)

        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij

        V_block = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero").to(DTYPE)
        O_block = O_block * alpha[:, None]
        O_block = tl.dot(P_block.to(DTYPE), V_block, O_block)

        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_KV, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_KV))
    
    # --- write back: store log-sum-exp (for bwd) and O ---
    offs_q  = start_q + tl.arange(0, BLOCK_Q)
    m_ptrs = M + pid_bh * SEQ_LEN + offs_q
    tl.store(m_ptrs, m_i + tl.log(l_i + 1e-20), mask=offs_q < SEQ_LEN)
    O_block = O_block / l_i[:, None]
    tl.store(O_block_ptr, O_block.to(O.type.element_ty), boundary_check=(0, 1))

@triton.autotune(
    [triton.Config({"BLOCK_Q": bq}, num_stages=ns, num_warps=nw)
     for bq in [32, 64, 128]
     for ns in NUM_STAGES_SWEEP
     for nw in NUM_WARPS_SWEEP],
    key=KEY_CACHE,
)
@triton.jit
def _attn_bwd_preprocess(
    O, dO, D,
    sOb, sOh, sOs, sOd,          # O strides
    sdb, sdh, sds, sdd,          # dO strides
    NUM_HEADS: tl.constexpr, SEQ_LEN: tl.constexpr,
    BLOCK_Q: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    pid_q  = tl.program_id(0)                          # Q-tile id
    pid_bh = tl.program_id(1)                          # packed (batch, head)
    start_q = pid_q * BLOCK_Q
    if start_q >= SEQ_LEN:
        return

    b = pid_bh // NUM_HEADS
    h = pid_bh %  NUM_HEADS
    off_bh_O  = (b * sOb  + h * sOh ).to(tl.int64)
    off_bh_dO = (b * sdb  + h * sdh ).to(tl.int64)

    O_blk = tl.make_block_ptr(
        O + off_bh_O, (SEQ_LEN, HEAD_DIM), (sOs, sOd), (start_q, 0), (BLOCK_Q, HEAD_DIM), (1, 0)
    )
    dO_blk = tl.make_block_ptr(
        dO + off_bh_dO, (SEQ_LEN, HEAD_DIM), (sds, sdd), (start_q, 0), (BLOCK_Q, HEAD_DIM), (1, 0)
    )

    O_block  = tl.load(O_blk,  boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    dO_block = tl.load(dO_blk, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    D_block  = tl.sum(dO_block * O_block, axis=1)

    offs_q = start_q + tl.arange(0, BLOCK_Q)
    tl.store(D + pid_bh * SEQ_LEN + offs_q, D_block, mask=offs_q < SEQ_LEN)


@triton.autotune(
    [
        triton.Config(
            {"BLOCK_Q": BLOCK_Q, "BLOCK_KV": BLOCK_KV, "GROUP_N": GROUP_N},
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for BLOCK_Q in [32, 64]
        for BLOCK_KV in [64, 128]
        for GROUP_N in GROUP_NM_SWEEP
        for num_stages in NUM_STAGES_SWEEP
        for num_warps in NUM_WARPS_SWEEP
    ],
    key=KEY_CACHE,
)
@triton.jit
def _attn_bwd_dk_dv(
    Q, K, V, dO, dK, dV, M, D,
    sqb, sqh, sqs, sqd,# Q strides
    skb, skh, sks, skd,# K strides
    svb, svh, svs, svd,# V strides
    sob, soh, sos, sod,# dO strides
    s_dkb, s_dkh, s_dks, s_dkd,# dK strides
    s_dvb, s_dvh, s_dvs, s_dvd,# dV strides
    NUM_HEADS: tl.constexpr, SEQ_LEN: tl.constexpr,
    BLOCK_Q: tl.constexpr, BLOCK_KV: tl.constexpr, softmax_scale: tl.constexpr,
    HEAD_DIM: tl.constexpr, DTYPE: tl.constexpr, GROUP_N: tl.constexpr
):
    # --- program ids ---
    pid_kv = tl.program_id(0)                 # which KV block
    pid_bh = tl.program_id(1)                 # packed (batch, head)
    b = pid_bh // NUM_HEADS
    h = pid_bh %  NUM_HEADS

    # --- base offsets for this (batch, head) slice ---
    off_bh_seq = (pid_bh * SEQ_LEN).to(tl.int64)
    M  += off_bh_seq
    D  += off_bh_seq

    num_tiles_kv = tl.cdiv(SEQ_LEN, BLOCK_KV)
    group_id     = pid_kv // GROUP_N
    group_start  = group_id * GROUP_N
    if group_start >= num_tiles_kv:
        return
    
    tiles_in_this = tl.minimum(GROUP_N, num_tiles_kv - group_start)
    kv_in_grp     = pid_kv - group_start
    kv_eff        = kv_in_grp % tiles_in_this
    rot           = pid_bh % tiles_in_this
    kv_tile_id    = group_start + ((kv_eff + rot) % tiles_in_this)

    start_kv = kv_tile_id * BLOCK_KV
    if start_kv >= SEQ_LEN:
        return

    off_bh_k  = (b * skb   + h * skh  ).to(tl.int64)
    off_bh_v  = (b * svb   + h * svh  ).to(tl.int64)
    off_bh_dk = (b * s_dkb + h * s_dkh).to(tl.int64)
    off_bh_dv = (b * s_dvb + h * s_dvh).to(tl.int64)
    off_bh_q  = (b * sqb   + h * sqh  ).to(tl.int64)
    off_bh_do = (b * sob   + h * soh  ).to(tl.int64)
    
    K_blk = tl.make_block_ptr( 
        K + off_bh_k, (SEQ_LEN, HEAD_DIM), (sks, skd),(start_kv, 0),(BLOCK_KV, HEAD_DIM),(1, 0)
    ) #  base,        shape,               strides,    offsets,      block_shape,        order
    V_blk = tl.make_block_ptr( 
        V + off_bh_v,(SEQ_LEN, HEAD_DIM), (svs, svd),(start_kv, 0),(BLOCK_KV, HEAD_DIM),(1, 0)
    )
    dK_blk = tl.make_block_ptr( 
        dK + off_bh_dk, (SEQ_LEN, HEAD_DIM), (s_dks, s_dkd), (start_kv, 0), (BLOCK_KV, HEAD_DIM), (1, 0)
    )
    dV_blk = tl.make_block_ptr( 
        dV + off_bh_dv,(SEQ_LEN, HEAD_DIM),(s_dvs, s_dvd),(start_kv, 0),(BLOCK_KV, HEAD_DIM),(1, 0)
    )
    Q_T_blk = tl.make_block_ptr( 
        Q + off_bh_q,(HEAD_DIM, SEQ_LEN),(sqd, sqs),(0, 0),(HEAD_DIM, BLOCK_Q),(0, 1)
    )
    dO_blk = tl.make_block_ptr( 
        dO + off_bh_do,(SEQ_LEN, HEAD_DIM),(sos, sod),(0, 0),(BLOCK_Q, HEAD_DIM),(1, 0)
    )

    dV_acc = tl.zeros((BLOCK_KV, HEAD_DIM), dtype=tl.float32)
    dK_acc = tl.zeros((BLOCK_KV, HEAD_DIM), dtype=tl.float32)
    s_fp32 = tl.full([1], softmax_scale, dtype=tl.float32)
    
    V_block = tl.load(V_blk, boundary_check=(0, 1), padding_option="zero").to(DTYPE)
    K_block = tl.load(K_blk, boundary_check=(0, 1), padding_option="zero").to(DTYPE)
    K_block = (K_block.to(tl.float32) * s_fp32).to(DTYPE)

    offs_kv  = start_kv + tl.arange(0, BLOCK_KV)
    kv_valid = offs_kv < SEQ_LEN
    
    num_steps = tl.cdiv(SEQ_LEN, BLOCK_Q)
    for qi in range(0, num_steps):
        start_q = qi * BLOCK_Q
        offs_q  = start_q + tl.arange(0, BLOCK_Q)
        q_valid = offs_q < SEQ_LEN               # <-- define inside loop for this tile
        mask    = kv_valid[:, None] & q_valid[None, :]

        # loads (boundary_check keeps you memory-safe; q_valid handles logic)
        qT_block = tl.load(Q_T_blk, boundary_check=(0, 1), padding_option="zero").to(DTYPE)
        dO_block = tl.load(dO_blk, boundary_check=(0, 1), padding_option="zero").to(DTYPE)

        # logically mask Q tile
        qT_block = tl.where(q_valid[None, :], qT_block, 0)
        dO_block = tl.where(q_valid[:,   None], dO_block, 0)

        # rowwise M and D for these queries
        m  = tl.load(M + offs_q, mask=q_valid, other=0.0).to(tl.float32)
        Di = tl.load(D + offs_q, mask=q_valid, other=0.0).to(tl.float32)

        # logits and probs; mask rows (kv) and cols (q) before exp
        S_T = tl.dot(K_block, qT_block)                     # [BLOCK_KV, BLOCK_Q]
        S_T = tl.where(mask, S_T, -float("inf"))
        P_T = tl.exp(S_T.to(tl.float32) - m[None, :])       # masked cols → 0

        # dV += Pᵀ @ dO
        dV_acc = tl.dot(P_T.to(DTYPE), dO_block, dV_acc)

        # dpᵀ = V @ dOᵀ ; dSᵀ = Pᵀ * (dpᵀ − Di)
        dpT   = tl.dot(V_block, tl.trans(dO_block)).to(tl.float32)
        dS_T  = (P_T * (dpT - Di[None, :])).to(DTYPE)
        dK_acc = tl.dot(dS_T, tl.trans(qT_block), dK_acc)

        # advance block pointers
        Q_T_blk = tl.advance(Q_T_blk, (0, BLOCK_Q))
        dO_blk  = tl.advance(dO_blk,  (BLOCK_Q, 0))

    dK_acc = (dK_acc * s_fp32).to(dK.type.element_ty)
    tl.store(dK_blk, dK_acc                       , boundary_check=(0, 1))
    tl.store(dV_blk, dV_acc.to(dV.type.element_ty), boundary_check=(0, 1))
    

@triton.autotune(
    [
        triton.Config(
            {"BLOCK_Q": BLOCK_Q, "BLOCK_KV": BLOCK_KV, "GROUP_N": GROUP_N},
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for BLOCK_Q in [64, 128]
        for BLOCK_KV in [32, 64]
        for GROUP_N in GROUP_NM_SWEEP
        for num_stages in NUM_STAGES_SWEEP
        for num_warps in NUM_WARPS_SWEEP
    ],
    key=KEY_CACHE,
)
@triton.jit
def _attn_bwd_dq(
    Q, K, V, dO, dQ, M, D,
    sqb, sqh, sqs, sqd, # Q strides
    skb, skh, sks, skd, # K strides
    svb, svh, svs, svd, # V strides
    sob, soh, sos, sod, # dO strides
    s_dqb, s_dqh, s_dqs, s_dqd, # dK strides
    NUM_HEADS: tl.constexpr , SEQ_LEN: tl.constexpr,
    BLOCK_Q: tl.constexpr, BLOCK_KV: tl.constexpr, 
    HEAD_DIM: tl.constexpr, DTYPE: tl.constexpr,
    GROUP_N: tl.constexpr, softmax_scale: tl.constexpr,
):
    pid_bh = tl.program_id(1)
    b = pid_bh // NUM_HEADS
    h = pid_bh %  NUM_HEADS
    
    off_bh_seq = (pid_bh * SEQ_LEN).to(tl.int64)
    M += off_bh_seq
    D += off_bh_seq

    # --- GROUP_M swizzle over Q tiles (tail-safe) ---
    pid_q = tl.program_id(0)
    num_tiles_m   = tl.cdiv(SEQ_LEN, BLOCK_Q)
    group_id      = pid_q // GROUP_N
    group_start   = group_id * GROUP_N
    
    if group_start >= num_tiles_m:
        return
    tiles_in_this = tl.minimum(GROUP_N, num_tiles_m - group_start)
    m_in_grp      = pid_q - group_start
    m_eff         = m_in_grp % tiles_in_this
    rot           = pid_bh % tiles_in_this
    m_swizzled    = group_start + ((m_eff + rot) % tiles_in_this)

    start_q = m_swizzled * BLOCK_Q
    if start_q >= SEQ_LEN:
        return
    
    off_bh_k  = (b * skb   + h * skh  ).to(tl.int64)
    off_bh_v  = (b * svb   + h * svh  ).to(tl.int64)
    off_bh_dq = (b * s_dqb + h * s_dqh).to(tl.int64)
    off_bh_q  = (b * sqb   + h * sqh  ).to(tl.int64)
    off_bh_do = (b * sob   + h * soh  ).to(tl.int64)
    
    # ---------- block pointers ----------
    Q_blk = tl.make_block_ptr(
        Q + off_bh_q,(SEQ_LEN, HEAD_DIM),(sqs, sqd),(start_q, 0),(BLOCK_Q, HEAD_DIM),(1, 0),
    ) #  base,        shape,               strides,    offsets,   block_shape,       order
    dO_blk = tl.make_block_ptr(
        dO + off_bh_do,(SEQ_LEN, HEAD_DIM),(sos, sod),(start_q, 0),(BLOCK_Q, HEAD_DIM),(1, 0),
    )
    K_T_blk = tl.make_block_ptr(
        K + off_bh_k,(HEAD_DIM, SEQ_LEN),(skd, sks),(0, 0),(HEAD_DIM, BLOCK_KV),(0, 1),
    )
    V_T_blk = tl.make_block_ptr(
        V + off_bh_v,(HEAD_DIM, SEQ_LEN),(svd, svs),(0, 0),(HEAD_DIM, BLOCK_KV),(0, 1),
    )
    dQ_blk = tl.make_block_ptr(
        dQ + off_bh_dq,(SEQ_LEN, HEAD_DIM),(s_dqs, s_dqd),(start_q, 0),(BLOCK_Q, HEAD_DIM),(1, 0),
    )

    # ---------- indices & constants ----------
    offs_q = start_q + tl.arange(0, BLOCK_Q)
    offs_kv = tl.arange(0, BLOCK_KV)

    # row-wise scalars
    m  = tl.load(M + offs_q, mask=offs_q < SEQ_LEN, other=0.0).to(tl.float32)[:, None]  # [BLOCK_Q,1], FP32
    Di = tl.load(D + offs_q, mask=offs_q < SEQ_LEN, other=0.0).to(tl.float32)          # [BLOCK_Q], FP32
    s_dt   = tl.full([1], softmax_scale, dtype=DTYPE)      # for tiles (BF16/FP16)
    s_fp32 = tl.full([1], softmax_scale, dtype=tl.float32) # for FP32 accumulators
    Q_block  = tl.load(Q_blk,  boundary_check=(0, 1), padding_option="zero").to(DTYPE) * s_dt
    dO_block = tl.load(dO_blk, boundary_check=(0, 1), padding_option="zero").to(DTYPE)
    dQ_block = tl.zeros((BLOCK_Q, HEAD_DIM), dtype=tl.float32)

    # ---------- loop over KV tiles ----------
    num_steps = tl.cdiv(SEQ_LEN, BLOCK_KV)
    for step in range(num_steps):
        K_T_block = tl.load(K_T_blk, boundary_check=(1,), padding_option="zero").to(DTYPE)
        V_T_block = tl.load(V_T_blk, boundary_check=(1,), padding_option="zero").to(DTYPE)
        
        start_kv = step * BLOCK_KV
        kv_idx   = start_kv + offs_kv
        kv_valid = kv_idx < SEQ_LEN
        S = tl.dot(Q_block, K_T_block, tl.zeros((BLOCK_Q, BLOCK_KV), tl.float32))
        S = tl.where(kv_valid[None, :], S, -float("inf"))
        P = tl.exp(S - m)                  # [BLOCK_Q, BLOCK_KV]

        # dP = dO @ Vᵀ  (match dtypes for dot)
        dP = tl.dot(dO_block, V_T_block, tl.zeros((BLOCK_Q, BLOCK_KV), tl.float32))
        dS = (P * (dP - Di[:, None])).to(DTYPE)
        dQ_block = tl.dot(dS, tl.trans(K_T_block), dQ_block)

        K_T_blk = tl.advance(K_T_blk, (0, BLOCK_KV))
        V_T_blk = tl.advance(V_T_blk, (0, BLOCK_KV))
    
    dQ_block *= s_fp32.to(tl.float32)
    tl.store(dQ_blk, dQ_block.to(dQ.type.element_ty), boundary_check=(0, 1))

class TritonAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V):
        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.size()
        comp_torch = _sdpa_comp_dtype(Q)
        comp_triton = _triton_compute_dtype(comp_torch)
        
        softmax_scale = 1 / (HEAD_DIM**0.5)
        O = torch.empty(Q.shape, dtype=Q.dtype, device=Q.device)
        M = torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN), device=Q.device, dtype=torch.float32
        )  
        ctx.softmax_scale = softmax_scale
        ctx.comp_triton = comp_triton
        ctx.scale = softmax_scale
        
        grid = lambda args: (
            triton.cdiv(SEQ_LEN, args["BLOCK_Q"]),
            BATCH_SIZE * NUM_HEADS,
        )
        _attn_fwd[grid](
            Q, K, V, M, O,
            *Q.stride(), *K.stride(), *V.stride(), *O.stride(),
            NUM_HEADS=Q.shape[1], SEQ_LEN=Q.shape[2], HEAD_DIM=HEAD_DIM, 
            softmax_scale=softmax_scale, DTYPE=comp_triton,
        )
        ctx.save_for_backward(Q, K, V, O, M)
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, _M = ctx.saved_tensors
        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.size()
        
        _D = torch.empty(_M.shape, dtype=_M.dtype, device=_M.device) 
        pre_grid = lambda meta: (triton.cdiv(SEQ_LEN, meta["BLOCK_Q"]),
                         BATCH_SIZE * NUM_HEADS)
        _attn_bwd_preprocess[pre_grid](
            O, dO, _D, *O.stride(), *dO.stride(),
            NUM_HEADS=NUM_HEADS, SEQ_LEN=SEQ_LEN, HEAD_DIM=HEAD_DIM,
        )
        
        dQ = torch.empty(Q.shape, dtype=Q.dtype, device=Q.device) 
        dK = torch.empty(K.shape, dtype=K.dtype, device=K.device)
        dV = torch.empty(V.shape, dtype=V.dtype, device=V.device)

        dkdv_grid = lambda meta: (triton.cdiv(SEQ_LEN, meta["BLOCK_KV"]),
                BATCH_SIZE * NUM_HEADS)
        
        _attn_bwd_dk_dv[dkdv_grid](
            Q, K, V, dO, dK, dV, _M, _D,
            *Q.stride(), *K.stride(), *V.stride(), *dO.stride(), *dK.stride(), *dV.stride(),
            NUM_HEADS=NUM_HEADS, SEQ_LEN=SEQ_LEN, HEAD_DIM=HEAD_DIM, 
            DTYPE=ctx.comp_triton, softmax_scale=ctx.softmax_scale
        )
        
        dq_grid = lambda meta: (triton.cdiv(SEQ_LEN, meta["BLOCK_Q"]),
                    BATCH_SIZE * NUM_HEADS)
        
        _attn_bwd_dq[dq_grid](
            Q, K, V, dO, dQ, _M, _D,
            *Q.stride(), *K.stride(), *V.stride(), *dO.stride(), *dQ.stride(),
            NUM_HEADS=NUM_HEADS, SEQ_LEN=SEQ_LEN, HEAD_DIM=HEAD_DIM, 
            DTYPE=ctx.comp_triton, softmax_scale=ctx.softmax_scale
        )
        
        return dQ, dK, dV

def sdpa_triton_fa(Q: Tensor, K: Tensor, V: Tensor):
    return TritonAttention.apply(Q, K, V)
    
