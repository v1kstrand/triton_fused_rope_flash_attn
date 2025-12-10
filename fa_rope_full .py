import torch
import triton
import triton.language as tl

GROUP_NM_SWEEP = [2, 4, 8]
NUM_STAGES_SWEEP = [2, 3, 4]
NUM_WARPS_SWEEP = [2, 4]

KEY_CACHE = ["BATCH_SIZE", "NUM_HEADS", "SEQ_LEN", "HEAD_DIM"]
def build_axial_rope_pairs(
    side_len: int,
    head_dim: int,
    device: torch.device,
    base: float = 10000.0,
):
    """
    2-D axial RoPE in *pairwise* form for an N x N grid.

    Returns:
        cos_pairs, sin_pairs: [N*N, D2]
            where D2 = head_dim // 2 is the number of complex pairs.
            - pairs 0 .. D2/2 - 1  : x-axis rotations
            - pairs D2/2 .. D2 - 1 : y-axis rotations
    """
    assert head_dim % 4 == 0, "need head_dim divisible by 4 for 2D axial RoPE"
    N = side_len
    D = head_dim

    # number of complex pairs across whole head
    num_pairs_total = D // 2          # D2
    # number of pairs per axis (x or y)
    num_pairs_axis = num_pairs_total // 2   # = D // 4

    # frequency ladder ω_k (1D) for each axis (k = 0..num_pairs_axis-1)
    inv_freq = base ** (-torch.arange(0, num_pairs_axis, device=device) / num_pairs_axis)  # [num_pairs_axis]

    # 1-D positions 0..N-1
    pos = torch.arange(N, device=device)  # [N]
    theta = torch.outer(pos, inv_freq)    # [N, num_pairs_axis]

    cos_1d = torch.cos(theta)             # [N, num_pairs_axis]
    sin_1d = torch.sin(theta)

    # ---- broadcast to 2D grid and flatten (lin = y*N + x) ----
    # x-axis depends on x coordinate
    cos_x_pairs = cos_1d[:, None, :].expand(N, N, num_pairs_axis).reshape(N * N, num_pairs_axis)
    sin_x_pairs = sin_1d[:, None, :].expand(N, N, num_pairs_axis).reshape(N * N, num_pairs_axis)

    # y-axis depends on y coordinate
    cos_y_pairs = cos_1d[None, :, :].expand(N, N, num_pairs_axis).reshape(N * N, num_pairs_axis)
    sin_y_pairs = sin_1d[None, :, :].expand(N, N, num_pairs_axis).reshape(N * N, num_pairs_axis)

    # ---- pack into [N*N, num_pairs_total] = [N*N, D//2] ----
    cos_pairs = torch.empty(N * N, num_pairs_total, device=device, dtype=cos_1d.dtype)
    sin_pairs = torch.empty_like(cos_pairs)

    # first half of pairs: x-axis, second half: y-axis
    cos_pairs[:, :num_pairs_axis] = cos_x_pairs
    sin_pairs[:, :num_pairs_axis] = sin_x_pairs
    cos_pairs[:, num_pairs_axis:] = cos_y_pairs
    sin_pairs[:, num_pairs_axis:] = sin_y_pairs

    return cos_pairs.contiguous(), sin_pairs.contiguous()

def _sdpa_comp_dtype(x: torch.Tensor) -> torch.dtype:
    dtype = torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled() else x.dtype
    if dtype is torch.float16:
        return tl.float16
    if dtype is torch.bfloat16:
        return tl.bfloat16
    if dtype is torch.float32:
        return tl.float32
    raise dtype


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
    Q, K, V, M, O,                 # Q,K,V:[B,H,S,D]; M:[B,H,S] (fp32); O:[B,H,S,D]
    # Strides (inputs)
    sqb, sqh, sqs, sqd,
    skb, skh, sks, skd,
    svb, svh, svs, svd,
    sob, soh, sos, sod,
    # Pairwise RoPE tables [N_pos, D2] where D2 = HEAD_DIM // 2
    COSP, SINP,
    cosp_s, cosp_p,
    sinp_s, sinp_p,
    # Meta
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    BATCH_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    D2: tl.constexpr,               # HEAD_DIM // 2 (num complex pairs)
    HAS_CLS: tl.constexpr,          # 1 if CLS at index 0
    softmax_scale: tl.constexpr,    # 1/sqrt(D)
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    DTYPE: tl.constexpr,            
    GROUP_M: tl.constexpr,
):
    softmax_scale_f = softmax_scale
    neg_large = -1e9
    # ---- swizzled tile ids ----
    pid_m  = tl.program_id(0)
    pid_bh = tl.program_id(1)
    num_tiles_m   = tl.cdiv(SEQ_LEN, BLOCK_Q)
    group_id      = pid_m // GROUP_M
    tiles_in_this = tl.minimum(GROUP_M, num_tiles_m - group_id * GROUP_M)
    m_in_grp      = pid_m - group_id * GROUP_M
    m_in_grp_eff  = m_in_grp % tiles_in_this
    rot           = pid_bh % tiles_in_this
    m_swizzled    = group_id * GROUP_M + ((m_in_grp_eff + rot) % tiles_in_this)
    start_q       = m_swizzled * BLOCK_Q
    if start_q >= SEQ_LEN or m_swizzled >= num_tiles_m:
        return

    # ---- (b,h) plane selection ----
    b = pid_bh // NUM_HEADS
    h = pid_bh %  NUM_HEADS

    off_bh_q = (b * sqb + h * sqh).to(tl.int32)
    off_bh_k = (b * skb + h * skh).to(tl.int32)
    off_bh_v = (b * svb + h * svh).to(tl.int32)
    off_bh_o = (b * sob + h * soh).to(tl.int32)

    # ---- indices/consts ----
    rows    = start_q + tl.arange(0, BLOCK_Q)            # [BQ]
    rows_i  = rows.to(tl.int32)
    q_valid = rows < SEQ_LEN
    cols    = tl.arange(0, BLOCK_KV)                     # [BKV]
    pair_ix = tl.arange(0, D2).to(tl.int32)              # complex pairs
    pair_cosp = pair_ix * cosp_p
    pair_sinp = pair_ix * sinp_p

    # ---- row offsets ----
    row_off_q = rows_i[:, None] * sqs                    # [BQ,1]
    row_off_o = rows_i[:, None] * sos                    # [BQ,1]

    # =========================
    # Q: load even/odd pairs
    # =========================
    base_Q = Q + off_bh_q

    qcol_e = (2 * pair_ix)     * sqd                     # [D2]
    qcol_o = (2 * pair_ix + 1) * sqd                     # [D2]

    # Q_even, Q_odd: [BQ, D2]
    Qe = tl.load(
        base_Q + row_off_q + qcol_e[None, :],
        mask=q_valid[:, None],
        other=0.,
        cache_modifier=".ca",
    ).to(DTYPE)
    Qo = tl.load(
        base_Q + row_off_q + qcol_o[None, :],
        mask=q_valid[:, None],
        other=0.,
        cache_modifier=".ca",
    ).to(DTYPE)

    # ---- Q-side RoPE (pairwise [N_pos, D2]) ----
    lin_q    = rows - HAS_CLS
    lin_q    = tl.maximum(lin_q, 0).to(tl.int32)         # clamp at 0
    is_cls_q = (HAS_CLS != 0) & (rows == 0)

    # pointer offsets into COSP/SINP: [BQ,D2]
    c_row = lin_q[:, None]   * cosp_s
    c_col = pair_cosp[None, :]
    s_row = lin_q[:, None]   * sinp_s
    s_col = pair_sinp[None, :]

    COS_q = tl.load(
        COSP + c_row + c_col,
        mask=q_valid[:, None],
        other=0.,
        cache_modifier=".ca",
    ).to(DTYPE)
    SIN_q = tl.load(
        SINP + s_row + s_col,
        mask=q_valid[:, None],
        other=0.,
        cache_modifier=".ca",
    ).to(DTYPE)

    # rotate Q pairs; CLS bypass -> identity
    Qe_r = tl.where(is_cls_q[:, None], Qe, Qe * COS_q - Qo * SIN_q)
    Qo_r = tl.where(is_cls_q[:, None], Qo, Qo * COS_q + Qe * SIN_q)

    # ---- online softmax accumulators ----
    m_i   = tl.full((BLOCK_Q,), -float("inf"), dtype=tl.float32)
    l_i   = tl.full((BLOCK_Q,),  0.0,          dtype=tl.float32)
    O_blk = tl.zeros((BLOCK_Q, HEAD_DIM),      dtype=DTYPE)

    # =========================
    # KV loop
    # =========================
    for start_kv in range(0, SEQ_LEN, BLOCK_KV):
        kv_cols   = start_kv + cols
        kv_valid  = kv_cols < SEQ_LEN

        # ---- K: load even/odd pairs [D2,BKV] ----
        base_K = K + off_bh_k
        k_even_blk = tl.make_block_ptr(
            base_K,
            (D2, SEQ_LEN),
            (2 * skd, sks),
            (0, start_kv),
            (D2, BLOCK_KV),
            (0, 1),
        )
        k_odd_blk = tl.make_block_ptr(
            base_K + skd,
            (D2, SEQ_LEN),
            (2 * skd, sks),
            (0, start_kv),
            (D2, BLOCK_KV),
            (0, 1),
        )

        Ke = tl.load(
            k_even_blk,
            boundary_check=(0, 1),
            padding_option="zero",
            cache_modifier=".ca",
        ).to(DTYPE)
        Ko = tl.load(
            k_odd_blk,
            boundary_check=(0, 1),
            padding_option="zero",
            cache_modifier=".ca",
        ).to(DTYPE)

        # ---- K-side RoPE; CLS col bypass ----
        lin_k    = kv_cols - HAS_CLS
        lin_k    = tl.maximum(lin_k, 0).to(tl.int32)
        is_cls_k = (HAS_CLS != 0) & (kv_cols == 0)

        ck_row = pair_cosp[:, None]
        ck_col = lin_k[None, :]   * cosp_s
        sk_row = pair_sinp[:, None]
        sk_col = lin_k[None, :]   * sinp_s

        COS_k = tl.load(
            COSP + ck_row + ck_col,
            mask=kv_valid[None, :],
            other=0.,
            cache_modifier=".ca",
        ).to(DTYPE)
        SIN_k = tl.load(
            SINP + sk_row + sk_col,
            mask=kv_valid[None, :],
            other=0.,
            cache_modifier=".ca",
        ).to(DTYPE)

        Ke_r = tl.where(is_cls_k[None, :], Ke, Ke * COS_k - Ko * SIN_k)
        Ko_r = tl.where(is_cls_k[None, :], Ko, Ko * COS_k + Ke * SIN_k)

        # ---- logits tile: 2× pair-dot ----
        S_tile = tl.dot(Qe_r, Ke_r, allow_tf32=True) + tl.dot(Qo_r, Ko_r, allow_tf32=True)   # [BQ,BKV]
        S_tile = S_tile * softmax_scale_f
        S_tile = tl.where(q_valid[:, None] & kv_valid[None, :], S_tile, neg_large)

        # ---- online softmax ----
        m_ij  = tl.maximum(m_i, tl.max(S_tile, axis=1))
        alpha = tl.where(q_valid, tl.exp(m_i - m_ij), 1.0)
        P_blk = tl.where(q_valid[:, None], tl.exp(S_tile - m_ij[:, None]), 0.0)
        l_ij  = tl.sum(P_blk, axis=1)

        # ---- accumulate O ----
        v_blk = tl.make_block_ptr(
            V + off_bh_v,
            (SEQ_LEN, HEAD_DIM),
            (svs, svd),
            (start_kv, 0),
            (BLOCK_KV, HEAD_DIM),
            (1, 0),
        )
        V_blk  = tl.load(
            v_blk,
            boundary_check=(0, 1),
            padding_option="zero",
            cache_modifier=".ca",
        ).to(DTYPE)

        O_blk  = O_blk * alpha[:, None].to(DTYPE)
        O_blk  += tl.dot(P_blk.to(DTYPE), V_blk, allow_tf32=True).to(DTYPE)           # (BQ,BKV) @ (BKV,D) + O_blk

        l_i = tl.where(q_valid, l_i * alpha + l_ij, l_i)
        m_i = tl.where(q_valid, m_ij, m_i)

    # ---- write back M and O ----
    m_ptrs = M + (b * NUM_HEADS + h) * SEQ_LEN + rows
    tl.store(m_ptrs, m_i + tl.log(l_i + 1e-9), mask=q_valid)

    inv_l = (1.0 / l_i).to(DTYPE)
    O_blk = O_blk * inv_l[:, None]
    d_idx = tl.arange(0, HEAD_DIM).to(tl.int32)[None, :]
    O_ptrs = (O + off_bh_o) + row_off_o + d_idx * sod
    tl.store(O_ptrs, O_blk.to(O.type.element_ty), mask=q_valid[:, None])


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
    BATCH_SIZE: tl.constexpr,
):
    pid_q  = tl.program_id(0)                          # Q-tile id
    pid_bh = tl.program_id(1)                          # packed (batch, head)
    start_q = pid_q * BLOCK_Q
    if start_q >= SEQ_LEN:
        return

    b = pid_bh // NUM_HEADS
    h = pid_bh %  NUM_HEADS
    off_bh_O  = (b * sOb  + h * sOh ).to(tl.int32)
    off_bh_dO = (b * sdb  + h * sdh ).to(tl.int32)

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
def _attn_bwd_dk_dv_rope(
    Q, K, V, dO, dK, dV, M, D,
    # Q strides [B,H,S,D]
    sqb, sqh, sqs, sqd,
    # K strides [B,H,S,D]
    skb, skh, sks, skd,
    # V strides [B,H,S,D]
    svb, svh, svs, svd,
    # dO strides [B,H,S,D]
    sob, soh, sos, sod,
    # dK strides [B,H,S,D]
    s_dkb, s_dkh, s_dks, s_dkd,
    # dV strides [B,H,S,D]
    s_dvb, s_dvh, s_dvs, s_dvd,
    # Pairwise RoPE tables [N_pos, D2] + (row, col) strides
    COSP, SINP,
    cosp_s, cosp_p,
    sinp_s, sinp_p,
    # Meta
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BATCH_SIZE: tl.constexpr,
    softmax_scale: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    DTYPE: tl.constexpr,
    GROUP_N: tl.constexpr,
    D2: tl.constexpr,        # HEAD_DIM // 2 (num complex pairs)
    HAS_CLS: tl.constexpr,
):
    tl.static_assert((HEAD_DIM % 4) == 0)
    softmax_scale_f = softmax_scale
    neg_large = -1e9

    # -------------------------
    # 0) program ids / (b,h) plane
    # -------------------------
    pid_kv = tl.program_id(0)
    pid_bh = tl.program_id(1)

    b = pid_bh // NUM_HEADS
    h = pid_bh % NUM_HEADS

    # flatten M, D as [B*H, S]
    off_bh_seq = (pid_bh * SEQ_LEN).to(tl.int32)
    M = M + off_bh_seq
    D = D + off_bh_seq

    # -------------------------
    # 1) swizzled KV tile id (same as non-RoPE kernel)
    # -------------------------
    num_tiles_kv = tl.cdiv(SEQ_LEN, BLOCK_KV)
    group_id     = pid_kv // GROUP_N
    group_start  = group_id * GROUP_N
    if group_start >= num_tiles_kv:
        return

    tiles_in_group = tl.minimum(GROUP_N, num_tiles_kv - group_start)
    kv_in_group    = pid_kv - group_start
    kv_eff         = kv_in_group % tiles_in_group
    rot            = pid_bh % tiles_in_group
    kv_tile_id     = group_start + ((kv_eff + rot) % tiles_in_group)

    start_kv = kv_tile_id * BLOCK_KV
    if start_kv >= SEQ_LEN:
        return

    # -------------------------
    # 2) base offsets & strides
    # -------------------------
    off_bh_q  = (b * sqb   + h * sqh  )
    off_bh_k  = (b * skb   + h * skh  )
    off_bh_v  = (b * svb   + h * svh  )
    off_bh_do = (b * sob   + h * soh  )
    off_bh_dk = (b * s_dkb + h * s_dkh)
    off_bh_dv = (b * s_dvb + h * s_dvh)

    pair_ix = tl.arange(0, D2).to(tl.int32)    # [D2] = complex pairs
    pair_cosp = pair_ix * cosp_p
    pair_sinp = pair_ix * sinp_p
    even    = 2 * pair_ix                      # 0,2,4,...
    odd     = 2 * pair_ix + 1                  # 1,3,5,...

    # -------------------------
    # 3) this KV tile: indices + V block
    # -------------------------
    cols_kv = tl.arange(0, BLOCK_KV)
    k_idx   = start_kv + cols_kv               # [BKV]
    kv_valid = k_idx < SEQ_LEN
    k_idx64  = k_idx.to(tl.int32)

    # V tile [BKV, D]
    base_V = V + off_bh_v
    v_blk = tl.make_block_ptr(
        base_V,
        (SEQ_LEN, HEAD_DIM),
        (svs, svd),
        (start_kv, 0),
        (BLOCK_KV, HEAD_DIM),
        (1, 0),
    )
    V_block = tl.load(
        v_blk,
        boundary_check=(0, 1),
        padding_option="zero",
        cache_modifier=".ca",
    ).to(DTYPE)  # [BKV, D]

    # -------------------------
    # 4) K pairs + K-side RoPE ⇒ K̂ (pairwise)
    # -------------------------
    base_K   = K + off_bh_k

    k_even_blk = tl.make_block_ptr(
        base_K,
        (D2, SEQ_LEN),
        (2 * skd, sks),
        (0, start_kv),
        (D2, BLOCK_KV),
        (0, 1),
    )
    k_odd_blk = tl.make_block_ptr(
        base_K + skd,
        (D2, SEQ_LEN),
        (2 * skd, sks),
        (0, start_kv),
        (D2, BLOCK_KV),
        (0, 1),
    )

    Ke = tl.load(
        k_even_blk,
        boundary_check=(0, 1),
        padding_option="zero",
        cache_modifier=".ca",
    ).to(DTYPE)  # [D2, BKV]
    Ko = tl.load(
        k_odd_blk,
        boundary_check=(0, 1),
        padding_option="zero",
        cache_modifier=".ca",
    ).to(DTYPE)  # [D2, BKV]

    # RoPE positions for keys
    lin_k = k_idx - HAS_CLS
    lin_k = tl.maximum(lin_k, 0).to(tl.int32)
    is_cls_k = (HAS_CLS != 0) & (k_idx == 0)

    pair_cosp_col = pair_cosp[:, None]
    pair_sinp_col = pair_sinp[:, None]
    lin_k_row   = lin_k[None, :]      # [1,BKV]

    ck = pair_cosp_col + lin_k_row * cosp_s
    sk = pair_sinp_col + lin_k_row * sinp_s

    COS_k = tl.load(
        COSP + ck,
        mask=kv_valid[None, :],
        other=0.0,
        cache_modifier=".ca",
    ).to(DTYPE)  # [D2, BKV]
    SIN_k = tl.load(
        SINP + sk,
        mask=kv_valid[None, :],
        other=0.0,
        cache_modifier=".ca",
    ).to(DTYPE)  # [D2, BKV]

    is_cls_k_bc = is_cls_k[None, :]   # [1,BKV]

    Ke_r = tl.where(is_cls_k_bc, Ke, Ke * COS_k - Ko * SIN_k)   # [D2,BKV]
    Ko_r = tl.where(is_cls_k_bc, Ko, Ke * SIN_k + Ko * COS_k)

    # -------------------------
    # 5) accumulators (V and K̂ pairs)
    # -------------------------
    dV_acc      = tl.zeros((BLOCK_KV, HEAD_DIM), dtype=tl.float32)
    dKe_hat_acc = tl.zeros((BLOCK_KV, D2),        dtype=tl.float32)
    dKo_hat_acc = tl.zeros((BLOCK_KV, D2),        dtype=tl.float32)

    # -------------------------
    # 6) loop over Q tiles
    # -------------------------
    num_q_tiles = tl.cdiv(SEQ_LEN, BLOCK_Q)
    base_Q  = Q  + off_bh_q
    base_dO = dO + off_bh_do
    d_idx = tl.arange(0, HEAD_DIM).to(tl.int32)[None, :]  # [1, D]

    for qi in range(0, num_q_tiles):
        start_q = qi * BLOCK_Q
        rows_q  = start_q + tl.arange(0, BLOCK_Q)
        q_valid = rows_q < SEQ_LEN
        rows64  = rows_q.to(tl.int32)

        mask_qk = q_valid[:, None] & kv_valid[None, :]

        # ---- dO tile [BQ, D] ----
        dO_ptrs = base_dO + rows64[:, None] * sos + d_idx * sod
        dO_block = tl.load(dO_ptrs,mask=q_valid[:, None],other=0.0,).to(DTYPE)  # [BQ, D]

        # ---- Q pairs [BQ, D2] ----
        row_off_q = rows64[:, None] * sqs

        qcol_e = even * sqd
        qcol_o = odd  * sqd

        Qe = tl.load(base_Q + row_off_q + qcol_e[None, :],mask=q_valid[:, None],other=0.0,).to(DTYPE)  # [BQ,D2]
        Qo = tl.load(base_Q + row_off_q + qcol_o[None, :],mask=q_valid[:, None],other=0.0,).to(DTYPE)  # [BQ,D2]

        # ---- Q-side RoPE → Q̂ [BQ,D2] ----
        lin_q = rows_q - HAS_CLS
        lin_q = tl.maximum(lin_q, 0).to(tl.int32)
        is_cls_q = (HAS_CLS != 0) & (rows_q == 0)
        is_cls_q_bc = is_cls_q[:, None]

        lin_q_col   = lin_q[:, None]      # [BQ,1]
        c_row = lin_q_col * cosp_s
        c_col = pair_cosp[None, :]
        s_row = lin_q_col * sinp_s
        s_col = pair_sinp[None, :]

        COS_q = tl.load(
            COSP + c_row + c_col,
            mask=q_valid[:, None],
            other=0.0,
            cache_modifier=".ca",
        ).to(DTYPE)  # [BQ,D2]
        SIN_q = tl.load(
            SINP + s_row + s_col,
            mask=q_valid[:, None],
            other=0.0,
            cache_modifier=".ca",
        ).to(DTYPE)

        Qe_r = tl.where(is_cls_q_bc, Qe, Qe * COS_q - Qo * SIN_q)
        Qo_r = tl.where(is_cls_q_bc, Qo, Qo * COS_q + Qe * SIN_q)

        # ---- logits S (with RoPE) [BQ, BKV] ----
        S_blk = tl.dot(Qe_r, Ke_r, allow_tf32=True) + tl.dot(Qo_r, Ko_r, allow_tf32=True)    # [BQ,BKV]
        S_blk = tl.where(mask_qk, S_blk * softmax_scale_f, neg_large)

        # ---- reconstruct P from M (logsumexp) ----
        m_i = tl.load(M + rows64, mask=q_valid, other=0.0).to(tl.float32)  # [BQ]
        Di  = tl.load(D + rows64, mask=q_valid, other=0.0).to(tl.float32)  # [BQ]
        P_blk = tl.exp(S_blk - m_i[:, None])  # [BQ, BKV]

        # ---- dV = Σ_i P_ij dO_i  => dV_acc [BKV,D] ----
        dV_acc += tl.dot(
            tl.trans(P_blk).to(DTYPE),   # [BKV,BQ]
            dO_block.to(DTYPE),          # [BQ,D]
            allow_tf32=True,
        )

        # ---- dS via FlashAttn formula ----
        dp_blk = tl.dot(dO_block, tl.trans(V_block), allow_tf32=True).to(tl.float32)  # [BQ,BKV]
        dS_blk = (P_blk * (dp_blk - Di[:, None])).to(DTYPE)          # [BQ,BKV]

        # ---- dK̂ pairs: dK̂_j = Σ_i dS_ij Q̂_i ----
        dKe_hat_acc += tl.dot(tl.trans(dS_blk), Qe_r, allow_tf32=True)    # [BKV,D2]
        dKo_hat_acc += tl.dot(tl.trans(dS_blk), Qo_r, allow_tf32=True)

    # -------------------------
    # 7) apply softmax_scale to dK̂
    # -------------------------
    dKe_hat_acc *= softmax_scale_f
    dKo_hat_acc *= softmax_scale_f

    # -------------------------
    # 8) un-rotate K̂-grads → base dK pairs
    # -------------------------
    # COS_k, SIN_k are [D2,BKV] => transpose to [BKV,D2]
    COSk_T = tl.trans(COS_k).to(tl.float32)   # [BKV,D2]
    SINk_T = tl.trans(SIN_k).to(tl.float32)   # [BKV,D2]

    is_cls_k_bc = is_cls_k[:, None]           # [BKV,1]

    # [d_e, d_o] = R^T * [dê, dô]
    dKe = tl.where(
        is_cls_k_bc,
        dKe_hat_acc,
        dKe_hat_acc * COSk_T + dKo_hat_acc * SINk_T,
    )
    dKo = tl.where(
        is_cls_k_bc,
        dKo_hat_acc,
        -dKe_hat_acc * SINk_T + dKo_hat_acc * COSk_T,
    )

    # -------------------------
    # 9) scatter pairs into dK[:,D]
    # -------------------------
    base_dK = dK + off_bh_dk
    k_idx64_b = k_idx64[:, None] * s_dks  # [BKV,1]

    # even dims
    col_even = even[None, :] * s_dkd
    ptrs = base_dK + k_idx64_b + col_even
    tl.store(ptrs, dKe.to(dK.type.element_ty), mask=kv_valid[:, None])

    # odd dims
    col_odd = odd[None, :] * s_dkd
    ptrs = base_dK + k_idx64_b + col_odd
    tl.store(ptrs, dKo.to(dK.type.element_ty), mask=kv_valid[:, None])

    # -------------------------
    # 10) write dV block
    # -------------------------
    dV_blk = tl.make_block_ptr(
        dV + off_bh_dv,
        (SEQ_LEN, HEAD_DIM),
        (s_dvs, s_dvd),
        (start_kv, 0),
        (BLOCK_KV, HEAD_DIM),
        (1, 0),
    )
    tl.store(dV_blk, dV_acc.to(dV.type.element_ty), boundary_check=(0, 1))

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
def _attn_bwd_dq_rope(
    Q, K, V, dO, dQ, M, D,          # tensors
    # Q strides [B,H,S,D]
    sqb, sqh, sqs, sqd,
    # K strides [B,H,S,D]
    skb, skh, sks, skd,
    # V strides [B,H,S,D]
    svb, svh, svs, svd,
    # dO strides [B,H,S,D]
    dob, doh, dos, dod,
    # dQ strides [B,H,S,D]
    s_dqb, s_dqh, s_dqs, s_dqd,
    # Pairwise RoPE tables [N_pos, D2] + (row, col) strides
    COSP, SINP,
    cosp_s, cosp_p,
    sinp_s, sinp_p,
    # meta
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    DTYPE: tl.constexpr,
    GROUP_M: tl.constexpr,        
    BATCH_SIZE: tl.constexpr,
    softmax_scale: tl.constexpr,
    D2: tl.constexpr,             # HEAD_DIM // 2 (num complex pairs)
    HAS_CLS: tl.constexpr,
):
    tl.static_assert((HEAD_DIM % 4) == 0)
    softmax_scale_f = softmax_scale
    neg_large = -1e9

    # -------------------------
    # 0) program ids
    # -------------------------
    pid_q  = tl.program_id(0)      # which Q tile
    pid_bh = tl.program_id(1)      # packed (b,h)

    b = pid_bh // NUM_HEADS
    h = pid_bh %  NUM_HEADS
    
    num_tiles_q = tl.cdiv(SEQ_LEN, BLOCK_Q)
    group_id    = pid_q // GROUP_M
    group_start = group_id * GROUP_M
    if group_start >= num_tiles_q:
        return
    
    tiles_in_grp = tl.minimum(GROUP_M, num_tiles_q - group_start)
    q_in_grp     = pid_q - group_start
    q_eff        = q_in_grp % tiles_in_grp
    rot          = pid_bh % tiles_in_grp
    q_tile_id    = group_start + ((q_eff + rot) % tiles_in_grp)

    # one tile in q-dim
    start_q = q_tile_id * BLOCK_Q
    if start_q >= SEQ_LEN:
        return

    # -------------------------
    # 1) base offsets & scalars
    # -------------------------
    off_bh_q  = (b * sqb   + h * sqh  )
    off_bh_k  = (b * skb   + h * skh  )
    off_bh_v  = (b * svb   + h * svh  )
    off_bh_dO = (b * dob   + h * doh  )
    off_bh_dQ = (b * s_dqb + h * s_dqh)

    # treat M, D as [B*H, S] slices
    off_bh_seq = (pid_bh * SEQ_LEN).to(tl.int32)
    M = M + off_bh_seq
    D = D + off_bh_seq

    # -------------------------
    # 2) Q rows, masks, pair indices
    # -------------------------
    rows    = start_q + tl.arange(0, BLOCK_Q)    # [BLOCK_Q]
    rows64  = rows.to(tl.int32)
    q_valid = rows < SEQ_LEN

    pair_ix = tl.arange(0, D2).to(tl.int32)      # [D2] complex pairs
    pair_cosp = pair_ix * cosp_p
    pair_sinp = pair_ix * sinp_p
    even    = 2 * pair_ix                        # 0,2,4,...
    odd     = 2 * pair_ix + 1                    # 1,3,5,...

    d_idx = tl.arange(0, HEAD_DIM).to(tl.int32)[None, :]  # [1, D]

    # -------------------------
    # 3) load dO tile [BQ, D] and rowwise D, M
    # -------------------------
    dO_ptrs  = (dO + off_bh_dO) + rows64[:, None] * dos + d_idx * dod
    dO_block = tl.load(dO_ptrs,mask=q_valid[:, None],other=0.0,).to(DTYPE)  # [BQ, D]

    Di = tl.load(D + rows64, mask=q_valid, other=0.0).to(tl.float32)           # [BQ]
    Mi = tl.load(M + rows64, mask=q_valid, other=neg_large).to(tl.float32) # [BQ]

    # -------------------------
    # 4) gather Q pairs & apply RoPE (pairwise, same as fwd)
    # -------------------------
    base_Q    = Q + off_bh_q
    row_off_q = rows64[:, None] * sqs

    qcol_e = even * sqd
    qcol_o = odd  * sqd

    Qe = tl.load(base_Q + row_off_q + qcol_e[None, :],mask=q_valid[:, None],other=0.0,).to(DTYPE)  # [BQ, D2]
    Qo = tl.load(base_Q + row_off_q + qcol_o[None, :],mask=q_valid[:, None],other=0.0,).to(DTYPE)  # [BQ, D2]

    # Q-side RoPE
    lin_q    = rows - HAS_CLS
    lin_q    = tl.maximum(lin_q, 0).to(tl.int32)
    is_cls_q = (HAS_CLS != 0) & (rows == 0)

    lin_q_col   = lin_q[:, None]          # [BQ,1]
    c_row_q = lin_q_col * cosp_s
    c_col_q = pair_cosp[None, :]
    s_row_q = lin_q_col * sinp_s
    s_col_q = pair_sinp[None, :]

    COS_q = tl.load(
        COSP + c_row_q + c_col_q,
        mask=q_valid[:, None],
        other=0.0,
        cache_modifier=".ca",
    ).to(DTYPE)  # [BQ,D2]
    SIN_q = tl.load(
        SINP + s_row_q + s_col_q,
        mask=q_valid[:, None],
        other=0.0,
        cache_modifier=".ca",
    ).to(DTYPE)  # [BQ,D2]

    is_cls_q_bc = is_cls_q[:, None]

    # rotated Q̂ pairs
    Qe_r = tl.where(is_cls_q_bc, Qe, Qe * COS_q - Qo * SIN_q)
    Qo_r = tl.where(is_cls_q_bc, Qo, Qo * COS_q + Qe * SIN_q)

    # -------------------------
    # 5) accumulators in rotated Q basis
    # -------------------------
    dQe_hat_acc = tl.zeros((BLOCK_Q, D2), dtype=tl.float32)
    dQo_hat_acc = tl.zeros((BLOCK_Q, D2), dtype=tl.float32)

    # -------------------------
    # 6) loop over KV tiles: rebuild S->P, then dS->dQ̂
    # -------------------------
    num_kv_tiles = tl.cdiv(SEQ_LEN, BLOCK_KV)
    cols_local   = tl.arange(0, BLOCK_KV)

    base_K = K + off_bh_k
    base_V = V + off_bh_v

    for kv_t in range(0, num_kv_tiles):
        start_kv = kv_t * BLOCK_KV
        k_idx    = start_kv + cols_local           # [BLOCK_KV]
        kv_valid = k_idx < SEQ_LEN
        k_idx64  = k_idx.to(tl.int32)

        mask_qk = q_valid[:, None] & kv_valid[None, :]

        # V tile [BKV, D] (for dP)
        v_blk = tl.make_block_ptr(
            base_V,
            (SEQ_LEN, HEAD_DIM),
            (svs, svd),
            (start_kv, 0),
            (BLOCK_KV, HEAD_DIM),
            (1, 0),
        )
        V_blk  = tl.load(
            v_blk,
            boundary_check=(0, 1),
            padding_option="zero",
            cache_modifier=".ca",
        ).to(DTYPE)  # [BKV, D]

        # K pairs [D2, BKV]
        k_even_blk = tl.make_block_ptr(
            base_K,
            (D2, SEQ_LEN),
            (2 * skd, sks),
            (0, start_kv),
            (D2, BLOCK_KV),
            (0, 1),
        )
        k_odd_blk = tl.make_block_ptr(
            base_K + skd,
            (D2, SEQ_LEN),
            (2 * skd, sks),
            (0, start_kv),
            (D2, BLOCK_KV),
            (0, 1),
        )

        Ke = tl.load(
            k_even_blk,
            boundary_check=(0, 1),
            padding_option="zero",
            cache_modifier=".ca",
        ).to(DTYPE)  # [D2,BKV]
        Ko = tl.load(
            k_odd_blk,
            boundary_check=(0, 1),
            padding_option="zero",
            cache_modifier=".ca",
        ).to(DTYPE)  # [D2,BKV]

        # K-side RoPE
        lin_k    = k_idx - HAS_CLS
        lin_k    = tl.maximum(lin_k, 0).to(tl.int32)
        is_cls_k = (HAS_CLS != 0) & (k_idx == 0)

        pair_cosp_col = pair_cosp[:, None]
        pair_sinp_col = pair_sinp[:, None]
        lin_k_row   = lin_k[None, :]      # [1,BKV]

        ck = pair_cosp_col + lin_k_row * cosp_s
        sk = pair_sinp_col + lin_k_row * sinp_s

        COS_k = tl.load(
            COSP + ck,
            mask=kv_valid[None, :],
            other=0.0,
            cache_modifier=".ca",
        ).to(DTYPE)  # [D2,BKV]
        SIN_k = tl.load(
            SINP + sk,
            mask=kv_valid[None, :],
            other=0.0,
            cache_modifier=".ca",
        ).to(DTYPE)  # [D2,BKV]

        is_cls_k_bc = is_cls_k[None, :]   # [1,BKV]

        Ke_r = tl.where(is_cls_k_bc, Ke, Ke * COS_k - Ko * SIN_k)
        Ko_r = tl.where(is_cls_k_bc, Ko, Ke * SIN_k + Ko * COS_k)

        # ---- rebuild logits S_tile (same as fwd) ----
        S_tile = (tl.dot(Qe_r, Ke_r, allow_tf32=True) + tl.dot(Qo_r, Ko_r, allow_tf32=True)).to(tl.float32)  # [BQ,BKV]
        S_tile = S_tile * softmax_scale_f
        S_tile = tl.where(mask_qk, S_tile, neg_large)

        # ---- reconstruct P from S and M: P = exp(S - M) ----
        P_blk = tl.exp(S_tile - Mi[:, None])             # [BQ,BKV]
        P_blk = tl.where(mask_qk, P_blk, 0.0).to(DTYPE)

        # ---- dP = dO @ Vᵀ ----
        dP_blk = tl.dot(
            dO_block.to(tl.float32),
            tl.trans(V_blk).to(tl.float32),
            allow_tf32=True,
        )  # [BQ,BKV]

        # ---- dS = P * (dP - D) ----
        dS_blk = (P_blk * (dP_blk - Di[:, None])).to(DTYPE)  # [BQ,BKV]

        # ---- accumulate dQ̂ in pair-space: [BQ,D2] ----
        dQe_hat_acc += tl.dot(dS_blk, tl.trans(Ke_r), allow_tf32=True)
        dQo_hat_acc += tl.dot(dS_blk, tl.trans(Ko_r), allow_tf32=True)

    # -------------------------
    # 7) apply softmax_scale once (chain rule for scaled logits)
    # -------------------------
    dQe_hat_acc *= softmax_scale_f
    dQo_hat_acc *= softmax_scale_f

    # -------------------------
    # 8) un-rotate dQ̂ → dQ (Rᵀ in Q space)
    # -------------------------
    COS_q_f = COS_q.to(tl.float32)
    SIN_q_f = SIN_q.to(tl.float32)
    is_cls_q_bc_f = is_cls_q_bc

    dQe = tl.where(
        is_cls_q_bc_f,
        dQe_hat_acc,
        dQe_hat_acc * COS_q_f + dQo_hat_acc * SIN_q_f,
    )
    dQo = tl.where(
        is_cls_q_bc_f,
        dQo_hat_acc,
        -dQe_hat_acc * SIN_q_f + dQo_hat_acc * COS_q_f,
    )

    # -------------------------
    # 9) scatter even/odd pairs into dQ[:, D]
    # -------------------------
    base_dQ = dQ + off_bh_dQ
    row_ix  = rows64[:, None] * s_dqs  # [BQ,1]

    # even dims
    col_ix = even[None, :] * s_dqd
    ptrs   = base_dQ + row_ix + col_ix
    tl.store(ptrs, dQe.to(dQ.type.element_ty), mask=q_valid[:, None])

    # odd dims
    col_ix = odd[None, :] * s_dqd
    ptrs   = base_dQ + row_ix + col_ix
    tl.store(ptrs, dQo.to(dQ.type.element_ty), mask=q_valid[:, None])

class TritonAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, cos_sin, H_img=14, has_cls=True):
        # ---- Shapes / dtypes ----
        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.size()
        assert (HEAD_DIM % 4) == 0, "HEAD_DIM must be divisible by 4 (2D RoPE requires pairs)."
        N_img = H_img ** 2
        if has_cls:
            assert SEQ_LEN == 1 + N_img, f"SEQ_LEN must be 1 + H_img*W_img when has_cls=True (got {SEQ_LEN} vs {1+N_img})."
        else:
            assert SEQ_LEN == N_img, f"SEQ_LEN must equal H_img*W_img when has_cls=False (got {SEQ_LEN} vs {N_img})."

        comp_triton = _sdpa_comp_dtype(Q)
        softmax_scale = 1.0 / (HEAD_DIM ** 0.5)

        # ---- Outputs ----
        O = torch.empty(Q.shape, dtype=Q.dtype, device=Q.device)
        M = torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN), device=Q.device, dtype=torch.float32
        ) 

        # ---- RoPE tables [N, P] (float32) ----
        COSP, SINP, = cos_sin.tables()

        # ---- Launch ----
        grid = lambda args: (
            triton.cdiv(SEQ_LEN, args["BLOCK_Q"]),
            BATCH_SIZE * NUM_HEADS,
        )
        _attn_fwd[grid](
            Q, K, V, M, O,
            # strides: Q, K, V, O
            *Q.stride(), *K.stride(), *V.stride(), *O.stride(),
            # RoPE tables + their (row, col) strides
            COSP, SINP,
            COSP.stride(0), COSP.stride(1),
            SINP.stride(0), SINP.stride(1),
            # meta
            NUM_HEADS=NUM_HEADS,
            SEQ_LEN=SEQ_LEN,
            BATCH_SIZE=BATCH_SIZE,
            HEAD_DIM=HEAD_DIM,
            D2=HEAD_DIM // 2,
            HAS_CLS=int(has_cls),
            softmax_scale=softmax_scale,
            DTYPE=comp_triton,
        )

        # ---- Save for backward ----
        ctx.softmax_scale = softmax_scale
        ctx.comp_triton = comp_triton
        ctx.has_cls = has_cls
        ctx.save_for_backward(Q, K, V, O, M, COSP, SINP)
        return O
    

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, M, COSP, SINP = ctx.saved_tensors
        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.size()
        
        dQ = torch.empty(Q.shape, dtype=Q.dtype, device=Q.device) 
        dK = torch.empty(K.shape, dtype=K.dtype, device=K.device)
        dV = torch.empty(V.shape, dtype=V.dtype, device=V.device)
        D = torch.empty(M.shape, dtype=M.dtype, device=M.device) 
        pre_grid = lambda meta: (triton.cdiv(SEQ_LEN, meta["BLOCK_Q"]),
                         BATCH_SIZE * NUM_HEADS)
        _attn_bwd_preprocess[pre_grid](
            O, dO, D, *O.stride(), *dO.stride(),
            NUM_HEADS=NUM_HEADS, SEQ_LEN=SEQ_LEN, HEAD_DIM=HEAD_DIM,
            BATCH_SIZE=BATCH_SIZE,
        )
        
        dkdv_grid = lambda meta: (
            triton.cdiv(SEQ_LEN, meta["BLOCK_KV"]),
            BATCH_SIZE * NUM_HEADS,
        )

        _attn_bwd_dk_dv_rope[dkdv_grid](
            Q, K, V, dO, dK, dV, M, D,
            # strides: Q, K, V, dO, dK, dV
            *Q.stride(), *K.stride(), *V.stride(), *dO.stride(),
            *dK.stride(), *dV.stride(),
            # RoPE tables + strides
            COSP, SINP,
            COSP.stride(0), COSP.stride(1),
            SINP.stride(0), SINP.stride(1),
            # meta
            NUM_HEADS=NUM_HEADS,
            SEQ_LEN=SEQ_LEN,
            BATCH_SIZE=BATCH_SIZE,
            softmax_scale=ctx.softmax_scale,
            HEAD_DIM=HEAD_DIM,
            DTYPE=ctx.comp_triton,
            D2=HEAD_DIM // 2,
            HAS_CLS=int(ctx.has_cls),
            # plus your existing ones:
            # BLOCK_Q=..., BLOCK_KV=..., GROUP_N=...
        )
        
        dq_grid = lambda meta: (
            triton.cdiv(SEQ_LEN, meta["BLOCK_Q"]),
            BATCH_SIZE * NUM_HEADS,
        )

        _attn_bwd_dq_rope[dq_grid](
            # tensors
            Q, K, V, dO, dQ, M, D,
            # strides Q, K, V, dO, dQ
            *Q.stride(),
            *K.stride(),
            *V.stride(),
            *dO.stride(),
            *dQ.stride(),
            # RoPE tables + strides
            COSP, SINP,
            COSP.stride(0), COSP.stride(1),
            SINP.stride(0), SINP.stride(1),
            # meta
            NUM_HEADS=NUM_HEADS,
            SEQ_LEN=SEQ_LEN,
            BATCH_SIZE=BATCH_SIZE,
            HEAD_DIM=HEAD_DIM,
            DTYPE=ctx.comp_triton,
            softmax_scale=ctx.softmax_scale,
            D2=HEAD_DIM // 2,
            HAS_CLS=int(ctx.has_cls),
        )
        
        return dQ, dK, dV, None, None, None, None

class CosSinTable(torch.nn.Module):
    """
    Holds pairwise axial RoPE tables for an H_img x H_img grid.

    COSP / SINP are [N_pos, D2], where:
        N_pos = H_img * H_img
        D2    = head_dim // 2 (number of complex pairs)
    """

    def __init__(self, base: float, H_img: int = 14, D: int = 64, device: str = "cuda"):
        super().__init__()
        device = torch.device(device)
        cos_pairs, sin_pairs = build_axial_rope_pairs(
            side_len=H_img,
            head_dim=D,
            device=device,
            base=base,
        )
        self.register_buffer("COSP", cos_pairs)  # [N_pos, D2]
        self.register_buffer("SINP", sin_pairs)  # [N_pos, D2]

    def tables(self):
        """
        Returns (COSP, SINP) in the layout expected by the Triton kernels:
            COSP, SINP: [N_pos, D2]
        """
        return self.COSP, self.SINP
    
    
def sdpa_triton_fa_rope(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, cos_sin: CosSinTable):
    return TritonAttention.apply(Q, K, V, cos_sin)

