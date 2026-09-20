import torch
import triton
import triton.language as tl


# =============================================================================
# Generic and A100 / SM80-specialized shared-mask N:M GEMV
# =============================================================================
#
# Representation:
#   mat_data  : [w, h], contiguous, i.e. storage offset = col * h + row
#   mat_index : [num_groups, N], uint8
#   vec       : [1, w]
#
#
# A100-specific choices:
#   * SM80 runtime dispatch
#   * FP32 FMA through inline PTX
#   * inline PTX cvt for fp16/bf16 -> fp32 upcast (explicit rounding mode)
#   * .cg for streamed sparse weights (avoid L1 pollution)
#   * .ca for the tiny/reused activation vector (keep in L1)
#
# =============================================================================

@triton.jit
def ptx_fma_f32(x, y, acc):
    return tl.inline_asm_elementwise(
        asm="""
        fma.rn.f32 $0, $1, $2, $3;
        """,
        constraints="=f,f,f,f",
        args=[x, y, acc],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def ptx_cvt_f16_to_f32(x):
    # cvt.rn.f32.f16: round-to-nearest-even upconvert
    return tl.inline_asm_elementwise(
        asm="""
        cvt.rn.f32.f16 $0, $1;
        """,
        constraints="=f,h",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def ptx_cvt_bf16_to_f32(x):
    return tl.inline_asm_elementwise(
        asm="""
        cvt.rn.f32.bf16 $0, $1;
        """,
        constraints="=f,h",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def to_f32(x):
    if x.dtype == tl.float16:
        return ptx_cvt_f16_to_f32(x)
    elif x.dtype == tl.bfloat16:
        return ptx_cvt_bf16_to_f32(x)
    else:
        return x.to(tl.float32)


# -----------------------------------------------------------------------------
# Generic path: conservative configs suitable for other NVIDIA GPUs.
# -----------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_ROWS": 32, "GROUPS_PER_PROGRAM": 1}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_ROWS": 64, "GROUPS_PER_PROGRAM": 1}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_ROWS": 128, "GROUPS_PER_PROGRAM": 1}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_ROWS": 256, "GROUPS_PER_PROGRAM": 1}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_ROWS": 64, "GROUPS_PER_PROGRAM": 2}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_ROWS": 128, "GROUPS_PER_PROGRAM": 2}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_ROWS": 256, "GROUPS_PER_PROGRAM": 2}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_ROWS": 64, "GROUPS_PER_PROGRAM": 4}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_ROWS": 128, "GROUPS_PER_PROGRAM": 4}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_ROWS": 256, "GROUPS_PER_PROGRAM": 4}, num_warps=4, num_stages=2),
    ],
    key=["h", "num_groups", "NNZ", "M"],
    reset_to_zero=["out_ptr"],
)
@triton.jit
def nmsparse_generic_kernel(
    vec_ptr,
    mat_data_ptr,
    mat_index_ptr,
    out_ptr,
    h,
    num_groups,
    vec_stride,
    mat_stride_col,
    mat_stride_row,
    NNZ: tl.constexpr,
    M: tl.constexpr,
    GROUPS_PER_PROGRAM: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_group = tl.program_id(1)

    row = pid_row * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row_mask = row < h
    first_group = pid_group * GROUPS_PER_PROGRAM

    acc = tl.zeros((BLOCK_ROWS,), dtype=tl.float32)

    for group_offset in tl.range(0, GROUPS_PER_PROGRAM):
        group = first_group + group_offset
        group_mask = group < num_groups
        base = group * NNZ
        vec_base = group * M

        for i in tl.static_range(0, NNZ):
            idx_i = tl.load(
                mat_index_ptr + base + i,
                mask=group_mask,
                other=0,
            ).to(tl.int32)

            x_i = to_f32(tl.load(
                vec_ptr + (vec_base + idx_i) * vec_stride,
                mask=group_mask,
                other=0.0,
            ))

            w_i = to_f32(tl.load(
                mat_data_ptr
                + row * mat_stride_row
                + (base + i) * mat_stride_col,
                mask=row_mask & group_mask,
                other=0.0,
            ))

            acc = ptx_fma_f32(w_i, x_i, acc)

    tl.atomic_add(
        out_ptr + row,
        acc,
        mask=row_mask,
        sem="relaxed",
    )


# -----------------------------------------------------------------------------
# A100 / SM80 path.
# -----------------------------------------------------------------------------
#
# Why retain atomics instead of collapsing groups into one program?
# Qwen decode has output sizes of only ~1K-3K. A no-atomic 1-D grid would create
# very few CTAs, while an SM80 A100 has many SMs. Splitting groups keeps enough
# CTAs to occupy the GPU. Larger GPP reduces atomic traffic without collapsing
# parallelism too aggressively.
# -----------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_ROWS": 32, "GROUPS_PER_PROGRAM": 1}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_ROWS": 64, "GROUPS_PER_PROGRAM": 1}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_ROWS": 128, "GROUPS_PER_PROGRAM": 1}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_ROWS": 256, "GROUPS_PER_PROGRAM": 1}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_ROWS": 64, "GROUPS_PER_PROGRAM": 2}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_ROWS": 128, "GROUPS_PER_PROGRAM": 2}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_ROWS": 256, "GROUPS_PER_PROGRAM": 2}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_ROWS": 64, "GROUPS_PER_PROGRAM": 4}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_ROWS": 128, "GROUPS_PER_PROGRAM": 4}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_ROWS": 256, "GROUPS_PER_PROGRAM": 4}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_ROWS": 64,  "GROUPS_PER_PROGRAM": 2},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_ROWS": 128, "GROUPS_PER_PROGRAM": 2},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_ROWS": 256, "GROUPS_PER_PROGRAM": 2},  num_warps=8, num_stages=2),
        triton.Config({"BLOCK_ROWS": 64,  "GROUPS_PER_PROGRAM": 4},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_ROWS": 128, "GROUPS_PER_PROGRAM": 4},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_ROWS": 256, "GROUPS_PER_PROGRAM": 4},  num_warps=8, num_stages=2),
        triton.Config({"BLOCK_ROWS": 64,  "GROUPS_PER_PROGRAM": 8},  num_warps=4, num_stages=3),
        triton.Config({"BLOCK_ROWS": 128, "GROUPS_PER_PROGRAM": 8},  num_warps=4, num_stages=3),
        triton.Config({"BLOCK_ROWS": 256, "GROUPS_PER_PROGRAM": 8},  num_warps=8, num_stages=3),
        triton.Config({"BLOCK_ROWS": 128, "GROUPS_PER_PROGRAM": 16}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_ROWS": 256, "GROUPS_PER_PROGRAM": 16}, num_warps=8, num_stages=3),
    ],
    key=["h", "num_groups", "NNZ", "M"],
    reset_to_zero=["out_ptr"],
)
@triton.jit
def nmsparse_a100_sm80_kernel(
    vec_ptr,
    mat_data_ptr,
    mat_index_ptr,
    out_ptr,
    h,
    num_groups,
    vec_stride,
    mat_stride_col,
    mat_stride_row,
    NNZ: tl.constexpr,
    M: tl.constexpr,
    GROUPS_PER_PROGRAM: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_group = tl.program_id(1)

    row_start = pid_row * BLOCK_ROWS
    # All A100 configs use BLOCK_ROWS >= 64; 128/256 are the intended fast
    # cases. The hint is valid because h is required to be divisible by 128.
    row = row_start + tl.arange(0, BLOCK_ROWS)
    row_mask = row < h

    first_group = pid_group * GROUPS_PER_PROGRAM
    acc = tl.zeros((BLOCK_ROWS,), dtype=tl.float32)

    for group_offset in tl.range(0, GROUPS_PER_PROGRAM):
        group = first_group + group_offset
        group_mask = group < num_groups
        base = group * NNZ
        vec_base = group * M

        for i in tl.static_range(0, NNZ):
            idx_i = tl.load(
                mat_index_ptr + base + i,
                mask=group_mask,
                other=0,
                cache_modifier=".ca",
            ).to(tl.int32)

						# As activations are often reused, hence with .ca cache modifier for loading within the L1 cache.
            x_i = to_f32(tl.load(
                vec_ptr + (vec_base + idx_i) * vec_stride,
                mask=group_mask,
                other=0.0,
                cache_modifier=".ca",
            ))

					  # Weights are loaded with .cg modifier to avoid loading within L1 cache (L2 and higher).
            w_i = to_f32(tl.load(
                mat_data_ptr
                + row * mat_stride_row
                + (base + i) * mat_stride_col,
                mask=row_mask & group_mask,
                other=0.0,
                cache_modifier=".cg",
            ))

            acc = ptx_fma_f32(w_i, x_i, acc)

    tl.atomic_add(
        out_ptr + row,
        acc,
        mask=row_mask,
        sem="relaxed",
    )


# =============================================================================
# Dispatcher / public API
# =============================================================================


def _is_a100_sm80(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    major, minor = torch.cuda.get_device_capability(device)
    return (major, minor) == (8, 0)


def nmsparse_spmv_forward_triton_ptx(
    vec: torch.Tensor,
    mat_data: torch.Tensor,
    mat_index: torch.Tensor,
    w: int,
    h: int,
    BLOCK_WIDTH: int,
    NUM_THREADS: int,
    VEC_WIDTH: int,
    minibatch: int,
    vecNum: int,
    groups_per_program: int = 1,
    arch: str = "auto",
):
    """Shared-mask N:M sparse GEMV with optional A100/SM80 specialization."""

    if not vec.is_cuda or not mat_data.is_cuda or not mat_index.is_cuda:
        raise ValueError("vec, mat_data, and mat_index must be CUDA tensors")

    if vec.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"unsupported vec dtype: {vec.dtype}")
    if mat_data.dtype != vec.dtype:
        raise TypeError("vec and mat_data must have same dtype")
    if mat_index.dtype != torch.uint8:
        raise TypeError("mat_index must be uint8")

    if vec.ndim != 2 or vec.shape[0] != minibatch:
        raise ValueError(f"vec must have shape [{minibatch}, {vecNum}]")
    if minibatch != 1:
        raise NotImplementedError("this specialization requires minibatch=1")
    if vec.shape[1] != vecNum:
        raise ValueError(f"vec.shape[1]={vec.shape[1]} != vecNum={vecNum}")
    if NUM_THREADS != 128:
        raise NotImplementedError("this kernel family assumes 128-thread base execution")
    if h % 128 != 0:
        raise ValueError("h must be divisible by 128")
    if w % BLOCK_WIDTH != 0:
        raise ValueError(f"w={w} must be divisible by N={BLOCK_WIDTH}")

    num_groups = w // BLOCK_WIDTH
    groups_per_program = max(1, min(int(groups_per_program), num_groups))

    if tuple(mat_data.shape) != (w, h):
        raise ValueError(f"mat_data must have shape {(w, h)}, got {tuple(mat_data.shape)}")
    if tuple(mat_index.shape) != (num_groups, BLOCK_WIDTH):
        raise ValueError(
            f"mat_index must have shape {(num_groups, BLOCK_WIDTH)}, "
            f"got {tuple(mat_index.shape)}"
        )
    if not mat_data.is_contiguous():
        raise ValueError("mat_data must be contiguous")
    if not mat_index.is_contiguous():
        raise ValueError("mat_index must be contiguous")
    if not vec.is_contiguous():
        raise ValueError("vec must be contiguous")

    out_fp32 = torch.zeros((1, h), dtype=torch.float32, device=vec.device)

    if arch == "auto":
        use_a100 = _is_a100_sm80(vec.device)
    elif arch.lower() in ("a100", "sm80"):
        use_a100 = True
    elif arch.lower() in ("generic", "default"):
        use_a100 = False
    else:
        raise ValueError(f"unknown arch={arch!r}; use auto, a100/sm80, or generic")

    kernel = nmsparse_a100_sm80_kernel if use_a100 else nmsparse_generic_kernel

    def grid(META):
        return (
            triton.cdiv(h, META["BLOCK_ROWS"]),
            triton.cdiv(num_groups, META["GROUPS_PER_PROGRAM"]),
        )

    kernel[grid](
        vec,
        mat_data,
        mat_index,
        out_fp32,
        h,
        num_groups,
        vec.stride(1),
        mat_data.stride(0),
        mat_data.stride(1),
        NNZ=BLOCK_WIDTH,
        M=VEC_WIDTH,
    )

    return out_fp32.to(vec.dtype)


__all__ = [
    "ptx_fma_f32",
    "ptx_cvt_f16_to_f32",
    "ptx_cvt_bf16_to_f32",
    "to_f32",
    "nmsparse_generic_kernel",
    "nmsparse_a100_sm80_kernel",
    "nmsparse_spmv_forward_triton_ptx",
]
