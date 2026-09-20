import os
import sys
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

# Allow importing triton_nmsparse_ptx from this package directory.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

_INC = os.path.join(os.environ.get("CONDA_PREFIX", ""), "targets/x86_64-linux/include")
_KERNEL_ROOT = Path(os.environ.get("NM_KERNEL_ROOT", str(_HERE)))
# NM_KERNEL:
#   guarded     -> multi-GPU CUDAGuard kernel (tensor-parallel 70B)
#   fp16_split  -> fp16 no-TILE CUDA kernel with SPLIT
#   triton_ptx  -> Triton inline-PTX shared-mask kernel (fp16/bf16)
#   default     -> original single-GPU iter7 fp32 kernel
_kernel = os.environ.get("NM_KERNEL", "")
if _kernel not in ("", "guarded", "fp16", "fp16_split", "triton_ptx"):
    raise ValueError(f"Unknown NM_KERNEL={_kernel!r}; refusing silent fallback")
_split_env = os.environ.get("NM_SPLIT", "1")
# Triton PTX selects GPP internally for every supported ratio.

_mod = None
_mode = "fp32"
_ptx_forward = None

if _kernel == "triton_ptx":
    from triton_spmv import nmsparse_spmv_forward_triton_ptx as _ptx_forward
    _mode = "triton_ptx"
elif _kernel == "guarded":
    _src = os.environ.get("NM_GUARDED_SRC", str(_KERNEL_ROOT / "kernels" / "iter7_universal_kernel_guarded.cu"))
    _name = "nmsparse_gptfast_guarded_tritonptx_copy"
    _mode = "fp32"
elif _kernel in ("fp16", "fp16_split"):
    _src = os.environ.get("NM_FP16_SPLIT_SRC", str(_KERNEL_ROOT / "kernels" / "nmsparse_fp16_split_sweep.cu"))
    _name = "nmsparse_gptfast_fp16_split_tritonptx_copy"
    _mode = "fp16_split"
else:
    _src = os.environ.get("NM_KERNEL_SRC", str(_KERNEL_ROOT / "kernels" / "iter7_universal_kernel.cu"))
    _name = "nmsparse_gptfast_tritonptx_copy"
    _mode = "fp32"

if _mode != "triton_ptx":
    if not os.path.exists(_src):
        raise FileNotFoundError(
            f"Missing CUDA source for NM_KERNEL={_kernel!r}: {_src}. "
            "Set NM_KERNEL_SRC / NM_GUARDED_SRC / NM_FP16_SPLIT_SRC to the correct file."
        )
    _cflags = ["-O3", "--use_fast_math"]
    if _INC and os.path.isdir(_INC):
        _cflags.append(f"-I{_INC}")
    _mod = load(name=_name, sources=[_src], extra_cuda_cflags=_cflags, verbose=False)


def _resolve_split(vec_width: int, nnz: int) -> int:
    s = _split_env.strip().lower()
    if s == "auto":
        if vec_width <= 16:
            cand = 1
        else:
            cand = 2
    else:
        cand = int(s)
    if cand not in (1, 2, 4, 8):
        cand = 1
    if nnz % cand != 0:
        for c in (8, 4, 2, 1):
            if c <= cand and nnz % c == 0:
                return c
        return 1
    return cand


@torch.library.custom_op("nmsparse::gemv", mutates_args=())
def gemv(x: torch.Tensor, mat_data: torch.Tensor, mat_index: torch.Tensor,
         w: int, h: int, block_width: int, vec_width: int) -> torch.Tensor:
    K = x.shape[-1]
    x2d = x.reshape(-1, K).contiguous()
    if _mode == "triton_ptx":
        if x2d.shape[0] != 1:
            raise RuntimeError(
                f"triton_ptx gemv only supports decode (rows==1), got {tuple(x2d.shape)}"
            )
        # Shared-mask layout: mat_index [num_groups, nnz], mat_data [w,h]
        xh = x2d if x2d.dtype == torch.float16 else x2d.to(torch.float16)
        md = mat_data if mat_data.dtype == torch.float16 else mat_data.to(torch.float16)
        out = _ptx_forward(
            xh, md, mat_index,
            int(w), int(h), int(block_width), 128, int(vec_width),
            1, int(K),
        )  # [1, h], dtype=vec.dtype
        out = out.to(dtype=x.dtype)
        return out.reshape(*x.shape[:-1], int(h))

    if _mode == "fp16_split":
        if x2d.shape[0] != 1:
            raise RuntimeError(
                f"fp16_split gemv only supports decode (rows==1), got {tuple(x2d.shape)}"
            )
        split = _resolve_split(int(vec_width), int(block_width))
        xh = x2d if x2d.dtype == torch.float16 else x2d.to(torch.float16)
        md = mat_data if mat_data.dtype == torch.float16 else mat_data.to(torch.float16)
        out = _mod.forward_split(
            xh, md, mat_index,
            int(w), int(h), int(block_width), 128, int(vec_width), int(split),
        )
        out = out.to(dtype=x.dtype)
        return out.reshape(*x.shape[:-1], int(h))

    out = _mod.forward(
        x2d, mat_data, mat_index,
        int(w), int(h), int(block_width), 128, int(vec_width),
        x2d.shape[0], int(K),
    )
    return out.reshape(*x.shape[:-1], int(h))


@gemv.register_fake
def _(x, mat_data, mat_index, w, h, block_width, vec_width):
    return x.new_empty((*x.shape[:-1], int(h)))
