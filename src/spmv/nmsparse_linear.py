import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import nmsparse_op  

DECODE_ONLY = os.environ.get("NM_DECODE_ONLY", "0") == "1"
WEIGHT_CPU = os.environ.get("NM_WEIGHT_CPU", "0") == "1" or DECODE_ONLY
_KERNEL = os.environ.get("NM_KERNEL", "")
_FP16_SPARSE = _KERNEL in ("fp16", "fp16_split", "triton_ptx")
_SHARED_MASK = _KERNEL == "triton_ptx"


def _make_sparse(K, N, M, seed, device):
    # Synthetic weights for performance only, scaled to avoid FP16 overflow.
    nnz = M // 2
    nb = K // M
    w = nb * nnz
    g = torch.Generator().manual_seed(seed)
    dtype = torch.float16 if _FP16_SPARSE else torch.float32
    mat_data = (torch.randn(w, N, generator=g, dtype=torch.float32) / (K ** 0.5)).to(dtype)
    if _SHARED_MASK:
        # Triton PTX shared-mask layout: [num_groups, nnz]
        mat_index = torch.empty(nb, nnz, dtype=torch.uint8)
        for bank in range(nb):
            mat_index[bank] = torch.randperm(M, generator=g)[:nnz].sort().values.to(torch.uint8)
    else:
        # iter7 / fp16_split layout: [w, N] per-output-row indices
        mat_index = torch.randint(0, M, (w, N), generator=g, dtype=torch.uint8)
    if device != "cpu":
        mat_data = mat_data.to(device)
        mat_index = mat_index.to(device)
    return mat_data, mat_index, w, nnz


class NmSparseLinear(nn.Module):
    def __init__(self, N, K, M, seed, device, dense_weight=None):
        super().__init__()
        assert K % M == 0 and N % 128 == 0, (K, N, M)
        self._weight_cpu = None
        self.weight = None
        if dense_weight is not None:
            if WEIGHT_CPU:
                self._weight_cpu = dense_weight.detach().cpu()
                del dense_weight
                if device != "cpu":
                    torch.cuda.empty_cache()
            else:
                self.weight = dense_weight
        md, mi, w, nnz = _make_sparse(K, N, M, seed, device)
        self.register_buffer("mat_data", md)
        self.register_buffer("mat_index", mi)
        self.h, self.w, self.block_width, self.vec_width = N, w, nnz, M

    def _prefill_weight(self, device):
        if self.weight is not None:
            return self.weight
        if self._weight_cpu is None:
            raise RuntimeError("prefill requires dense weight")
        return self._weight_cpu.to(device)

    def forward(self, x):
        if x.shape[-2] == 1:
            return torch.ops.nmsparse.gemv(
                x, self.mat_data, self.mat_index,
                self.w, self.h, self.block_width, self.vec_width,
            )
        return F.linear(x, self._prefill_weight(x.device))


# Block layouts this module knows how to swap into NmSparseLinear.
#
# gpt-fast's own Transformer block fuses attention into a single
# `attention.wqkv` / `attention.wo` pair and the MLP into
# `feed_forward.w1/w2/w3`. A Hugging Face `Qwen3DecoderLayer` (and most
# other HF decoder blocks) instead exposes the attention and MLP
# sub-modules under different names, each already split into separate
# `nn.Linear`s rather than fused tensors: `self_attn.{q_proj,k_proj,
# v_proj,o_proj}` and `mlp.{gate_proj,up_proj,down_proj}`. Both layouts
# are supported here so `swap_to_nmsparse` works against either a
# gpt-fast checkpoint or a `transformers` model loaded via
# `AutoModelForCausalLM`.
_ATTENTION_LAYOUTS = (
    ("attention", ("wqkv", "wo")),
    ("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
)

_MLP_LAYOUTS = (
    ("feed_forward", ("w1", "w3", "w2")),
    ("mlp", ("gate_proj", "up_proj", "down_proj")),
)


def _resolve_layout(blk, layouts, kind):
    """Find which known (submodule_attr, linear_names) layout `blk` uses."""
    for submodule_attr, linear_names in layouts:
        submodule = getattr(blk, submodule_attr, None)
        if submodule is not None:
            return submodule, linear_names
    known = ", ".join(f".{attr}" for attr, _ in layouts)
    raise AttributeError(
        f"decoder block {type(blk).__name__} has none of the known "
        f"{kind} sub-modules ({known}); swap_to_nmsparse does not "
        f"recognize this block's layout."
    )


def _swap_linears(submodule, linear_names, M, device, seed, cnt):
    for name in linear_names:
        old = getattr(submodule, name)
        N, K = old.weight.shape
        dense_weight = None if DECODE_ONLY else old.weight
        del old
        torch.cuda.empty_cache()
        setattr(
            submodule,
            name,
            NmSparseLinear(N, K, M, seed + cnt, device, dense_weight=dense_weight),
        )
        cnt += 1
    return cnt


def swap_to_nmsparse(model, M, device, seed=0, block_devices=None):
    """
    `block_devices`, if given, is a sequence with one device (index or
    "cuda:N" string) per decoder block, overriding `device` for that
    block. Used to place NMSparse buffers on the same multi-GPU layer
    split as a `device_map`-loaded dense model, so the two can be
    compared under identical pipeline topology.
    """
    cnt = 0
    for i, blk in enumerate(model.layers):
        blk_device = device
        if block_devices is not None:
            d = block_devices[i]
            blk_device = d if isinstance(d, str) else f"cuda:{d}"

        attn, attn_names = _resolve_layout(blk, _ATTENTION_LAYOUTS, "attention")
        cnt = _swap_linears(attn, attn_names, M, blk_device, seed, cnt)

        mlp, mlp_names = _resolve_layout(blk, _MLP_LAYOUTS, "feed-forward")
        cnt = _swap_linears(mlp, mlp_names, M, blk_device, seed, cnt)
    return cnt