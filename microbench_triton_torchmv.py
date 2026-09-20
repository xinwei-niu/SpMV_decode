from __future__ import annotations

import csv
import os
from pathlib import Path

import torch
import triton
import triton.testing

from triton_spmv import (
    nmsparse_a100_sm80_kernel,
)


# =============================================================================
# Configuration
# =============================================================================

ROOT = Path(__file__).resolve().parent

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

DTYPE_NAME = os.environ.get("DTYPE", "bf16").lower()

if DTYPE_NAME == "fp16":
    DTYPE = torch.float16
elif DTYPE_NAME == "bf16":
    DTYPE = torch.bfloat16
else:
    raise ValueError("DTYPE must be fp16 or bf16")

GPU = int(os.environ.get("GPU", "0"))
DEVICE = torch.device(f"cuda:{GPU}")

# do_bench uses milliseconds.
WARMUP_MS = float(os.environ.get("WARMUP_MS", "1000"))
REP_MS = float(os.environ.get("REP_MS", "3000"))

SEED = int(os.environ.get("SEED", "0"))

NUM_THREADS = 128
MINIBATCH = 1
BLOCK_ROWS = 128

# This benchmark targets nmsparse_ptx_kernel, whose GROUPS_PER_PROGRAM
# is selected internally by @triton.autotune.
#
# nmsparse_ptx_kernel is the path used by the current wrapper for N < 32.
RATIOS = [
    (2, 4),
    (4, 8),
    (8, 16),
    (16, 32),
    (32, 64),
    (64, 128),
    (128, 256),
]

DIMENSIONS = [
    (4096, 4096),
    (14336, 4096),
    (4096, 14336),
    (8192, 8192),
    (28672, 8192),
    (8192, 28672),
]

OUT_CSV = RESULTS / f"ptx_autotune_{DTYPE_NAME}.csv"
OUT_TXT = RESULTS / f"ptx_autotune_{DTYPE_NAME}.txt"


# =============================================================================
# Sparse case
# =============================================================================

def make_sparse_case(
    n_out: int,
    n_in: int,
    nnz: int,
    m: int,
):
    if m != 2 * nnz:
        raise ValueError(
            f"Expected 50% sparsity, got {nnz}:{m}"
        )

    if n_in % m != 0:
        raise ValueError(
            f"n_in={n_in} must be divisible by M={m}"
        )

    if n_out % BLOCK_ROWS != 0:
        raise ValueError(
            f"n_out={n_out} must be divisible by {BLOCK_ROWS}"
        )

    num_groups = n_in // m
    w = num_groups * nnz
    h = n_out

    # -------------------------------------------------------------------------
    # GEMV vector.
    # -------------------------------------------------------------------------
    vec = torch.randn(
        n_in,
        device=DEVICE,
        dtype=DTYPE,
    ).contiguous()

    # -------------------------------------------------------------------------
    # Sparse values: [w, h], contiguous, stride=(h, 1).
    # -------------------------------------------------------------------------
    mat_data = torch.randn(
        w,
        h,
        device=DEVICE,
        dtype=DTYPE,
    ).contiguous()

    # -------------------------------------------------------------------------
    # Compact shared metadata: [num_groups, nnz].
    # -------------------------------------------------------------------------
    mat_index = torch.empty(
        num_groups,
        nnz,
        device=DEVICE,
        dtype=torch.uint8,
    ).contiguous()

    # -------------------------------------------------------------------------
    # Dense logical sparse matrix, used only for correctness.
    # -------------------------------------------------------------------------
    weight_sparse = torch.zeros(
        h,
        n_in,
        device=DEVICE,
        dtype=DTYPE,
    ).contiguous()

    # -------------------------------------------------------------------------
    # Generate shared N:M pattern.
    # -------------------------------------------------------------------------
    for group in range(num_groups):

        local = (
            torch.randperm(
                m,
                device=DEVICE,
            )[:nnz]
            .sort()
            .values
        )

        mat_index[group] = local.to(torch.uint8)

        global_col = group * m + local

        base = group * nnz

        weight_sparse[:, global_col] = (
            mat_data[base:base + nnz, :].transpose(0, 1)
        )

    # -------------------------------------------------------------------------
    # Sanity checks.
    # -------------------------------------------------------------------------
    assert mat_data.shape == (w, h)
    assert mat_data.stride() == (h, 1)

    assert mat_index.shape == (num_groups, nnz)
    assert mat_index.stride() == (nnz, 1)

    return (
        vec,
        weight_sparse,
        mat_data,
        mat_index,
        w,
        h,
        num_groups,
    )


# =============================================================================
# Dense GEMV
# =============================================================================

def make_dense_case(
    n_out: int,
    n_in: int,
):
    x = torch.randn(
        n_in,
        device=DEVICE,
        dtype=DTYPE,
    ).contiguous()

    W = torch.randn(
        n_out,
        n_in,
        device=DEVICE,
        dtype=DTYPE,
    ).contiguous()

    return x, W


def run_dense(x, W):
    return torch.mv(W, x)


# =============================================================================
# Timing helper
# =============================================================================

def bench(fn):
    # do_bench uses milliseconds.
    ms = triton.testing.do_bench(
        fn,
        warmup=WARMUP_MS,
        rep=REP_MS,
        return_mode="median",
    )
    return float(ms) * 1000.0


# =============================================================================
# Direct autotuned PTX kernel
# =============================================================================

def launch_ptx(
    vec,
    mat_data,
    mat_index,
    h,
    num_groups,
    nnz,
    m,
    out_fp32,
):
    # The kernel chooses GROUPS_PER_PROGRAM from its autotune configs.
    grid = lambda META: (
        triton.cdiv(h, META["BLOCK_ROWS"]),
        triton.cdiv(
            num_groups,
            META["GROUPS_PER_PROGRAM"],
        ),
    )

    nmsparse_a100_sm80_kernel[grid](
        vec,
        mat_data,
        mat_index,
        out_fp32,

        h,
        num_groups,

        vec.stride(0),

        mat_data.stride(0),
        mat_data.stride(1),

        NNZ=nnz,
        M=m,
    )


def get_best_gpp():
    """
    Best-effort extraction of the GPP selected by Triton's Autotuner.

    Returns None if the installed Triton version does not expose best_config
    in the expected form.
    """
    best = getattr(nmsparse_a100_sm80_kernel, "best_config", None)

    if best is None:
        return None

    try:
        kwargs = best.kwargs
        return kwargs.get("GROUPS_PER_PROGRAM")
    except AttributeError:
        pass

    # Some Triton versions may expose a mapping.
    if isinstance(best, dict):
        try:
            last = next(reversed(best.values()))
            return last.kwargs.get("GROUPS_PER_PROGRAM")
        except Exception:
            return None

    return None


def run_ptx_once(
    vec,
    mat_data,
    mat_index,
    h,
    num_groups,
    nnz,
    m,
    out_fp32,
):
    # atomic_add requires a clean output.
    out_fp32.zero_()

    launch_ptx(
        vec,
        mat_data,
        mat_index,
        h,
        num_groups,
        nnz,
        m,
        out_fp32,
    )

    return out_fp32


# =============================================================================
# Timing
# =============================================================================

def bench_kernel(
    vec,
    mat_data,
    mat_index,
    h,
    num_groups,
    nnz,
    m,
    out_fp32,
):
    def fn():
        out_fp32.zero_()

        launch_ptx(
            vec,
            mat_data,
            mat_index,
            h,
            num_groups,
            nnz,
            m,
            out_fp32,
        )

        return out_fp32

    ms = triton.testing.do_bench(
        fn,
        warmup=WARMUP_MS,
        rep=REP_MS,
        return_mode="median",
    )

    return float(ms) * 1000.0


# =============================================================================
# Correctness
# =============================================================================

def check_correctness(
    vec,
    weight_sparse,
    mat_data,
    mat_index,
    h,
    num_groups,
    nnz,
    m,
    out_fp32,
):
    # High-precision reference.
    ref = torch.mv(
        weight_sparse.float(),
        vec.float(),
    )

    run_ptx_once(
        vec,
        mat_data,
        mat_index,
        h,
        num_groups,
        nnz,
        m,
        out_fp32,
    )

    out = out_fp32.float()

    torch.cuda.synchronize()

    if ref.shape != out.shape:
        raise RuntimeError(
            f"shape mismatch: ref={ref.shape}, out={out.shape}"
        )

    diff = (ref - out).abs()

    max_abs = diff.max().item()
    mean_abs = diff.mean().item()

    rmse = torch.sqrt(
        torch.mean(diff * diff)
    ).item()

    ref_norm = torch.linalg.vector_norm(ref)
    diff_norm = torch.linalg.vector_norm(diff)

    rel_l2 = (
        diff_norm
        / ref_norm.clamp_min(1e-12)
    ).item()

    return (
        max_abs,
        mean_abs,
        rmse,
        rel_l2,
    )


# =============================================================================
# Main
# =============================================================================

def main():

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    torch.cuda.set_device(GPU)

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    print()
    print("=" * 110)
    print("Autotuned N:M Inline-PTX GEMV Benchmark")
    print("=" * 110)

    print(f"GPU       : {torch.cuda.get_device_name(GPU)}")
    print(f"dtype     : {DTYPE_NAME}")
    print(f"warmup    : {WARMUP_MS} ms")
    print(f"rep       : {REP_MS} ms")
    print(f"ratios    : {RATIOS}")
    print(f"seed      : {SEED}")

    print("=" * 110)

    results = []

    for n_out, n_in in DIMENSIONS:

        dim = f"({n_out},{n_in})"

        print()
        print(f"--- {dim} ---")

        # ---------------------------------------------------------------------
        # Dense baseline.
        # ---------------------------------------------------------------------
        dense_x, dense_W = make_dense_case(
            n_out,
            n_in,
        )

        dense_us = bench(
            lambda: run_dense(
                dense_x,
                dense_W,
            )
        )

        print(
            f"Dense GEMV: {dense_us:10.2f} us"
        )

        # ---------------------------------------------------------------------
        # Sparse ratios.
        # ---------------------------------------------------------------------
        for nnz, m in RATIOS:

            ratio = f"{nnz}:{m}"

            (
                vec,
                weight_sparse,
                mat_data,
                mat_index,
                w,
                h,
                num_groups,
            ) = make_sparse_case(
                n_out,
                n_in,
                nnz,
                m,
            )

            print()
            print(
                f"  {ratio} (groups={num_groups})"
            )

            # -----------------------------------------------------------------
            # Output buffer.
            # -----------------------------------------------------------------
            out_fp32 = torch.empty(
                (h,),
                dtype=torch.float32,
                device=DEVICE,
            )

            # -----------------------------------------------------------------
            # First launch triggers Triton autotuning.
            # -----------------------------------------------------------------
            run_ptx_once(
                vec,
                mat_data,
                mat_index,
                h,
                num_groups,
                nnz,
                m,
                out_fp32=out_fp32,
            )

            torch.cuda.synchronize()

            best_gpp = get_best_gpp()

            # -----------------------------------------------------------------
            # Correctness after autotuning.
            # -----------------------------------------------------------------
            (
                max_abs,
                mean_abs,
                rmse,
                rel_l2,
            ) = check_correctness(
                vec,
                weight_sparse,
                mat_data,
                mat_index,
                h,
                num_groups,
                nnz,
                m,
                out_fp32=out_fp32,
            )

            # -----------------------------------------------------------------
            # Steady-state timing.
            # -----------------------------------------------------------------
            ptx_us = bench_kernel(
                vec,
                mat_data,
                mat_index,
                h,
                num_groups,
                nnz,
                m,
                out_fp32=out_fp32,
            )

            speedup = dense_us / ptx_us

            gpp_str = (
                str(best_gpp)
                if best_gpp is not None
                else "?"
            )

            print(
                f"    AUTO(GPP={gpp_str:>2}): "
                f"{ptx_us:9.2f} us  "
                f"{speedup:5.2f}x  "
                f"max={max_abs:.3e}  "
                f"rmse={rmse:.3e}  "
                f"rel_l2={rel_l2:.3e}"
            )

            results.append(
                {
                    "dim": dim,
                    "n_out": n_out,
                    "n_in": n_in,
                    "ratio": ratio,
                    "N": nnz,
                    "M": m,
                    "groups": num_groups,
                    "selected_gpp": best_gpp,
                    "dtype": DTYPE_NAME,
                    "dense_us": dense_us,
                    "ptx_us": ptx_us,
                    "speedup_vs_dense": speedup,
                    "max_abs": max_abs,
                    "mean_abs": mean_abs,
                    "rmse": rmse,
                    "rel_l2": rel_l2,
                }
            )

    # =========================================================================
    # Save CSV
    # =========================================================================

    with OUT_CSV.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=results[0].keys(),
        )

        writer.writeheader()
        writer.writerows(results)

    # =========================================================================
    # Save TXT
    # =========================================================================

    with OUT_TXT.open(
        "w",
    ) as f:

        f.write(
            "Autotuned N:M Inline-PTX GEMV benchmark\n"
        )
        f.write(
            f"GPU={torch.cuda.get_device_name(GPU)}\n"
        )
        f.write(
            f"dtype={DTYPE_NAME}\n"
        )
        f.write(
            f"warmup_ms={WARMUP_MS}\n"
        )
        f.write(
            f"rep_ms={REP_MS}\n"
        )
        f.write(
            f"ratios={RATIOS}\n\n"
        )

        f.write(
            "=== RESULTS ===\n\n"
        )

        for row in results:
            f.write(
                f"{row['dim']} {row['ratio']}: "
                f"GPP={row['selected_gpp']} "
                f"PTX={row['ptx_us']:.2f} us "
                f"speedup={row['speedup_vs_dense']:.2f}x "
                f"max={row['max_abs']:.3e} "
                f"rmse={row['rmse']:.3e} "
                f"rel_l2={row['rel_l2']:.3e}\n"
            )

    print()
    print("=" * 110)
    print("Saved:")
    print(f"  {OUT_CSV}")
    print(f"  {OUT_TXT}")
    print("=" * 110)


if __name__ == "__main__":
    main()