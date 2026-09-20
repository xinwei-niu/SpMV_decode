#!/usr/bin/env python3
"""
Fixed-context CUDA Graph decode benchmark for Qwen3-14B from a local
Hugging Face snapshot.

Synthetic sparse weights; no quality claim.
Includes one-token attention/KV-cache update/output logits/sampling in the
measured CUDA-Graph replay. Prefill and host/device transfers are excluded
from the timed region.

This is adapted from the GPT-fast fixed-context benchmark, but it loads the
local Hugging Face safetensors checkpoint directly instead of requiring a
GPT-fast .pth checkpoint or generate.py.

Default model directory:
    <repo>/models/qwen3-14b/Qwen/Qwen3-14B

The sparse path uses the local nmsparse_linear.py implementation:
    sparse.swap_to_nmsparse(model.model, M, "cuda")
    (passed the inner Qwen3Model decoder stack, not the Qwen3ForCausalLM
    wrapper -- swap_to_nmsparse expects an object with `.layers` directly
    on it)

Environment variables:
    MODEL_DIR       local HF model directory
    NM_M            0,4,8,16,32,64,128,256
    CTX             fixed context length, default 512
    MODEL_DTYPE     float16/fp16 only
    HF_ATTN         sdpa (default) or another supported HF attention backend
    BENCH_WARMUP    default 30
    BENCH_REPEAT    default 30
    BENCH_INNER     default 10
    CUTE_*          unused; retained outside this file
    BENCH_CPU_THREADS default 1
    BENCH_COMPILE_DECODE  0 (default) or 1 -- torch.compile the decode
                          step before CUDA Graph capture. Off by default:
                          see the note above the torch.compile call in
                          main() for known transformers/Dynamo
                          incompatibilities on some transformers versions.

The output format intentionally keeps the original RESULT line so the same
summary script can consume this benchmark.
"""

from __future__ import annotations

import gc
import json
import os
import statistics
import sys
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Local repository imports
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = ROOT / "src" / "spmv"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT))

# NMSparse configuration must be set before importing nmsparse_linear.
os.environ.setdefault("NM_KERNEL", "triton_ptx")
os.environ["NM_DECODE_ONLY"] = "1"

from transformers import AutoModelForCausalLM

import nmsparse_linear as sparse


SUPPORTED_M = (0, 4, 8, 16, 32, 64, 128, 256)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_nmsparse_layer_class():
    cls = getattr(sparse, "NmSparseLinear", None)
    return cls


@torch.inference_mode()
def _warmup_nmsparse_shapes(model, warmup=1):
    """Compile/autotune one representative invocation per NMSparse shape."""
    cls = _get_nmsparse_layer_class()
    if cls is None:
        print(
            "WARNING: nmsparse_linear.NmSparseLinear was not found; "
            "skipping shape warmup.",
            flush=True,
        )
        return

    seen = set()
    for _ in range(max(1, warmup)):
        for layer in model.modules():
            if not isinstance(layer, cls):
                continue

            key = (
                int(layer.h),
                int(layer.w),
                int(getattr(layer, "block_width", -1)),
                int(getattr(layer, "vec_width", -1)),
            )

            if key in seen:
                continue

            # Keep the exact warmup convention used by the original NMSparse
            # fixed-context benchmark.
            x = torch.randn(
                1,
                1,
                int(layer.w) * 2,
                device="cuda",
                dtype=torch.float16,
            )
            y = layer(x)

            if not torch.isfinite(y).all() or not torch.count_nonzero(y):
                raise RuntimeError(
                    f"Invalid NMSparse warmup output {key}"
                )

            seen.add(key)

    print(
        f"AUTOTUNED shapes={sorted(seen)}",
        flush=True,
    )


def _make_static_cache(model, ctx: int, dtype: torch.dtype):
    """Construct a StaticCache across Transformers cache API variants."""
    try:
        from transformers.cache_utils import StaticCache
    except ImportError as exc:
        raise RuntimeError(
            "This Transformers version does not provide StaticCache. "
            "A fixed-context CUDA Graph benchmark requires a static cache."
        ) from exc

    max_cache_len = ctx + 1

    try:
        return StaticCache(
            config=model.config,
            max_batch_size=1,
            max_cache_len=max_cache_len,
            device="cuda",
            dtype=dtype,
        )
    except TypeError:
        # Compatibility with versions where max_batch_size is inferred.
        return StaticCache(
            config=model.config,
            max_cache_len=max_cache_len,
            device="cuda",
            dtype=dtype,
        )


@torch.inference_mode()
def _initialize_synthetic_cache(
    cache,
    *,
    ctx: int,
    dtype: torch.dtype,
    device: torch.device,
    model=None,
):
    """
    Initialize the given StaticCache with synthetic random KV contents.

    Supports:
        old Transformers:
            cache.key_cache[i]
            cache.value_cache[i]

        new Transformers:
            cache.layers[i].keys
            cache.layers[i].values

    `cache` must be the actual StaticCache instance created by
    `_make_static_cache` -- it is no longer discovered from `model._cache`
    or `model.cache`, since a plain forward-pass call (as opposed to
    `.generate()`) never populates those attributes on the model. `model`
    is only used, when present, for config lookups in the lazy
    (`early_initialization`) branch below.
    """

    if cache is None:
        raise RuntimeError(
            "No StaticCache was provided to _initialize_synthetic_cache."
        )

    # ------------------------------------------------------------------
    # Legacy Transformers API
    # ------------------------------------------------------------------
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        if len(cache.key_cache) != len(cache.value_cache):
            raise RuntimeError(
                f"Mismatched cache layers: "
                f"{len(cache.key_cache)} vs "
                f"{len(cache.value_cache)}"
            )

        for layer_idx, (k, v) in enumerate(
            zip(cache.key_cache, cache.value_cache)
        ):
            if k is None or v is None:
                raise RuntimeError(
                    f"Uninitialized KV cache at layer {layer_idx}"
                )

            if ctx > k.shape[-2]:
                raise ValueError(
                    f"ctx={ctx} exceeds cache capacity "
                    f"{k.shape[-2]} at layer {layer_idx}"
                )

            k.zero_()
            v.zero_()

            k[..., :ctx, :].normal_(std=0.02)
            v[..., :ctx, :].normal_(std=0.02)

        return cache

    # ------------------------------------------------------------------
    # New Transformers StaticCache API
    # ------------------------------------------------------------------
    layers = getattr(cache, "layers", None)

    if layers is None:
        raise RuntimeError(
            "StaticCache exposes neither key_cache/value_cache nor layers. "
            f"cache type={type(cache)}"
        )

    # StaticCache layers may be lazy.
    if hasattr(cache, "early_initialization"):
        need_init = any(
            getattr(layer, "keys", None) is None
            or getattr(layer, "values", None) is None
            for layer in layers
        )

        if need_init:
            if model is None:
                raise RuntimeError(
                    "StaticCache requires lazy initialization but no "
                    "`model` was provided to read config from."
                )

            config = model.config

            if hasattr(config, "get_text_config"):
                try:
                    config = config.get_text_config(decoder=True)
                except TypeError:
                    config = config

            num_kv_heads = getattr(
                config,
                "num_key_value_heads",
                getattr(config, "num_attention_heads", None),
            )

            num_heads = getattr(
                config,
                "num_attention_heads",
                None,
            )

            head_dim = getattr(
                config,
                "head_dim",
                None,
            )

            if head_dim is None:
                hidden_size = getattr(
                    config,
                    "hidden_size",
                    None,
                )

                if hidden_size is None or num_heads is None:
                    raise RuntimeError(
                        "Could not infer head_dim from model config."
                    )

                head_dim = hidden_size // num_heads

            cache.early_initialization(
                batch_size=1,
                num_heads=int(num_kv_heads),
                head_dim=int(head_dim),
                dtype=dtype,
                device=device,
            )

    # Fill synthetic context.
    for layer_idx, layer in enumerate(cache.layers):
        k = getattr(layer, "keys", None)
        v = getattr(layer, "values", None)

        if k is None or v is None:
            continue

        if ctx > k.shape[-2]:
            raise ValueError(
                f"ctx={ctx} exceeds cache capacity "
                f"{k.shape[-2]} at layer {layer_idx}"
            )

        k.zero_()
        v.zero_()

        k[..., :ctx, :].normal_(std=0.02)
        v[..., :ctx, :].normal_(std=0.02)

    return cache


@torch.inference_mode()
def decode_one_token(
    model,
    token: torch.Tensor,
    position_ids: torch.Tensor,
    cache_position: torch.Tensor,
    attention_mask: torch.Tensor,
    cache,
):
    """Single fixed-position Qwen3 decode + sampling."""
    outputs = model(
        input_ids=token,
        attention_mask=attention_mask,
        position_ids=position_ids,
        cache_position=cache_position,
        past_key_values=cache,
        use_cache=True,
        logits_to_keep=1,
    )

    logits = outputs.logits[:, -1, :]
    probs = torch.softmax(logits.float(), dim=-1).to(logits.dtype)
    next_token = torch.multinomial(probs, num_samples=1)

    return next_token, probs


@torch.inference_mode()
def load_qwen3_14b(
    model_dir: Path,
    *,
    dtype: torch.dtype,
    attention_backend: str,
    model_id: str | None = None,
    source: str | None = None,
):
    """Load a Qwen3 checkpoint from a local directory or Hugging Face.

    Supported environment controls:
        MODEL_SOURCE=local|hf|auto
        MODEL_DIR=/path/to/local/checkpoint
        HF_MODEL_ID=Qwen/Qwen3-14B or another HF repo id

    `auto` prefers a local directory when it exists and contains the expected
    checkpoint files; otherwise it downloads from the configured HF model id.
    """
    model_source = (source or os.environ.get("MODEL_SOURCE", "auto")).lower()
    model_name = model_id or os.environ.get("HF_MODEL_ID", "Qwen/Qwen3-14B")

    local_ok = model_dir.is_dir() and (model_dir / "config.json").exists()
    if model_source == "local" and not local_ok:
        raise FileNotFoundError(
            f"Qwen3 local model directory does not exist or is incomplete: {model_dir}"
        )

    if model_source == "hf" or (model_source == "auto" and not local_ok):
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            attn_implementation=attention_backend,
        ).to("cuda")
        model.eval()
        return model

    required = ("config.json", "tokenizer.json")
    missing = [name for name in required if not (model_dir / name).exists()]
    has_indexed_weights = (model_dir / "model.safetensors.index.json").exists()
    has_single_weights = (model_dir / "model.safetensors").exists()
    has_sharded_weights = any(model_dir.glob("model-*.safetensors"))
    if not (has_indexed_weights or has_single_weights or has_sharded_weights):
        missing.append("model.safetensors, model.safetensors.index.json, or model-*.safetensors")
    if missing:
        raise RuntimeError(
            f"Local Qwen3 directory is missing required files: {missing}"
        )

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation=attention_backend,
    ).to("cuda")
    model.eval()

    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@torch.inference_mode()
def main():
    torch.set_num_threads(
        int(os.environ.get("BENCH_CPU_THREADS", "1"))
    )

    model_dir = Path(
        os.environ.get(
            "MODEL_DIR",
            str(HERE / "models" / "qwen3-14b" / "Qwen" / "Qwen3-14B"),
        )
    ).expanduser().resolve()

    m = int(os.environ.get("NM_M", "0"))
    ctx = int(os.environ.get("CTX", "512"))
    warmup = int(os.environ.get("BENCH_WARMUP", "30"))
    repeat = int(os.environ.get("BENCH_REPEAT", "30"))
    inner = int(os.environ.get("BENCH_INNER", "10"))
    attention_backend = os.environ.get("HF_ATTN", "sdpa")

    dtype_name = os.environ.get("MODEL_DTYPE", "float16").lower()

    if min(ctx, warmup, repeat, inner) < 1:
        raise ValueError(
            "CTX, warmup, repeat, inner must be positive"
        )

    if m not in SUPPORTED_M:
        raise ValueError(
            f"Unsupported NM_M={m}; expected one of {SUPPORTED_M}"
        )

    if dtype_name not in ("float16", "fp16"):
        raise ValueError(
            "This benchmark requires FP16 on both dense and sparse paths"
        )

    dtype = torch.float16

    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    # Check local imports.
    module_dir = Path(sparse.__file__).resolve().parent
    allowed_import_dirs = {ROOT.resolve(), SRC.resolve()}
    if module_dir not in allowed_import_dirs:
        raise RuntimeError(
            f"Foreign nmsparse_linear import: {sparse.__file__}"
        )

    # We keep the original NMSparse kernel selection requirement.
    if os.environ.get("NM_KERNEL", "triton_ptx") != "triton_ptx":
        raise RuntimeError(
            "This benchmark requires NM_KERNEL=triton_ptx"
        )

    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "transformers": __import__("transformers").__version__,
                "model_dir": str(model_dir),
                "actual_mode": "triton_ptx" if m else "dense",
                "dtype": "float16",
                "M": m,
                "ctx": ctx,
                "warmup": warmup,
                "repeat": repeat,
                "inner": inner,
                "attention": attention_backend,
                "measurement": "fixed_context_cuda_graph_decode_with_sampling",
                "synthetic_sparse_weights": bool(m),
                "cache_contents": "synthetic_random",
            }
        ),
        flush=True,
    )

    # -----------------------------------------------------------------------
    # Load Qwen3-14B from the local HF safetensor snapshot.
    # -----------------------------------------------------------------------
    model = load_qwen3_14b(
        model_dir,
        dtype=dtype,
        attention_backend=attention_backend,
        model_id=os.environ.get("HF_MODEL_ID", "Qwen/Qwen3-14B"),
        source=os.environ.get("MODEL_SOURCE", "auto"),
    )

    if m:
        # `swap_to_nmsparse` expects the bare decoder stack (an object with
        # a `.layers` ModuleList directly on it), but `AutoModelForCausalLM`
        # for Qwen3 returns the `Qwen3ForCausalLM` wrapper: the decoder
        # layers live one level down, at `model.model.layers`, with the
        # top-level object instead exposing `.model` (the inner
        # `Qwen3Model`) and a separate `.lm_head`. Hand it the inner
        # decoder module so `for blk in model.layers` resolves correctly.
        decoder = getattr(model, "model", model)
        count = sparse.swap_to_nmsparse(
            decoder,
            m,
            "cuda",
        )
        print(
            f"SWAPPED {count} linears",
            flush=True,
        )

        # Trigger NMSparse compilation/autotuning before graph capture.
        # This scans model.modules() recursively, so it still finds the
        # swapped layers regardless of nesting depth -- pass the full
        # top-level model here, not just the decoder.
        _warmup_nmsparse_shapes(
            model,
            warmup=1,
        )

    # -----------------------------------------------------------------------
    # Static cache and fixed-context protocol.
    # -----------------------------------------------------------------------
    cache = _make_static_cache(
        model,
        ctx,
        dtype,
    )

    cache = _initialize_synthetic_cache(
        cache,
        ctx=ctx,
        dtype=torch.float16,
        device=torch.device("cuda"),
        model=model,
    )

    token = torch.tensor(
        [[123]],
        device="cuda",
        dtype=torch.long,
    )

    position_ids = torch.tensor(
        [[ctx]],
        device="cuda",
        dtype=torch.long,
    )

    cache_position = torch.tensor(
        [ctx],
        device="cuda",
        dtype=torch.long,
    )

    attention_mask = torch.ones(
        (1, ctx + 1),
        device="cuda",
        dtype=torch.long,
    )

    # -----------------------------------------------------------------------
    # Optionally compile before CUDA graph capture.
    #
    # torch.compile is DISABLED by default (BENCH_COMPILE_DECODE=0). Recent
    # `transformers` releases (observed on 5.11.0) wrap `forward` in nested
    # decorator stacks (transformers/utils/generic.py,
    # transformers/utils/output_capturing.py) that are actively fragile
    # under Dynamo tracing:
    #   - with fullgraph=True, Dynamo refuses to trace raw
    #     `func.__code__.co_varnames` introspection in generic.py and
    #     raises `Unsupported`.
    #   - with fullgraph=False (graph breaks allowed), the eager fallback
    #     re-enters output_capturing.py's wrapper and hits
    #     `NameError: name 'torch' is not defined` -- a bug in that
    #     transformers version's own module (it only resolves `torch` in
    #     certain call contexts), not in this script.
    #
    # torch.compile here was only ever meant to fuse kernels before the
    # CUDA Graph capture below; the fixed-context timing this benchmark
    # reports comes from the CUDA Graph replay itself, which works fine
    # against the eager model. So by default we skip compilation entirely
    # and capture eager execution directly. Set BENCH_COMPILE_DECODE=1 to
    # re-enable torch.compile (e.g. once a newer/older `transformers`
    # without this bug is installed), with fullgraph=False since
    # fullgraph=True is known to fail against the generic.py wrapper.
    # -----------------------------------------------------------------------
    if os.environ.get("BENCH_COMPILE_DECODE", "0") == "1":
        decode = torch.compile(
            decode_one_token,
            fullgraph=False,
            mode="default",
        )
    else:
        decode = decode_one_token

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    with torch.cuda.stream(stream):
        for _ in range(max(3, warmup)):
            out = decode(
                model,
                token,
                position_ids,
                cache_position,
                attention_mask,
                cache,
            )

    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    if not torch.isfinite(out[1]).all():
        raise RuntimeError(
            "Non-finite decode probabilities"
        )

    # -----------------------------------------------------------------------
    # CUDA Graph capture.
    # -----------------------------------------------------------------------
    graph = torch.cuda.CUDAGraph()

    with torch.cuda.graph(graph):
        graph_out = decode(
            model,
            token,
            position_ids,
            cache_position,
            attention_mask,
            cache,
        )

    for _ in range(warmup):
        graph.replay()

    torch.cuda.synchronize()

    # -----------------------------------------------------------------------
    # Timed CUDA Graph replay.
    # -----------------------------------------------------------------------
    samples = []

    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(inner):
            graph.replay()
        end.record()
        end.synchronize()

        samples.append(
            start.elapsed_time(end) / inner
        )

    if not torch.isfinite(graph_out[1]).all():
        raise RuntimeError(
            "Non-finite probabilities after replay"
        )

    med = statistics.median(samples)
    mean = statistics.mean(samples)
    stdev = statistics.stdev(samples) if repeat > 1 else 0.0

    tag = (
        f"M={m}({m // 2}:{m})"
        if m
        else "dense_float16"
    )

    actual_mode = "triton_ptx" if m else "dense"

    print(
        "SAMPLES_DEC_MS="
        + ",".join(f"{x:.6f}" for x in samples),
        flush=True,
    )

    print(
        f"RESULT CTX={ctx} {tag} dtype=float16 "
        f"actual_mode={actual_mode} "
        f"measurement=fixed_context_cuda_graph "
        f"warmup={warmup} repeat={repeat} "
        f"decode_ms_per_token={med:.6f} "
        f"mean={mean:.6f} "
        f"stdev={stdev:.6f}",
        flush=True,
    )

    del graph
    del decode
    del cache
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()