"""Fixed-context CUDA Graph decode benchmark. Synthetic sparse weights; no quality claim.
Includes attention, KV update, output head and sampling, excludes prefill and transfers.
Uses the same attention backend as generate.py; no automatic backend switch.
"""
import os
import sys
import json
import statistics
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = ROOT / "src" / "spmv"
for candidate in (ROOT, SRC, HERE):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
os.environ.setdefault("NM_KERNEL", "triton_ptx")
os.environ["NM_DECODE_ONLY"] = "1"
import generate as G
import nmsparse_op
import nmsparse_linear as sparse

ALLOWED_IMPORT_DIRS = {ROOT.resolve(), SRC.resolve(), HERE.resolve()}


def _validate_benchmark_model(model):
    if not hasattr(model, "setup_caches") or not hasattr(model, "layers"):
        raise RuntimeError(
            "The loaded Llama model does not implement this benchmark's "
            "project Transformer cache API (setup_caches/layers). A raw "
            "Transformers checkpoint cannot run this benchmark yet."
        )
    return model


def load_llama_hf_model(
    *,
    model_dir: Path | None = None,
    dtype: torch.dtype,
    attention_backend: str,
    model_id: str | None = None,
    source: str | None = None,
):
    """Load a Llama model from a local checkpoint directory or from Hugging Face."""
    model_source = (source or os.environ.get("MODEL_SOURCE", "auto")).lower()
    local_dir = model_dir or Path(os.environ.get("MODEL_DIR", str(ROOT / "models" / "llama-3.1-8b")))
    hf_id = model_id or os.environ.get("HF_MODEL_ID", "meta-llama/Llama-3.1-8B")
    local_ok = local_dir.exists() and (local_dir / "config.json").exists()

    if model_source not in {"auto", "local", "hf"}:
        raise ValueError("MODEL_SOURCE must be one of: auto, local, hf")

    if model_source == "local" and not local_ok:
        raise FileNotFoundError(f"Local model directory does not exist: {local_dir}")

    if model_source == "hf" or (model_source == "auto" and not local_ok):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                hf_id,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                attn_implementation=attention_backend,
            ).to("cuda")
        except OSError as exc:
            endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
            raise RuntimeError(
                f"Unable to load Hugging Face model {hf_id!r}. "
                f"The configured endpoint is {endpoint}. Either authenticate/connect "
                "to Hugging Face, or set MODEL_SOURCE=local and provide MODEL_DIR "
                "containing config.json and the model weights."
            ) from exc
        model.eval()
        return _validate_benchmark_model(model)

    model = AutoModelForCausalLM.from_pretrained(
        str(local_dir),
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation=attention_backend,
    ).to("cuda")
    model.eval()
    return _validate_benchmark_model(model)

@torch.inference_mode()
def main():
    # Avoid CPU thread oversubscription during checkpoint conversion and synthesis.
    torch.set_num_threads(int(os.environ.get("BENCH_CPU_THREADS", "1")))
    for module in (G, nmsparse_op, sparse):
        module_dir = Path(module.__file__).resolve().parent
        if module_dir not in ALLOWED_IMPORT_DIRS:
            raise RuntimeError(f"Foreign import: {module.__file__}")
    if nmsparse_op._mode != "triton_ptx":
        raise RuntimeError("This benchmark requires NM_KERNEL=triton_ptx")
    m = int(os.environ.get("NM_M", "0"))
    ctx = int(os.environ.get("CTX", "512"))
    warmup = int(os.environ.get("BENCH_WARMUP", "30"))
    repeat = int(os.environ.get("BENCH_REPEAT", "30"))
    inner = int(os.environ.get("BENCH_INNER", "10"))
    if min(ctx, warmup, repeat, inner) < 1:
        raise ValueError("CTX, warmup, repeat, inner must be positive")
    if m not in (0, 4, 8, 16, 32, 64, 128, 256):
        raise ValueError("Unsupported NM_M")
    dtype_name = os.environ.get("MODEL_DTYPE", "float16")
    if dtype_name not in ("float16", "fp16"):
        raise ValueError("This comparison requires FP16 on both paths")
    ckpt = Path(os.environ.get("CKPT", str(HERE / "checkpoints/llama-3.1-8b/model.pth")))
    model_dir = Path(os.environ.get("MODEL_DIR", str(ROOT / "models" / "llama-3.1-8b")))
    attention_backend = os.environ.get("HF_ATTN", "sdpa")
    torch.manual_seed(1234)
    print(json.dumps({"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "op_file": nmsparse_op.__file__, "actual_mode": "triton_ptx" if m else "dense",
        "dtype": "float16", "M": m, "ctx": ctx, "warmup": warmup, "repeat": repeat,
        "inner": inner, "attention": attention_backend,
        "measurement": "fixed_context_cuda_graph_decode_with_sampling",
        "synthetic_sparse_weights": bool(m), "cache_contents": "synthetic_random"}), flush=True)

    model_source = os.environ.get("MODEL_SOURCE", "auto").lower()
    if model_source == "hf" or (model_source == "auto" and not ckpt.exists()):
        model = load_llama_hf_model(
            model_dir=model_dir,
            dtype=torch.float16,
            attention_backend=attention_backend,
            model_id=os.environ.get("HF_MODEL_ID", "meta-llama/Llama-3.1-8B"),
            source=model_source,
        )
    else:
        model = G._load_model(ckpt, "cuda", torch.float16, False)
    if m:
        count = sparse.swap_to_nmsparse(model, m, "cuda")
        print(f"SWAPPED {count} linears", flush=True)
        seen = set()
        for layer in model.modules():
            if isinstance(layer, sparse.NmSparseLinear):
                key = (layer.h, layer.w, layer.block_width, layer.vec_width)
                if key not in seen:
                    x = torch.randn(1, 1, layer.w * 2, device="cuda", dtype=torch.float16)
                    y = layer(x)
                    if not torch.isfinite(y).all() or not torch.count_nonzero(y):
                        raise RuntimeError(f"Invalid warmup output {key}")
                    seen.add(key)
        print(f"AUTOTUNED shapes={sorted(seen)}", flush=True)
    with torch.device("cuda"):
        model.setup_caches(max_batch_size=1, max_seq_length=ctx + 1)
    # Identical synthetic context protocol for dense and sparse; no prefill in timing.
    for layer in model.layers:
        layer.attention.kv_cache.k_cache.normal_(std=0.02)
        layer.attention.kv_cache.v_cache.normal_(std=0.02)
    token = torch.tensor([[123]], device="cuda", dtype=torch.int64)
    pos = torch.tensor([ctx], device="cuda", dtype=torch.int64)
    block_mask = None if G.NO_FLASH else G.create_block_mask(
        G.causal_mask, 1, 1, model.max_seq_length, model.max_seq_length, device="cuda")
    decode = torch.compile(G.decode_one_token, fullgraph=True, mode="default")
    # Warm up compilation and autotuning BEFORE explicit graph capture.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(max(3, warmup)):
            out = decode(model, token, pos, block_mask, temperature=1.0, top_k=None)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    if not torch.isfinite(out[1]).all():
        raise RuntimeError("Non-finite decode probabilities")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = decode(model, token, pos, block_mask, temperature=1.0, top_k=None)
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeat):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(inner):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / inner)
    if not torch.isfinite(graph_out[1]).all():
        raise RuntimeError("Non-finite probabilities after replay")
    med = statistics.median(samples)
    tag = f"M={m}({m//2}:{m})" if m else "dense_float16"
    print("SAMPLES_DEC_MS=" + ",".join(f"{x:.6f}" for x in samples), flush=True)
    print(f"RESULT CTX={ctx} {tag} dtype=float16 actual_mode={'triton_ptx' if m else 'dense'} "
          f"measurement=fixed_context_cuda_graph warmup={warmup} repeat={repeat} "
          f"decode_ms_per_token={med:.6f} mean={statistics.mean(samples):.6f} "
          f"stdev={statistics.stdev(samples) if repeat > 1 else 0:.6f}", flush=True)

if __name__ == "__main__":
    main()
