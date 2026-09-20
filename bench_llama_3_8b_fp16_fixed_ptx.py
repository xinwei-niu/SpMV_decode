"""Compatibility wrapper to the benchmarks package."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from benchmarks.bench_llama_3_8b_fp16_fixed_ptx import main

if __name__ == "__main__":
    main()

@torch.inference_mode()
def main():
    # Avoid CPU thread oversubscription during checkpoint conversion and synthesis.
    torch.set_num_threads(int(os.environ.get("BENCH_CPU_THREADS", "1")))
    for module in (G, nmsparse_op, sparse):
        if Path(module.__file__).resolve().parent != HERE:
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
    torch.manual_seed(1234)
    print(json.dumps({"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "op_file": nmsparse_op.__file__, "actual_mode": "triton_ptx" if m else "dense",
        "dtype": "float16", "M": m, "ctx": ctx, "warmup": warmup, "repeat": repeat,
        "inner": inner, "attention": "sdpa" if G.NO_FLASH else "flex_attention",
        "measurement": "fixed_context_cuda_graph_decode_with_sampling",
        "synthetic_sparse_weights": bool(m), "cache_contents": "synthetic_random"}), flush=True)
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
