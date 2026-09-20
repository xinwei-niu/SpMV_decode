# SpMV Kernel for Efficient Pruned Model Decoding

Kernel implementations for sparse Generalized Matrix-Vector Multiplication (GEMV) targeting efficient decoding with N:M semi-structured pruned language models.

## TL;DR

This project provides Triton PTX kernels for accelerating N:M sparse
matrix-vector multiplication during LLM decoding. It includes reproducible
benchmarks for Llama 3.1, Qwen3, and Qwen3-0.6B, with local Hugging Face
checkpoint support and `uv`-based setup.

```bash
uv sync
MODEL_SOURCE=local \
MODEL_DIR=/path/to/Qwen3-14B \
NM_CONFIGS=0 \
uv run run_bench_qwen_14b_fp16.sh
```

## Table of Contents

- [Setup with uv](#setup-with-uv)
- [Kernel](#kernel)
	- [A100 / SM80 Optimization](#a100--sm80-optimization)
- [E2E Per-Decoded-Token Latency](#e2e-per-decoded-token-latency)
	- [Llama 3.1 8B](#llama-31-8b)
	- [Qwen3 14B](#qwen3-14b)
	- [Qwen3 32B](#qwen3-32b)
- [Summary](#summary)
- [TODO](#todo)

## Setup with uv

Create the project environment and install the locked dependencies:

```bash
uv sync
```

Run commands inside that environment without activating it:

```bash
uv run python -m py_compile benchmarks/bench_qwen_3_0_6b_fp16_fixed_ptx.py
MODEL_SOURCE=local \
MODEL_DIR=/path/to/Qwen3-0.6B \
NM_CONFIGS=0 \
uv run ./scripts/run_bench_qwen_0_6b_fp16.sh
```

For an activated environment, use `source .venv/bin/activate` after
`uv sync`. The existing `requirements.txt` is retained for legacy pip-based
setups; new environments should use `pyproject.toml` and `uv.lock`.

## Kernel

The kernel stores:

```text
mat_data  : [w, h]      # sparse weights
mat_index : [w/N, N]    # uint8 sparse indices
vec       : [1, w]      # activation vector
```

Each program processes a block of output rows and multiple N groups:

* `program_id(0)` → output rows
* `program_id(1)` → sparse groups

Within the kernel, partial results are accumulated with relaxed atomics. FP32 FMA via [inline PTX](https://triton-lang.org/main/python-api/generated/triton.language.inline_asm_elementwise.html) `fma.rn.f32` is used for accumulation, with the final result converted back to the input dtype.

### A100 / SM80 Optimization

For A100, the kernel uses ([`cache_modifier`](https://github.com/xinwei-niu/SpMV_decode/blob/main/triton_spmv.py#186)) ([reference](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators)) to apply different cache hints:

| Tensor                  | Cache hint | Rationale           |
| :---------------------- | :--------: | :------------------ |
| Activation vector `x_i` |    `.ca`   | Favor L1/L2 reuse   |
| Sparse weights `w_i`    |    `.cg`   | Reduce L1 pollution |

## E2E Per-Decoded-Token Latency

All sparse configurations use **50% sparsity** under the corresponding N:M pattern, measured on a single A100-SXM4-80GB GPU.

### Llama 3.1 8B

|  Pattern  | Sparsity | Decode (ms/token) |    Speedup | Mean (ms/token) |   Std (ms) |
| :-------: | :------: | ----------------: | ---------: | --------------: | ---------: |
|   Dense   |    0%    |           10.3904 |     1.000× |         10.3909 |     0.0030 |
|    2:4    |    50%   |            8.0753 |     1.287× |          8.0716 |     0.0082 |
|    4:8    |    50%   |            8.1507 |     1.275× |          8.1532 |     0.0075 |
|    8:16   |    50%   |            6.9512 |     1.495× |          6.9689 |     0.0946 |
|   16:32   |    50%   |            6.9400 |     1.497× |          6.9401 |     0.0027 |
| **32:64** |  **50%** |        **6.8521** | **1.516×** |      **6.8518** | **0.0024** |
|   64:128  |    50%   |            7.0335 |     1.477× |          7.0369 |     0.0154 |
|  128:256  |    50%   |            7.2585 |     1.431× |          7.2590 |     0.0030 |

### Qwen3 14B

|  Pattern  | Sparsity | Decode (ms/token) |    Speedup | Mean (ms/token) |   Std (ms) |
| :-------: | :------: | ----------------: | ---------: | --------------: | ---------: |
|   Dense   |    0%    |           25.0951 |     1.000× |         25.0944 |     0.0077 |
|    2:4    |    50%   |           20.3801 |     1.231× |         20.3804 |     0.0047 |
|    4:8    |    50%   |           20.4739 |     1.226× |         20.4724 |     0.0063 |
|    8:16   |    50%   |           18.0455 |     1.391× |         18.0453 |     0.0033 |
|   16:32   |    50%   |           17.8283 |     1.408× |         17.8286 |     0.0028 |
| **32:64** |  **50%** |       **17.6093** | **1.425×** |     **17.6094** | **0.0054** |
|   64:128  |    50%   |           17.9491 |     1.398× |         17.9525 |     0.0156 |
|  128:256  |    50%   |           19.1014 |     1.314× |         19.1376 |     0.0778 |

### Qwen3 32B

|  Pattern  | Sparsity | Decode (ms/token) |    Speedup | Mean (ms/token) |   Std (ms) |
| :-------: | :------: | ----------------: | ---------: | --------------: | ---------: |
|   Dense   |    0%    |           52.3758 |     1.000× |         52.3753 |     0.0078 |
|    2:4    |    50%   |           39.8042 |     1.316× |         39.8484 |     0.1119 |
|    4:8    |    50%   |           38.5639 |     1.358× |         38.5913 |     0.0853 |
|    8:16   |    50%   |           33.7534 |     1.552× |         33.7851 |     0.0751 |
| **16:32** |  **50%** |       **33.7054** | **1.554×** |     **33.7087** | **0.0148** |
|   32:64   |    50%   |           33.7408 |     1.552× |         33.7448 |     0.0131 |
|   64:128  |    50%   |           34.4702 |     1.519× |         34.4715 |     0.0063 |
|  128:256  |    50%   |           36.5712 |     1.432× |         36.5772 |     0.0223 |

## Summary

End-to-end results show consistent decoding speedups across Llama 3.1 8B, Qwen3 14B, and Qwen3 32B under 50% N:M sparsity.

| Model        | Best Pattern | Decode (ms/token) |    Speedup |
| :----------- | :----------: | ----------------: | ---------: |
| Llama 3.1 8B |   **32:64**  |        **6.8521** | **1.516×** |
| Qwen3 14B    |   **32:64**  |       **17.6093** | **1.425×** |
| Qwen3 32B    |   **16:32**  |       **33.7054** | **1.554×** |

Performance is not monotonic with N:M group size. Across these models, the strongest results occur in the intermediate range of **8:16–32:64**, while larger grouping factors show a gradual reduction in speedup.



## TODO

* [x] Release the E2E benchmark code.
* [x] Release benchmark scripts for Llama 3.1 8B, Qwen3 14B, and Qwen3 32B.
* [x] Add scripts for reproducing the reported decode-latency results.
* [ ] Add benchmark configuration and command-line examples.
* [x] Add standalone kernel microbenchmark scripts.
* [ ] Add result parsing and table-generation scripts.
