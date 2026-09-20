#!/usr/bin/env bash
# Fresh output directory per run. Exit nonzero if any configuration/summary fails.
set -euo pipefail
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"
ROOTDIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
PYTHON="${PYTHON:-$(command -v python3 2>/dev/null || command -v python 2>/dev/null || echo python)}"
if [[ -d "$CUDA_HOME/bin" ]]; then
  export PATH="$CUDA_HOME/bin:$PATH"
fi
if [[ -d "$CUDA_HOME/lib" ]]; then
  export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
fi
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOTDIR${PYTHONPATH:+:$PYTHONPATH}"
export MODEL_DTYPE=float16
export NM_KERNEL=triton_ptx
export NM_WEIGHT_CPU="${NM_WEIGHT_CPU:-0}"
export CTX="${CTX:-512}"
export BENCH_WARMUP="${BENCH_WARMUP:-30}"
export BENCH_REPEAT="${BENCH_REPEAT:-30}"
export MODEL_SOURCE="${MODEL_SOURCE:-auto}"
export MODEL_DIR="${MODEL_DIR:-$ROOTDIR/models/llama-3.1-8b}"
export HF_MODEL_ID="${HF_MODEL_ID:-meta-llama/Llama-3.1-8B}"
export CKPT="${CKPT:-$ROOTDIR/checkpoints/llama-3.1-8b/model.pth}"
BENCH_MODE="${BENCH_MODE:-fixed}"
resolve_entry() {
  local candidate="$1"
  if [[ -z "$candidate" ]]; then
    return 1
  fi
  if [[ "$candidate" == /* ]]; then
    printf '%s\n' "$candidate"
    return 0
  fi
  for p in "$ROOTDIR/$candidate" "$ROOTDIR/benchmarks/$candidate" "$WORKDIR/$candidate"; do
    if [[ -f "$p" ]]; then
      printf '%s\n' "$p"
      return 0
    fi
  done
  printf '%s\n' "$candidate"
  return 1
}
case "$BENCH_MODE" in
  fixed) ENTRY="${ENTRY:-bench_llama_3_8b_fp16_fixed_ptx.py}" ;;
  paired) ENTRY="${ENTRY:-bench_llama31_8b_fp16_split_ratios.py}" ;;
  *) echo "BENCH_MODE must be fixed or paired" >&2; exit 2 ;;
esac
ENTRY_PATH="$(resolve_entry "$ENTRY")" || { echo "Benchmark entry not found: $ENTRY" >&2; exit 2; }
OUTDIR="${OUTDIR:-$ROOTDIR/runs/llama31_8b_triton_ptx_fp16_fixed_$(date -u +%Y%m%d_%H%M%S)_$$}"
# Refuse to overwrite an existing run, including the historical failed logs.
mkdir -p "$OUTDIR/logs"
MASTER="$OUTDIR/summary_runner.log"
log(){ echo "[$(date -u +%F\ %T\ UTC)] $*" | tee -a "$MASTER"; }
read -r -a CONFIGS <<< "${NM_CONFIGS:-0 4 8 16 32 64 128 256}"
log "START mode=$BENCH_MODE source=$MODEL_SOURCE model=$HF_MODEL_ID model_dir=$MODEL_DIR CTX=$CTX warmup=$BENCH_WARMUP repeat=$BENCH_REPEAT GPP=autotune CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}"
cd "$ROOTDIR"
failures=0
for M in "${CONFIGS[@]}"; do
  case "$M" in 0|4|8|16|32|64|128|256) ;; *) log "Invalid NM_M=$M"; exit 2;; esac
  export NM_M="$M"
  if [[ "$M" == 0 ]]; then tag=dense; else tag="$((M/2))_$M"; fi
  log "START $tag"
  if "$PYTHON" -u "$ENTRY_PATH" > "$OUTDIR/logs/$tag.log" 2>&1; then rc=0; else rc=$?; fi
  printf '%s\n' "$rc" > "$OUTDIR/logs/$tag.exitcode"
  res="$(grep '^RESULT' "$OUTDIR/logs/$tag.log" | tail -1 || true)"
  if [[ "$rc" != 0 || -z "$res" ]]; then failures=$((failures+1)); fi
  log "DONE $tag rc=$rc $res"
done
if [[ -f "$ROOTDIR/summarize_ptx_bench.py" ]]; then
  if ! "$PYTHON" "$ROOTDIR/summarize_ptx_bench.py" "$OUTDIR"; then
    log "SUMMARY_FAILED"; exit 1
  fi
else
  log "summarize_ptx_bench.py not found; using built-in summary"
fi
if [[ "$failures" != 0 ]]; then log "FAILED configurations=$failures"; exit 1; fi
log "ALL_DONE_SUCCESS"
