#!/usr/bin/env bash
set -euo pipefail

ROOTDIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
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
export NM_DECODE_ONLY=1
export NM_WEIGHT_CPU="${NM_WEIGHT_CPU:-0}"
export CTX="${CTX:-512}"
export BENCH_WARMUP="${BENCH_WARMUP:-30}"
export BENCH_REPEAT="${BENCH_REPEAT:-30}"
export BENCH_INNER="${BENCH_INNER:-10}"
export BENCH_CPU_THREADS="${BENCH_CPU_THREADS:-1}"
export MODEL_SOURCE="${MODEL_SOURCE:-auto}"
export HF_MODEL_ID="${HF_MODEL_ID:-Qwen/Qwen3-0.6B}"
export MODEL_DIR="${MODEL_DIR:-$ROOTDIR/models/qwen3-0.6b}"
export HF_ATTN="${HF_ATTN:-sdpa}"

OUTDIR="${OUTDIR:-$ROOTDIR/runs/qwen3_0.6b_triton_ptx_fp16_fixed_$(date -u +%Y%m%d_%H%M%S)_$$}"
mkdir -p "$OUTDIR/logs"
MASTER="$OUTDIR/summary_runner.log"
log() {
  echo "[$(date -u +%F\ %T\ UTC)] $*" | tee -a "$MASTER"
}

read -r -a CONFIGS <<< "${NM_CONFIGS:-0 4 8 16 32 64 128 256}"
log "START model=Qwen3-0.6B source=$MODEL_SOURCE model_dir=$MODEL_DIR"
log "CTX=$CTX warmup=$BENCH_WARMUP repeat=$BENCH_REPEAT inner=$BENCH_INNER"
log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}"

case "$MODEL_SOURCE" in
  local)
    if [[ ! -d "$MODEL_DIR" || ! -f "$MODEL_DIR/config.json" ]]; then
      log "Local model directory is missing or incomplete: $MODEL_DIR"
      exit 2
    fi
    log "Using local model directory: $MODEL_DIR"
    ;;
  auto)
    if [[ -d "$MODEL_DIR" && -f "$MODEL_DIR/config.json" ]]; then
      log "Using local model directory: $MODEL_DIR"
    else
      log "No local model found; falling back to HF model $HF_MODEL_ID"
    fi
    ;;
  hf)
    log "Using Hugging Face model $HF_MODEL_ID"
    ;;
  *)
    log "Unsupported MODEL_SOURCE=$MODEL_SOURCE; expected auto, local, or hf"
    exit 2
    ;;
esac

failures=0
cd "$ROOTDIR"
for M in "${CONFIGS[@]}"; do
  case "$M" in
    0|4|8|16|32|64|128|256) ;;
    *) log "Invalid NM_M=$M"; exit 2 ;;
  esac

  export NM_M="$M"
  if [[ "$M" == 0 ]]; then
    tag=dense
  else
    tag="$((M/2))_$M"
  fi
  log "START $tag"
  if "$PYTHON" -u benchmarks/bench_qwen_3_0_6b_fp16_fixed_ptx.py \
      > "$OUTDIR/logs/$tag.log" 2>&1; then
    rc=0
  else
    rc=$?
  fi
  printf '%s\n' "$rc" > "$OUTDIR/logs/$tag.exitcode"
  result="$(grep '^RESULT' "$OUTDIR/logs/$tag.log" | tail -1 || true)"
  if [[ "$rc" != 0 || -z "$result" ]]; then
    failures=$((failures + 1))
  fi
  log "DONE $tag rc=$rc $result"
done

if [[ "$failures" != 0 ]]; then
  log "FAILED configurations=$failures"
  exit 1
fi
log "ALL_DONE_SUCCESS"
