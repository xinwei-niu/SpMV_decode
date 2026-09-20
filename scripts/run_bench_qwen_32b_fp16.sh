set -euo pipefail

export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"

ROOTDIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

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
export HF_MODEL_ID="${HF_MODEL_ID:-Qwen/Qwen3-32B}"

# Local Qwen3-32B checkpoint directory. Override with MODEL_DIR when needed.
export MODEL_DIR="${MODEL_DIR:-$ROOTDIR/models/qwen3-32b}"

# Supported by the Qwen3-32B benchmark. The benchmark itself decides how
# this backend is used; this variable is kept explicit for reproducibility.
export HF_ATTN="${HF_ATTN:-sdpa}"

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

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
  fixed)
    ENTRY="${ENTRY:-bench_qwen_3_32b_fp16_fixed_ptx.py}"
    ;;
  paired)
    ENTRY="${ENTRY:-bench_qwen3_32b_fp16_split_ratios.py}"
    ;;
  *)
    echo "BENCH_MODE must be fixed or paired" >&2
    exit 2
    ;;
esac

ENTRY_PATH="$(resolve_entry "$ENTRY")" || {
  echo "Benchmark entry not found: $ENTRY" >&2
  exit 2
}

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

OUTDIR="${OUTDIR:-$ROOTDIR/runs/qwen3_32b_triton_ptx_fp16_fixed_$(date -u +%Y%m%d_%H%M%S)_$$}"

# Refuse to overwrite an existing run.
mkdir -p "$OUTDIR/logs"

MASTER="$OUTDIR/summary_runner.log"

log() {
  echo "[$(date -u +%F\ %T\ UTC)] $*" | tee -a "$MASTER"
}

# Dense + the supported 50% N:M variants.
read -r -a CONFIGS <<< "${NM_CONFIGS:-0 4 8 16 32 64 128 256}"

log "START mode=$BENCH_MODE"
log "MODEL_DIR=$MODEL_DIR"
log "CTX=$CTX"
log "warmup=$BENCH_WARMUP repeat=$BENCH_REPEAT inner=$BENCH_INNER"
log "HF_ATTN=$HF_ATTN"
log "GPP=autotune"
log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}"

cd "$WORKDIR"

# ---------------------------------------------------------------------------
# Basic configuration checks
# ---------------------------------------------------------------------------

case "$CTX" in
  ''|*[!0-9]*)
    log "Invalid CTX=$CTX"
    exit 2
    ;;
esac

if [[ "$MODEL_SOURCE" == "local" ]]; then
  if [[ ! -d "$MODEL_DIR" ]]; then
    log "MODEL_DIR does not exist: $MODEL_DIR"
    exit 2
  fi
  if [[ ! -f "$MODEL_DIR/config.json" ]]; then
    log "Missing $MODEL_DIR/config.json"
    exit 2
  fi
elif [[ "$MODEL_SOURCE" == "auto" ]]; then
  if [[ -d "$MODEL_DIR" && -f "$MODEL_DIR/config.json" ]]; then
    log "Using local model directory: $MODEL_DIR"
  else
    log "No local model found; falling back to HF model $HF_MODEL_ID"
  fi
elif [[ "$MODEL_SOURCE" == "hf" ]]; then
  log "Using Hugging Face model $HF_MODEL_ID"
else
  log "Unsupported MODEL_SOURCE=$MODEL_SOURCE; expected auto, local, or hf"
  exit 2
fi

failures=0

# ---------------------------------------------------------------------------
# Run each N:M configuration
# ---------------------------------------------------------------------------

for M in "${CONFIGS[@]}"; do
  case "$M" in
    0|4|8|16|32|64|128|256)
      ;;
    *)
      log "Invalid NM_M=$M"
      exit 2
      ;;
  esac

  export NM_M="$M"

  if [[ "$M" == 0 ]]; then
    tag=dense
  else
    tag="$((M/2))_$M"
  fi

  log "START $tag"

  if "$PYTHON" -u "$ENTRY_PATH" \
      > "$OUTDIR/logs/$tag.log" 2>&1; then
    rc=0
  else
    rc=$?
  fi

  printf '%s\n' "$rc" > "$OUTDIR/logs/$tag.exitcode"

  res="$(grep '^RESULT' "$OUTDIR/logs/$tag.log" | tail -1 || true)"

  if [[ "$rc" != 0 || -z "$res" ]]; then
    failures=$((failures+1))
  fi

  log "DONE $tag rc=$rc $res"
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

if [[ -f "$WORKDIR/summarize_ptx_bench.py" ]]; then
  if ! "$PYTHON" "$WORKDIR/summarize_ptx_bench.py" "$OUTDIR"; then
    log "SUMMARY_FAILED"
    exit 1
  fi
else
  log "summarize_ptx_bench.py not found; using built-in summary"

  "$PYTHON" - "$OUTDIR" <<'PY'
import re
import sys
from pathlib import Path

outdir = Path(sys.argv[1])
rows = []

for log_file in sorted((outdir / "logs").glob("*.log")):
    text = log_file.read_text(errors="replace")
    matches = re.findall(
        r"^RESULT\s+CTX=(\d+)\s+(\S+)\s+dtype=(\S+)\s+"
        r"actual_mode=(\S+).*?decode_ms_per_token=([0-9.]+)"
        r"(?:\s+mean=([0-9.]+))?"
        r"(?:\s+stdev=([0-9.]+))?",
        text,
        flags=re.MULTILINE,
    )

    if matches:
        ctx, tag, dtype, mode, median_ms, mean_ms, stdev_ms = matches[-1]
        rows.append(
            {
                "ctx": int(ctx),
                "tag": tag,
                "dtype": dtype,
                "mode": mode,
                "median_ms": float(median_ms),
                "mean_ms": float(mean_ms) if mean_ms else None,
                "stdev_ms": float(stdev_ms) if stdev_ms else None,
            }
        )

print()
print("=" * 84)
print("QWEN3-32B FIXED-CONTEXT CUTE/PTX SUMMARY")
print("=" * 84)
print(
    f"{'ratio':<12}"
    f"{'median_ms':>16}"
    f"{'speedup':>14}"
)

dense = next((r for r in rows if r["tag"] == "dense"), None)

for row in rows:
    speedup = (
        dense["median_ms"] / row["median_ms"]
        if dense
        else 1.0
    )

    print(
        f"{row['tag']:<12}"
        f"{row['median_ms']:>16.6f}"
        f"{speedup:>13.3f}x"
    )
PY
fi

if [[ "$failures" != 0 ]]; then
  log "FAILED configurations=$failures"
  exit 1
fi

log "ALL_DONE_SUCCESS"