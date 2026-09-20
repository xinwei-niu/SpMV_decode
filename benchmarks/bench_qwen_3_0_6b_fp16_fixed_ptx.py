#!/usr/bin/env python3
"""Qwen3-0.6B fixed-context benchmark entrypoint.

This reuses the validated Qwen3 Hugging Face benchmark implementation while
providing 0.6B-specific defaults. Override MODEL_DIR, MODEL_SOURCE, or
HF_MODEL_ID for a different local checkpoint or Hub repository.
"""

import os

os.environ.setdefault("MODEL_SOURCE", "auto")
os.environ.setdefault("MODEL_DIR", "models/qwen3-0.6b")
os.environ.setdefault("HF_MODEL_ID", "Qwen/Qwen3-0.6B")

from bench_qwen_31_14b_fp16_fixed_ptx import main


if __name__ == "__main__":
    main()
