#!/usr/bin/env python3
"""CLI for WavLM partial fine-tuning (Gate 0 + nested grouped CV).

Usage:
  PYTHONPATH=. python fine_tuning/train_wavlm_partial.py --stage preflight
  PYTHONPATH=. python fine_tuning/train_wavlm_partial.py --stage all --limit 40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fine_tuning.paths import (  # noqa: E402
    CORPUS_FINAL_AUDIO_DIR,
    SPLIT_MANIFEST_CSV,
    WAVLM_FINETUNED_ROOT,
    WAVLM_MODEL_ID,
)
from fine_tuning.stages import ExperimentConfig, run_experiment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="WavLM partial fine-tuning pipeline")
    parser.add_argument(
        "--stage",
        choices=[
            "all",
            "preflight",
            "layer-cache",
            "frozen-gate",
            "nested-cv",
            "leakage-check",
            "train-full",
        ],
        default="all",
    )
    parser.add_argument("--manifest", type=Path, default=SPLIT_MANIFEST_CSV)
    parser.add_argument("--corpora-root", type=Path, default=CORPUS_FINAL_AUDIO_DIR)
    parser.add_argument("--output-dir", type=Path, default=WAVLM_FINETUNED_ROOT)
    parser.add_argument("--model-id", type=str, default=WAVLM_MODEL_ID)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--include-leaky-calibration",
        action="store_true",
        help="Also run the old contaminated Gate 0 protocol for inflation estimate",
    )
    parser.add_argument(
        "--force-partial",
        action="store_true",
        help=(
            "When running --stage nested-cv alone, ignore Gate 0 prefer_frozen_mix. "
            "Not needed for --stage all (always runs nested partial FT)."
        ),
    )
    args = parser.parse_args()
    cfg = ExperimentConfig(
        manifest_path=args.manifest,
        corpora_root=args.corpora_root,
        output_dir=args.output_dir,
        model_id=args.model_id,
        device=args.device,
        limit=args.limit,
        stage=args.stage,
        outer_folds=args.outer_folds,
        inner_folds=args.inner_folds,
        seed=args.seed,
        repeats=args.repeats,
        include_leaky_calibration=args.include_leaky_calibration,
        ignore_gate0=args.force_partial,
    )
    results = run_experiment(cfg)
    print(f"[done] stages={list(results)}")


if __name__ == "__main__":
    main()
