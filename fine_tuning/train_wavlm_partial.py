#!/usr/bin/env python3
"""CLI for WavLM partial / full fine-tuning (Gate 0 + nested grouped CV).

Usage:
  PYTHONPATH=. python fine_tuning/train_wavlm_partial.py --stage preflight
  PYTHONPATH=. python fine_tuning/train_wavlm_partial.py --stage all --limit 40
  PYTHONPATH=. python fine_tuning/train_wavlm_partial.py --stage full-ft-nested
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fine_tuning.paths import (  # noqa: E402
    CORPUS_FINAL_AUDIO_DIR,
    FULL_FT_BATCH_SIZE,
    FULL_FT_HEAD_PRETRAIN_EPOCHS,
    FULL_FT_MARGIN_DELTA,
    FULL_FT_PRIMARY_POOL,
    FULL_FT_SECONDARY_POOL,
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
            "full-ft-smoke",
            "full-ft-nested",
            "full-ft-leakage",
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
    parser.add_argument(
        "--full-ft-smoke-clips",
        type=int,
        default=FULL_FT_BATCH_SIZE,
        help="Clips for --stage full-ft-smoke (fp32 forward+backward).",
    )
    parser.add_argument(
        "--full-ft-outer-fold",
        type=int,
        default=None,
        help="Run a single outer fold (Slurm array). Overrides SLURM_ARRAY_TASK_ID.",
    )
    parser.add_argument(
        "--partial-oof-csv",
        type=Path,
        default=None,
        help="Partial FT OOF file for gates and comparative gain contrast.",
    )
    parser.add_argument(
        "--full-ft-margin",
        type=float,
        default=FULL_FT_MARGIN_DELTA,
        help="Absolute macro-F1 margin δ for clip-level gates (A) and (B).",
    )
    parser.add_argument(
        "--full-ft-include-secondary",
        action="store_true",
        help="Also run final-layer pool arm (not used for headline).",
    )
    parser.add_argument(
        "--full-ft-batch-size",
        type=int,
        default=FULL_FT_BATCH_SIZE,
        help="Optimizer batch size in clips. Lower if longest-clip smoke OOMs.",
    )
    parser.add_argument(
        "--full-ft-head-pretrain-epochs",
        type=int,
        default=FULL_FT_HEAD_PRETRAIN_EPOCHS,
        help="Fit-row cached-feature head pretraining before matched branches.",
    )
    parser.add_argument(
        "--full-ft-aggregate-only",
        action="store_true",
        help="Read completed per-fold files and write reports; never train.",
    )
    args = parser.parse_args()

    outer_fold = args.full_ft_outer_fold
    if outer_fold is None and "SLURM_ARRAY_TASK_ID" in os.environ:
        outer_fold = int(os.environ["SLURM_ARRAY_TASK_ID"])

    arms = (FULL_FT_PRIMARY_POOL,)
    if args.full_ft_include_secondary:
        arms = (FULL_FT_PRIMARY_POOL, FULL_FT_SECONDARY_POOL)

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
        full_ft_smoke_clips=args.full_ft_smoke_clips,
        full_ft_outer_fold=outer_fold,
        partial_oof_csv=args.partial_oof_csv,
        full_ft_margin=args.full_ft_margin,
        full_ft_arms=arms,
        full_ft_batch_size=args.full_ft_batch_size,
        full_ft_head_pretrain_epochs=args.full_ft_head_pretrain_epochs,
        full_ft_aggregate_only=args.full_ft_aggregate_only,
    )
    results = run_experiment(cfg)
    print(f"[done] stages={list(results)}")


if __name__ == "__main__":
    main()
