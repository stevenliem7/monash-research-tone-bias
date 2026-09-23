#!/usr/bin/env python3
"""Kaggle entrypoint for WavLM partial fine-tuning."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

FINE_TUNING_ROOT = Path(__file__).resolve().parent
REPO_ROOT = FINE_TUNING_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fine_tuning.data import remap_audio_paths  # noqa: E402
from fine_tuning.paths import CORPUS_FINAL_AUDIO_DIR, SPLIT_MANIFEST_CSV  # noqa: E402
from fine_tuning.stages import ExperimentConfig, run_experiment  # noqa: E402

DEFAULT_OUTPUT = Path("/kaggle/working/wavlm_partial")


def _prepare_manifest(manifest: Path, audio_dir: Path, output_dir: Path) -> Path:
    df = pd.read_csv(manifest)
    if "is_labelled" in df.columns:
        df = df[df["is_labelled"]].copy()
    remapped = remap_audio_paths(df, audio_dir, strict=True)
    out = output_dir / "manifest_remapped.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    remapped.to_csv(out, index=False)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Kaggle WavLM partial fine-tuning")
    p.add_argument("--manifest", type=Path, default=SPLIT_MANIFEST_CSV)
    p.add_argument("--audio-dir", type=Path, default=CORPUS_FINAL_AUDIO_DIR)
    p.add_argument("--corpora-root", type=Path, default=None, help=argparse.SUPPRESS)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument(
        "--stage",
        default="all",
        choices=[
            "all",
            "preflight",
            "layer-cache",
            "frozen-gate",
            "nested-cv",
            "leakage-check",
            "train-full",
        ],
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--outer-folds", type=int, default=5)
    p.add_argument("--inner-folds", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument(
        "--include-leaky-calibration",
        action="store_true",
        help="Also run the old contaminated Gate 0 protocol for inflation estimate",
    )
    p.add_argument(
        "--force-partial",
        action="store_true",
        help=(
            "When running --stage nested-cv alone, ignore Gate 0 prefer_frozen_mix. "
            "Not needed for --stage all (always runs nested partial FT)."
        ),
    )
    args = p.parse_args()

    audio_dir = args.audio_dir if args.corpora_root is None else args.corpora_root
    manifest = _prepare_manifest(args.manifest, audio_dir, args.output_dir)

    cfg = ExperimentConfig(
        manifest_path=manifest,
        corpora_root=audio_dir,
        output_dir=args.output_dir,
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
    run_experiment(cfg)


if __name__ == "__main__":
    main()
