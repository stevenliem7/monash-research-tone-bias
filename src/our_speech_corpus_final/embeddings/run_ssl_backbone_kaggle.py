#!/usr/bin/env python3
"""
Authors:
    Steven Liem (steven.liem@sydney.edu.au)

Kaggle end-to-end runner: extract HF SSL embeddings → same valence-head CV grid.

Does NOT invent a new evaluation pipeline. After writing a new .npz it calls
evaluate_valence_head_cv.py / train_valence_head_cv.py unchanged so results are
directly comparable to the emotion2vec CV JSON you already have.

Kaggle notebook (GPU):

    !pip -q install transformers soundfile scipy scikit-learn

    # wav2vec2-large-robust: extract + CV
    !python /kaggle/input/.../run_ssl_backbone_kaggle.py \
      --model wav2vec2-large-robust \
      --corpora-root /kaggle/input/.../our_speech_corpus_final \
      --manifest /kaggle/input/.../our_speech_corpus_manifest_split_80_20.csv \
      --output-dir /kaggle/working/ssl_wav2vec2 \
      --run-cv

    # WavLM-large: extract + CV
    !python /kaggle/input/.../run_ssl_backbone_kaggle.py \
      --model wavlm-large \
      --corpora-root /kaggle/input/.../our_speech_corpus_final \
      --manifest /kaggle/input/.../our_speech_corpus_manifest_split_80_20.csv \
      --output-dir /kaggle/working/ssl_wavlm \
      --run-cv

Smoke test:

    !python .../run_ssl_backbone_kaggle.py --model wav2vec2-large-robust \
      --corpora-root ... --manifest ... --output-dir /kaggle/working/smoke \
      --limit 20 --run-cv --folds 2 --repeats 1

Note: use a single backslash for line breaks in notebook cells (not \\).

--corpora-root may be the parent of our_speech_corpus_final/ or the WAV folder
itself (including the double-nested upload layout).

References:
    https://huggingface.co/facebook/wav2vec2-large-robust
    https://huggingface.co/microsoft/wavlm-large
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

for _p in Path(__file__).resolve().parents:
    _loader = _p / "load_config.py"
    if _loader.is_file():
        _spec = importlib.util.spec_from_file_location("load_config", _loader)
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        _mod.bootstrap(__file__)
        break


import argparse
import sys
from pathlib import Path

import pandas as pd

from config import bootstrap_imports, extend_our_speech_corpus_paths

bootstrap_imports(__file__)
extend_our_speech_corpus_paths()

from extract_emotion2vec_embeddings_kaggle import remap_audio_paths
from extract_hf_ssl_embeddings import MODEL_ALIASES, extract_embeddings, resolve_model_id


def _safe_stem(model_id: str) -> str:
    """Turn a HF model id into a filesystem-safe stem.

    Args:
        model_id: Full HF id or alias.

    Returns:
        str: Safe filename stem.
    """
    return resolve_model_id(model_id).replace("/", "_")


def main() -> None:
    """CLI: remap audio → extract SSL embeddings → optional CV evaluation.

    Args:
        None. Command-line flags: ``--model``, ``--corpora-root``, ``--manifest``,
        ``--output-dir``, ``--device``, ``--limit``, ``--splits``, ``--run-cv``,
        ``--folds``, ``--repeats``, ``--seed``, and ``--acoustic-features``.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description="Kaggle: HF SSL extract + valence-head CV"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=f"Alias or HF id. Aliases: {', '.join(MODEL_ALIASES)}",
    )
    parser.add_argument(
        "--corpora-root",
        type=Path,
        required=True,
        help="Parent of our_speech_corpus_final/, or that folder itself",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="our_speech_corpus_manifest_split_80_20.csv (or current split CSV)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/kaggle/working/ssl_backbone"),
        help="Directory for remapped CSV, NPZ, and CV JSON",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--splits",
        nargs="*",
        default=["train", "test"],
        help="Rows to embed (default: train test = all labelled)",
    )
    parser.add_argument(
        "--run-cv",
        action="store_true",
        help="After extraction, run evaluate_valence_head_cv unchanged",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--acoustic-features",
        action="store_true",
        help="Pass through to CV script (trivial acoustics baseline)",
    )
    args = parser.parse_args()

    if not args.corpora_root.is_dir():
        raise FileNotFoundError(f"corpora-root not found: {args.corpora_root}")
    if not args.manifest.is_file():
        raise FileNotFoundError(f"manifest not found: {args.manifest}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Only remap rows we will embed (default: train+test). Checking all 3200
    # rows hard-fails when a few unlabeled WAVs are absent from the upload.
    df = pd.read_csv(args.manifest)
    if args.splits:
        before = len(df)
        df = df[df["split"].isin(args.splits)].copy()
        print(f"[kaggle] filtered splits={args.splits}: {before} -> {len(df)} rows")

    remapped = remap_audio_paths(df, args.corpora_root, strict=True)
    remapped_csv = args.output_dir / "manifest_remapped.csv"
    remapped.to_csv(remapped_csv, index=False)

    stem = _safe_stem(args.model)
    npz_path = args.output_dir / f"our_speech_corpus_{stem}.npz"

    summary = extract_embeddings(
        model_id=args.model,
        manifest_path=remapped_csv,
        output_path=npz_path,
        device=args.device,
        limit=args.limit,
        splits=tuple(args.splits) if args.splits else None,
    )
    print(f"[kaggle] embeddings: {summary}")

    if not args.run_cv:
        print("[kaggle] skipping CV (--run-cv not set)")
        return

    # Import here so extract-only runs don't require the CV module path quirks.
    from train_valence_head_cv import evaluate_all

    cv_out = args.output_dir / f"our_speech_corpus_valence_head_cv_{stem}.json"
    evaluate_all(
        embeddings_path=npz_path,
        manifest_path=remapped_csv,
        output_path=cv_out,
        folds=args.folds,
        repeats=args.repeats,
        seed=args.seed,
        acoustic=args.acoustic_features,
    )
    print(f"[kaggle] cv results: {cv_out}")


if __name__ == "__main__":
    main()
