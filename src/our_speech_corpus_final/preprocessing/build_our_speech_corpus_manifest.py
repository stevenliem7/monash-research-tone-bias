#!/usr/bin/env python3
"""
Authors:
    Steven Liem (steven.liem@sydney.edu.au)

Build a unified our_speech_corpus manifest and an 80/20 stratified train/test split. Remember to set up the config file with the correct paths and seed.
The full pipeline is as follows:
  1. Load heet_dataset_clean.csv (Corpus A) and heet_dataset_new_negative_clean.csv (Corpus B)
  2. Derive clip_id as A_<stem> / B_<stem> from the source WAV basename
  3. Point audio_path at corpora_cleaned/our_speech_corpus_final/{clip_id}.wav
  4. Derive is_labelled and duration_s
  5. Write our_speech_corpus_unified.csv
  6. Stratify labelled rows by ground_truth_label x source_batch into train/test (80/20)
  7. Mark unlabelled rows as unlabeled_pool and write our_speech_corpus_manifest_split_80_20.csv

Requires aggregate_our_speech_corpus_final.py to have been run first.

ground_truth_label is the only supervised target. valence_label / emotion_label are prompt-side metadata and must not be used as audio ground truth.

Usage:
    uv run python src/our_speech_corpus_final/build_our_speech_corpus_manifest.py
    uv run python src/our_speech_corpus_final/build_our_speech_corpus_manifest.py --seed 42 --train-fraction 0.8

References:
    None
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
import random
import sys
import wave
from pathlib import Path

import pandas as pd

from config import bootstrap_imports, extend_our_speech_corpus_paths

bootstrap_imports(__file__)
extend_our_speech_corpus_paths()
from config import (
    CORPUS_A_AUDIO_DIR,
    CORPUS_B_AUDIO_DIR,
    CORPUS_FINAL_AUDIO_DIR,
    HEET_CLEAN_CSV,
    HEET_NEW_NEGATIVE_CSV,
    MANIFEST_COLUMNS,
    SOURCE_BATCH_A,
    SOURCE_BATCH_B,
    SPLIT_COLUMN,
    SPLIT_MANIFEST_CSV,
    SPLIT_SEED,
    SPLIT_VALUES,
    TRAIN_FRACTION,
    UNIFIED_MANIFEST_CSV,
    VALENCE_CLASSES,
)


def _normalise_gt(value: object) -> str:
    """Normalise a ground_truth_label cell to lowercase valence or empty.

    Args:
        value: Raw cell value from CSV (may be NaN / blank / mixed case).

    Returns:
        str: One of positive/neutral/negative, or "" if missing/invalid.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "na"}:
        return ""
    return text.lower()


def _wav_duration_seconds(path: Path) -> float | None:
    """Read WAV duration in seconds from file headers.

    Args:
        path: Path to a WAV file.

    Returns:
        float | None: Duration in seconds, or None if the file cannot be read.
    """
    try:
        with wave.open(str(path), "rb") as wf:
            rate = wf.getframerate()
            if rate <= 0:
                return None
            return wf.getnframes() / float(rate)
    except Exception:
        return None


def _source_basename(raw_path: str, row_index: int) -> str:
    """Extract the original WAV basename from a source CSV audio_path cell.

    Args:
        raw_path: Path string from the source CSV.
        row_index: Fallback index when the path is blank.

    Returns:
        str: Basename such as ``1_happy_positive.wav``.
    """
    if not raw_path or str(raw_path).lower() == "nan":
        return f"row{row_index}.wav"
    return Path(str(raw_path)).name


def _make_clip_id(source_batch: str, basename: str) -> str:
    """Build a globally unique clip_id from batch prefix and original WAV stem.

    Args:
        source_batch: SOURCE_BATCH_A or SOURCE_BATCH_B.
        basename: Original WAV basename (unprefixed), e.g. ``1_happy_positive.wav``.

    Returns:
        str: Clip ID of the form ``A_<stem>`` or ``B_<stem>``.
    """
    stem = Path(basename).stem
    prefix = "A" if source_batch == SOURCE_BATCH_A else "B"
    # Avoid double-prefixing if a path already used the aggregated name.
    if stem.startswith(f"{prefix}_"):
        return stem
    return f"{prefix}_{stem}"


def _resolve_wav_for_duration(clip_id: str, basename: str, source_batch: str) -> Path | None:
    """Find a readable WAV for duration: prefer aggregated final, else legacy dirs.

    Args:
        clip_id: Aggregated clip id (``A_<stem>`` / ``B_<stem>``).
        basename: Original unprefixed WAV basename.
        source_batch: SOURCE_BATCH_A or SOURCE_BATCH_B.

    Returns:
        Path | None: Existing WAV path, or None if not found.
    """
    final_path = CORPUS_FINAL_AUDIO_DIR / f"{clip_id}.wav"
    if final_path.is_file():
        return final_path.resolve()

    legacy_dir = CORPUS_A_AUDIO_DIR if source_batch == SOURCE_BATCH_A else CORPUS_B_AUDIO_DIR
    legacy = legacy_dir / basename
    if legacy.is_file():
        return legacy.resolve()
    return None


def _load_source_csv(path: Path, source_batch: str) -> pd.DataFrame:
    """Load one source CSV into the unified manifest schema.

    ``audio_path`` is always written as
    ``corpora_cleaned/our_speech_corpus_final/{clip_id}.wav``.

    Args:
        path: Path to heet_dataset_clean.csv or heet_dataset_new_negative_clean.csv.
        source_batch: Provenance tag written into source_batch.

    Returns:
        pd.DataFrame: Rows with MANIFEST_COLUMNS populated.

    Raises:
        ValueError: If the CSV has neither ``question`` nor ``Question``.
    """
    df = pd.read_csv(path)
    df = df.rename(columns={"Question": "question"})
    if "question" not in df.columns:
        raise ValueError(f"{path} missing question/Question column")

    rows: list[dict] = []
    for i, row in df.iterrows():
        raw_audio = str(row.get("audio_path", "")).strip()
        basename = _source_basename(raw_audio, int(i))
        clip_id = _make_clip_id(source_batch, basename)
        final_audio = (CORPUS_FINAL_AUDIO_DIR / f"{clip_id}.wav").resolve()
        readable = _resolve_wav_for_duration(clip_id, basename, source_batch)

        gt = _normalise_gt(row.get("ground_truth_label"))
        if gt and gt not in VALENCE_CLASSES:
            gt = ""

        rows.append(
            {
                "clip_id": clip_id,
                "source_batch": source_batch,
                "question": str(row.get("question", "")).strip(),
                "emotion_label": str(row.get("emotion_label", "")).strip().lower(),
                "valence_label": str(row.get("valence_label", "")).strip().lower(),
                "audio_path": str(final_audio),
                "response_text": str(row.get("response_text", "")).strip(),
                "ground_truth_label": gt,
                "is_labelled": bool(gt),
                "duration_s": _wav_duration_seconds(readable) if readable else None,
            }
        )

    return pd.DataFrame(rows)


def _stratified_train_test_indices(
    df: pd.DataFrame,
    label_col: str,
    group_cols: list[str],
    test_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Assign train/test indices with stratification on label × group columns.

    Within each stratum, at least one row stays in train when n > 1. Singleton
    strata are always assigned to train.

    Args:
        df: Labelled rows only (index preserved).
        label_col: Primary stratification column (ground_truth_label).
        group_cols: Additional stratification columns (e.g. source_batch).
        test_fraction: Fraction of each stratum assigned to test.
        seed: RNG seed for reproducible shuffles.

    Returns:
        tuple[list[int], list[int]]: (train_indices, test_indices).
    """
    rng = random.Random(seed)
    strat_key = df[label_col].astype(str)
    for col in group_cols:
        strat_key = strat_key + "__" + df[col].astype(str)

    train_idx: list[int] = []
    test_idx: list[int] = []

    grouped = df.groupby(strat_key, sort=False)
    for _, group in grouped:
        indices = list(group.index)
        rng.shuffle(indices)
        n = len(indices)
        if n == 1:
            train_idx.extend(indices)
            continue

        n_test = int(round(n * test_fraction))
        n_test = max(1, min(n - 1, n_test))
        test_idx.extend(indices[:n_test])
        train_idx.extend(indices[n_test:])

    return train_idx, test_idx


def build_manifests(
    output_unified: Path = UNIFIED_MANIFEST_CSV,
    output_split: Path = SPLIT_MANIFEST_CSV,
    train_fraction: float = TRAIN_FRACTION,
    seed: int = SPLIT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge Corpus A/B, write the unified manifest, and create the 80/20 split.

    Args:
        output_unified: Destination for our_speech_corpus_unified.csv.
        output_split: Destination for our_speech_corpus_manifest_split_80_20.csv.
        train_fraction: Fraction of labelled rows assigned to train.
        seed: RNG seed for the stratified split.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (unified_manifest, split_manifest).

    Raises:
        ValueError: If duplicate clip_id values are produced after the merge.
    """
    df_a = _load_source_csv(HEET_CLEAN_CSV, SOURCE_BATCH_A)
    df_b = _load_source_csv(HEET_NEW_NEGATIVE_CSV, SOURCE_BATCH_B)
    unified = pd.concat([df_a, df_b], ignore_index=True)

    if unified["clip_id"].duplicated().any():
        dupes = unified.loc[unified["clip_id"].duplicated(), "clip_id"].tolist()
        raise ValueError(f"Duplicate clip_id values: {dupes[:5]}")

    unified = unified[MANIFEST_COLUMNS]
    output_unified.parent.mkdir(parents=True, exist_ok=True)
    unified.to_csv(output_unified, index=False)

    split_df = unified.copy()
    split_df[SPLIT_COLUMN] = SPLIT_VALUES[2]  # unlabeled_pool

    labelled = split_df[split_df["is_labelled"]].copy()
    train_idx, test_idx = _stratified_train_test_indices(
        labelled,
        label_col="ground_truth_label",
        group_cols=["source_batch"],
        test_fraction=1.0 - train_fraction,
        seed=seed,
    )
    split_df.loc[train_idx, SPLIT_COLUMN] = SPLIT_VALUES[0]
    split_df.loc[test_idx, SPLIT_COLUMN] = SPLIT_VALUES[1]

    split_df.to_csv(output_split, index=False)
    return unified, split_df


def main() -> None:
    """CLI entrypoint: build unified and split manifests, then print counts.

    Args:
        None. Command-line flags: ``--unified``, ``--split``, ``--train-fraction``,
        and ``--seed``.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Build our_speech_corpus manifests")
    parser.add_argument("--unified", type=Path, default=UNIFIED_MANIFEST_CSV)
    parser.add_argument("--split", type=Path, default=SPLIT_MANIFEST_CSV)
    parser.add_argument("--train-fraction", type=float, default=TRAIN_FRACTION)
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    args = parser.parse_args()

    unified, split_df = build_manifests(
        output_unified=args.unified,
        output_split=args.split,
        train_fraction=args.train_fraction,
        seed=args.seed,
    )

    labelled = int(unified["is_labelled"].sum())
    print(f"[our_speech_corpus_final manifest] wrote unified: {args.unified} ({len(unified)} rows)")
    print(f"[our_speech_corpus_final manifest] wrote split:   {args.split} ({len(split_df)} rows)")
    print(f"[our_speech_corpus_final manifest] labelled: {labelled} / {len(unified)}")
    print(
        f"[our_speech_corpus_final manifest] split counts: "
        f"train={int((split_df[SPLIT_COLUMN] == 'train').sum())}, "
        f"test={int((split_df[SPLIT_COLUMN] == 'test').sum())}, "
        f"unlabeled_pool={int((split_df[SPLIT_COLUMN] == 'unlabeled_pool').sum())}"
    )


if __name__ == "__main__":
    main()
