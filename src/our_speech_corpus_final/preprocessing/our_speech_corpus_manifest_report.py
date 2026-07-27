#!/usr/bin/env python3
"""
Authors:
    Steven Liem (steven.liem@sydney.edu.au)

Generate a QC report for the unified our_speech_corpus manifests. The full
pipeline is as follows:
  1. Load our_speech_corpus_unified.csv and our_speech_corpus_manifest_split_80_20.csv
  2. Check audio file existence and duration coverage
  3. Summarise labelled / unlabelled counts and class balance by source_batch
  4. Summarise train / test / unlabeled_pool split balance
  5. Optionally attach an embedding_qc section from extract_emotion2vec_embeddings.py
  6. Write our_speech_corpus_qc_report.json

Usage:
    uv run python src/our_speech_corpus_final/our_speech_corpus_manifest_report.py
    uv run python src/our_speech_corpus_final/our_speech_corpus_manifest_report.py \\
        --unified data/our_speech_corpus/our_speech_corpus_unified.csv \\
        --split data/our_speech_corpus/our_speech_corpus_manifest_split_80_20.csv \\
        --output data/our_speech_corpus/our_speech_corpus_qc_report.json

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
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import bootstrap_imports, extend_our_speech_corpus_paths

bootstrap_imports(__file__)
extend_our_speech_corpus_paths()
from config import (
    MANIFEST_COLUMNS,
    QC_REPORT_JSON,
    SPLIT_COLUMN,
    SPLIT_MANIFEST_CSV,
    UNIFIED_MANIFEST_CSV,
    VALENCE_CLASSES,
)


def _count_table(df: pd.DataFrame, cols: list[str]) -> list[dict]:
    """Group-count rows into a list of JSON-serialisable records.

    Args:
        df: Input DataFrame.
        cols: Column names to group by.

    Returns:
        list[dict]: One record per group with a ``count`` field.
    """
    if not cols:
        return []
    grouped = df.groupby(cols, dropna=False).size().reset_index(name="count")
    return grouped.to_dict(orient="records")


def _duration_stats_by_label(df: pd.DataFrame) -> list[dict]:
    """Compute per-class duration mean/median for labelled rows.

    Args:
        df: Unified manifest with is_labelled, ground_truth_label, duration_s.

    Returns:
        list[dict]: Per ground_truth_label duration summary rows.
    """
    labelled = df[df["is_labelled"]].copy()
    duration = pd.to_numeric(labelled["duration_s"], errors="coerce")
    rows: list[dict] = []
    for label, grp in labelled.groupby("ground_truth_label", dropna=False):
        d = duration.loc[grp.index]
        rows.append(
            {
                "ground_truth_label": str(label),
                "count": int(len(grp)),
                "duration_mean_s": float(d.mean()) if d.notna().any() else None,
                "duration_median_s": float(d.median()) if d.notna().any() else None,
            }
        )
    return rows


def build_qc_report(
    unified_path: Path = UNIFIED_MANIFEST_CSV,
    split_path: Path = SPLIT_MANIFEST_CSV,
    embedding_summary: dict | None = None,
) -> dict:
    """Assemble the QC report dictionary from unified and split manifests.

    Args:
        unified_path: Path to our_speech_corpus_unified.csv.
        split_path: Path to our_speech_corpus_manifest_split_80_20.csv.
        embedding_summary: Optional embedding QC block from extraction.

    Returns:
        dict: Full QC report payload (row counts, audio, labels, split, schema).
    """
    unified = pd.read_csv(unified_path)
    split_df = pd.read_csv(split_path)

    audio_exists = unified["audio_path"].apply(lambda p: Path(str(p)).is_file())
    duration = pd.to_numeric(unified["duration_s"], errors="coerce")

    invalid_gt = unified.loc[
        unified["is_labelled"]
        & ~unified["ground_truth_label"].astype(str).str.lower().isin(VALENCE_CLASSES),
        "clip_id",
    ].tolist()

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "unified_manifest": str(unified_path),
            "split_manifest": str(split_path),
        },
        "row_counts": {
            "total": int(len(unified)),
            "labelled": int(unified["is_labelled"].sum()),
            "unlabelled": int((~unified["is_labelled"]).sum()),
            "by_source_batch": _count_table(unified, ["source_batch"]),
        },
        "audio_qc": {
            "files_found": int(audio_exists.sum()),
            "files_missing": int((~audio_exists).sum()),
            "duration_s": {
                "count": int(duration.notna().sum()),
                "min": float(duration.min()) if duration.notna().any() else None,
                "max": float(duration.max()) if duration.notna().any() else None,
                "mean": float(duration.mean()) if duration.notna().any() else None,
                "median": float(duration.median()) if duration.notna().any() else None,
            },
            "duration_by_ground_truth_label": _duration_stats_by_label(unified),
        },
        "label_qc": {
            "invalid_ground_truth_label_clip_ids": invalid_gt,
            "label_distribution_overall": _count_table(
                unified[unified["is_labelled"]], ["ground_truth_label"]
            ),
            "label_distribution_by_batch": _count_table(
                unified[unified["is_labelled"]], ["source_batch", "ground_truth_label"]
            ),
        },
        "split_qc": {
            "split_counts": _count_table(split_df, [SPLIT_COLUMN]),
            "split_by_batch_and_label": _count_table(
                split_df[split_df["is_labelled"]],
                [SPLIT_COLUMN, "source_batch", "ground_truth_label"],
            ),
        },
        "schema": {
            "expected_columns": MANIFEST_COLUMNS + [SPLIT_COLUMN],
            "unified_columns": list(unified.columns),
            "split_columns": list(split_df.columns),
        },
        "embedding_qc": embedding_summary or {},
    }
    return report


def write_qc_report(
    output_path: Path = QC_REPORT_JSON,
    unified_path: Path = UNIFIED_MANIFEST_CSV,
    split_path: Path = SPLIT_MANIFEST_CSV,
    embedding_summary: dict | None = None,
) -> dict:
    """Build the QC report and write it to JSON.

    Args:
        output_path: Destination for our_speech_corpus_qc_report.json.
        unified_path: Path to the unified manifest CSV.
        split_path: Path to the split manifest CSV.
        embedding_summary: Optional embedding QC block to attach.

    Returns:
        dict: The report that was written.
    """
    report = build_qc_report(unified_path, split_path, embedding_summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


def main() -> None:
    """CLI entrypoint: write the QC report and print a short summary.

    Args:
        None. Command-line flags: ``--unified``, ``--split``, and ``--output``.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Write our_speech_corpus QC report")
    parser.add_argument("--unified", type=Path, default=UNIFIED_MANIFEST_CSV)
    parser.add_argument("--split", type=Path, default=SPLIT_MANIFEST_CSV)
    parser.add_argument("--output", type=Path, default=QC_REPORT_JSON)
    args = parser.parse_args()

    report = write_qc_report(
        output_path=args.output,
        unified_path=args.unified,
        split_path=args.split,
    )
    print(f"[qc] wrote {args.output}")
    print(
        f"[qc] rows={report['row_counts']['total']}, "
        f"labelled={report['row_counts']['labelled']}, "
        f"audio_found={report['audio_qc']['files_found']}"
    )


if __name__ == "__main__":
    main()
