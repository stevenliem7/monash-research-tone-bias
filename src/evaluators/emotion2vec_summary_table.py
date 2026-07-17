#!/usr/bin/env python3
"""
Build a per-corpus summary table from emotion2vec valence metrics (3-way).

Reads results/emotion2vec/<corpus>/metrics.json for all 8 cleaned corpora and
writes a CSV with:

  Acc, Macro-F1, Bal. Acc,
  Pos P, Pos R, Pos F1,
  Neu P, Neu R, Neu F1,
  Neg P, Neg R, Neg F1,
  F1 (weighted)

Usage:
    uv run python emotion2vec_summary_table.py
    uv run python emotion2vec_summary_table.py --results-root path/to/emotion2vec
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = WORKSPACE / "results" / "emotion2vec"

CORPUS_DIRS = (
    "emovoice_cleaned",
    "iemocap_human_cleaned",
    "iemocap_synth_cleaned",
    "tess_human_cleaned",
    "tess_indextts_cleaned",
    "deepdialogue_xtts_cleaned",
    "styletalk_cleaned",
    "our_speech_corpus_cleaned",
)

COLUMNS = [
    "corpus",
    "Acc",
    "Macro-F1",
    "Bal. Acc",
    "Pos P",
    "Pos R",
    "Pos F1",
    "Neu P",
    "Neu R",
    "Neu F1",
    "Neg P",
    "Neg R",
    "Neg F1",
    "F1",
]


def row_from_metrics(corpus: str, metrics: dict) -> dict:
    pc = metrics.get("per_class") or {}
    pos = pc.get("positive") or {}
    neu = pc.get("neutral") or {}
    neg = pc.get("negative") or {}

    def g(d: dict, key: str):
        v = d.get(key)
        return float(v) if v is not None else None

    return {
        "corpus": corpus,
        "Acc": g(metrics, "accuracy"),
        "Macro-F1": g(metrics, "macro_f1"),
        "Bal. Acc": g(metrics, "balanced_accuracy"),
        "Pos P": g(pos, "precision"),
        "Pos R": g(pos, "recall"),
        "Pos F1": g(pos, "f1"),
        "Neu P": g(neu, "precision"),
        "Neu R": g(neu, "recall"),
        "Neu F1": g(neu, "f1"),
        "Neg P": g(neg, "precision"),
        "Neg R": g(neg, "recall"),
        "Neg F1": g(neg, "f1"),
        "F1": g(metrics, "weighted_f1"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS,
        help="emotion2vec results directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV (default: <results-root>/summary_metrics_table.csv)",
    )
    args = parser.parse_args()

    results_root = args.results_root
    out_path = args.output or (results_root / "summary_metrics_table.csv")

    rows: list[dict] = []
    for corpus in CORPUS_DIRS:
        path = results_root / corpus / "metrics.json"
        if not path.exists():
            print(f"[skip] missing {path}")
            continue
        with path.open(encoding="utf-8") as f:
            metrics = json.load(f)
        rows.append(row_from_metrics(corpus, metrics))

    if not rows:
        raise FileNotFoundError(f"No metrics.json found under {results_root}")

    df = pd.DataFrame(rows)[COLUMNS]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, float_format="%.4f")

    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}" if pd.notna(x) else ""))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
