#!/usr/bin/env python3
"""
Authors:
    Steven Liem (steven.liem@sydney.edu.au)

Build a per-corpus summary table from emotion2vec valence metrics (3-way).

Reads results/emotion2vec/<corpus>/metrics.json for all 8 cleaned corpora and
writes a CSV with:

  Acc, Macro-F1, Bal. Acc,
  Pos P, Pos R, Pos F1,
  Neu P, Neu R, Neu F1,
  Neg P, Neg R, Neg F1,
  F1 (weighted)

Our-speech rows are split:
  *_prompted  — metrics vs filename/prompted valence (not human labels)
  *_human_GT  — metrics vs HEET human labels (human-labelled ∩ emotion2vec-mappable, n=229)

Usage:
    uv run python src/evaluators/emotion2vec_summary_table.py
    uv run python src/evaluators/emotion2vec_summary_table.py --results-root path/to/emotion2vec
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from human_gt_comparable import (
    DEFAULT_E2V_PREDS,
    DEFAULT_HEET,
    build_comparable,
    evaluate_human_gt,
    print_coverage,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = Path(__file__).resolve().parents[3]
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
    """Convert a nested metrics dictionary into one summary-table row.

    Args:
        corpus: Corpus name written to the first column.
        metrics: Aggregate and per-class emotion2vec metrics.

    Returns:
        dict: Flat summary row containing accuracy, precision, recall, and F1.
    """
    pc = metrics.get("per_class") or {}
    pos = pc.get("positive") or {}
    neu = pc.get("neutral") or {}
    neg = pc.get("negative") or {}

    def get_metric(d: dict, key: str):
        """Read one optional metric and convert it to a float.

        Args:
            d: Dictionary containing metric values.
            key: Metric key to retrieve.

        Returns:
            float | None: Metric value, or None when the key is absent.
        """
        v = d.get(key)
        return float(v) if v is not None else None

    return {
        "corpus": corpus,
        "Acc": get_metric(metrics, "accuracy"),
        "Macro-F1": get_metric(metrics, "macro_f1"),
        "Bal. Acc": get_metric(metrics, "balanced_accuracy"),
        "Pos P": get_metric(pos, "precision"),
        "Pos R": get_metric(pos, "recall"),
        "Pos F1": get_metric(pos, "f1"),
        "Neu P": get_metric(neu, "precision"),
        "Neu R": get_metric(neu, "recall"),
        "Neu F1": get_metric(neu, "f1"),
        "Neg P": get_metric(neg, "precision"),
        "Neg R": get_metric(neg, "recall"),
        "Neg F1": get_metric(neg, "f1"),
        "F1": get_metric(metrics, "weighted_f1"),
    }


def display_corpus_name(corpus: str) -> str:
    """Rename our-speech prompted-valence metrics so they are not misread as human GT.

    Args:
        corpus: Raw corpus directory name.

    Returns:
        str: Display name used in the summary CSV.
    """
    if corpus == "our_speech_corpus_cleaned":
        return "our_speech_corpus_cleaned_prompted"
    return corpus


def save_human_gt_confusion_matrix(cm: pd.DataFrame, output_path: Path) -> None:
    """Save the human-GT valence confusion matrix as a PNG.

    Args:
        cm: Count matrix with human GT as rows and emotion2vec predictions as columns.
        output_path: Destination PNG path.

    Returns:
        None
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted valence (emotion2vec)")
    ax.set_ylabel("Human ground-truth valence")
    ax.set_title(f"Our speech corpus: human GT vs emotion2vec (n={int(cm.values.sum())})")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    """Build, print, and save the emotion2vec valence summary table.

    Args:
        None

    Returns:
        None
    """
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
    parser.add_argument(
        "--heet",
        type=Path,
        default=DEFAULT_HEET,
        help="HEET CSV containing the human ground_truth_label values",
    )
    parser.add_argument(
        "--e2v-predictions",
        type=Path,
        default=DEFAULT_E2V_PREDS,
        help="emotion2vec predictions used to define the shared human-GT set",
    )
    args = parser.parse_args()

    results_root = args.results_root
    out_path = args.output or (results_root / "summary_metrics_table.csv")
    predictions_path = args.e2v_predictions
    diagram_path = (
        results_root
        / "result_diagrams"
        / "confusion_matrix_counts"
        / "our_speech_corpus_cleaned_human_gt_confusion_counts.png"
    )

    rows: list[dict] = []
    for corpus in CORPUS_DIRS:
        path = results_root / corpus / "metrics.json"
        if not path.exists():
            print(f"[skip] missing {path}")
            continue
        with path.open(encoding="utf-8") as f:
            metrics = json.load(f)
        rows.append(row_from_metrics(display_corpus_name(corpus), metrics))

    comparable, coverage_stats = build_comparable(args.heet, predictions_path)
    print_coverage(coverage_stats)

    e2v_preds = pd.read_csv(predictions_path)
    metrics, cm, coverage = evaluate_human_gt(
        comparable,
        e2v_preds,
        "pred_valence",
        instrument="emotion2vec",
    )
    save_human_gt_confusion_matrix(cm, diagram_path)
    print(f"[emotion2vec] Confusion matrix diagram: {diagram_path}")
    print(cm.to_string())
    rows.append(row_from_metrics("our_speech_corpus_cleaned_human_GT", metrics))
    print(
        f"[emotion2vec] Human-GT summary: "
        f"n={coverage['n']} "
        f"n_evaluated={coverage['n_evaluated']} "
        f"neg_support={coverage['neg_support']}"
    )

    if not rows:
        raise FileNotFoundError(f"No metrics.json found under {results_root}")

    df = pd.DataFrame(rows)[COLUMNS]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, float_format="%.4f")

    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}" if pd.notna(x) else ""))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
