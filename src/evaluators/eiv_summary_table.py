#!/usr/bin/env python3
"""
Authors:
    Steven Liem (steven.liem@sydney.edu.au)

Build Empathic-Insight-Voice all-36 valence summaries from cached metrics / predictions. Writes:

  1. summary_metrics_table_{valence_head,top1,mass}.csv
     Wide Acc / Macro-F1 / per-class columns (same layout as emotion2vec).

  2. summary_comparison.csv
     Long form: corpus, method, n, accuracy, balanced_accuracy, macro_f1,
     neg_precision, neg_recall, pos_f1, neu_f1, neg_f1
     — includes our_speech prompted + human_GT rows for every method.

Our-speech naming:
  *_prompted  — vs filename/prompted valence
  *_human_GT  — vs HEET human labels (human-labelled ∩ emotion2vec-mappable, n=229)

Usage:
    uv run python src/evaluators/eiv_summary_table.py
    uv run python src/evaluators/eiv_summary_table.py --results-root path/to/empathic_insight_voice_all36
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
DEFAULT_RESULTS = WORKSPACE / "results" / "empathic_insight_voice_all36"

METHODS = {
    "valence_head": "pred_valence_head",
    "top1": "pred_top1",
    "mass": "pred_mass",
}

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

WIDE_COLUMNS = [
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

COMPARISON_COLUMNS = [
    "corpus",
    "method",
    "n",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "neg_precision",
    "neg_recall",
    "pos_f1",
    "neu_f1",
    "neg_f1",
]


def display_corpus_name(corpus: str) -> str:
    """Rename our-speech prompted metrics so they are not misread as human GT.

    Args:
        corpus: Raw corpus directory name.

    Returns:
        str: Display name used in summary CSVs.
    """
    if corpus == "our_speech_corpus_cleaned":
        return "our_speech_corpus_cleaned_prompted"
    return corpus


def row_from_metrics(corpus: str, metrics: dict) -> dict:
    """Convert nested metrics into one wide summary-table row.

    Args:
        corpus: Corpus name for the first column.
        metrics: Aggregate and per-class valence metrics.

    Returns:
        dict: Wide Acc / P / R / F1 row.
    """
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


def comparison_row(corpus: str, method: str, metrics: dict) -> dict:
    """Convert nested metrics into one long summary_comparison row.

    Args:
        corpus: Corpus display name.
        method: valence_head / top1 / mass.
        metrics: Aggregate and per-class valence metrics.

    Returns:
        dict: Long-form comparison row.
    """
    pc = metrics.get("per_class") or {}
    n = metrics.get("n")
    if n is None:
        n = sum(int((pc.get(lab) or {}).get("support") or 0) for lab in ("positive", "neutral", "negative"))

    def g(d: dict, key: str):
        v = d.get(key)
        return float(v) if v is not None else None

    return {
        "corpus": corpus,
        "method": method,
        "n": int(n) if n is not None else None,
        "accuracy": g(metrics, "accuracy"),
        "balanced_accuracy": g(metrics, "balanced_accuracy"),
        "macro_f1": g(metrics, "macro_f1"),
        "neg_precision": g(pc.get("negative") or {}, "precision"),
        "neg_recall": g(pc.get("negative") or {}, "recall"),
        "pos_f1": g(pc.get("positive") or {}, "f1"),
        "neu_f1": g(pc.get("neutral") or {}, "f1"),
        "neg_f1": g(pc.get("negative") or {}, "f1"),
    }


def save_human_gt_confusion_matrix(cm: pd.DataFrame, method: str, output_path: Path) -> None:
    """Save the human-GT valence confusion matrix as a PNG.

    Args:
        cm: Count matrix with human GT as rows and EIV predictions as columns.
        method: Aggregation method name used in the plot title.
        output_path: Destination PNG path.

    Returns:
        None
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel(f"Predicted valence (EIV {method})")
    ax.set_ylabel("Human ground-truth valence")
    ax.set_title(f"Our speech: human GT vs EIV {method} (n={int(cm.values.sum())})")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    """Build wide per-method tables and a long summary_comparison.csv.

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
        help="EIV all-36 results directory",
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
    predictions_path = results_root / "our_speech_corpus_cleaned" / "predictions.csv"
    diagram_dir = results_root / "result_diagrams" / "confusion_matrix_counts"

    comparable, coverage_stats = build_comparable(args.heet, args.e2v_predictions)
    print_coverage(coverage_stats)

    eiv_preds = pd.read_csv(predictions_path) if predictions_path.exists() else None
    comparison_rows: list[dict] = []

    for method, pred_col in METHODS.items():
        wide_rows: list[dict] = []

        for corpus in CORPUS_DIRS:
            path = results_root / corpus / "metrics.json"
            if not path.exists():
                print(f"[skip] missing {path}")
                continue
            with path.open(encoding="utf-8") as f:
                block = json.load(f)
            method_metrics = (block.get("methods") or {}).get(method)
            if not method_metrics:
                print(f"[skip] {corpus}: no method {method}")
                continue

            name = display_corpus_name(corpus)
            wide_rows.append(row_from_metrics(name, method_metrics))
            comparison_rows.append(comparison_row(name, method, method_metrics))

        if eiv_preds is not None:
            diagram_path = (
                diagram_dir / f"our_speech_corpus_cleaned_human_gt_{method}_cm.png"
            )
            metrics, cm, coverage = evaluate_human_gt(
                comparable,
                eiv_preds,
                pred_col,
                instrument=f"eiv_{method}",
            )
            # Ensure n is present for comparison CSV.
            metrics = {**metrics, "n": coverage["n_evaluated"]}
            save_human_gt_confusion_matrix(cm, method, diagram_path)
            print(f"[eiv_{method}] Confusion matrix diagram: {diagram_path}")
            print(cm.to_string())
            print(
                f"[eiv_{method}] Human-GT summary: "
                f"n={coverage['n']} "
                f"n_evaluated={coverage['n_evaluated']} "
                f"neg_support={coverage['neg_support']}"
            )
            gt_name = "our_speech_corpus_cleaned_human_GT"
            wide_rows.append(row_from_metrics(gt_name, metrics))
            comparison_rows.append(comparison_row(gt_name, method, metrics))

        if not wide_rows:
            print(f"[skip] no rows for method {method}")
            continue

        out_path = results_root / f"summary_metrics_table_{method}.csv"
        wide = pd.DataFrame(wide_rows)[WIDE_COLUMNS]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        wide.to_csv(out_path, index=False, float_format="%.4f")
        print(f"\n=== {method} ===")
        print(wide.to_string(index=False, float_format=lambda x: f"{x:.4f}" if pd.notna(x) else ""))
        print(f"Wrote {out_path}")

    if not comparison_rows:
        raise FileNotFoundError(f"No metrics found under {results_root}")

    comparison = pd.DataFrame(comparison_rows)[COMPARISON_COLUMNS]
    # Stable order: corpora as listed, prompted then human_GT, methods in METHODS order.
    method_order = list(METHODS.keys())
    comparison["method"] = pd.Categorical(comparison["method"], method_order, ordered=True)
    comparison = comparison.sort_values(["corpus", "method"]).reset_index(drop=True)
    comparison_path = results_root / "summary_comparison.csv"
    comparison.to_csv(comparison_path, index=False, float_format="%.6g")
    print(f"\n=== summary_comparison ===")
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)))
    print(f"Wrote {comparison_path}")


if __name__ == "__main__":
    main()
