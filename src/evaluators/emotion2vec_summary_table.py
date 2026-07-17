"""
Build a per-corpus summary table from emotion2vec valence metrics (3-way).

Reads results/emotion2vec/<corpus>/metrics.json for all 8 cleaned corpora and writes a CSV with the following columns:

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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS = WORKSPACE / "results" / "emotion2vec"
DEFAULT_HEET = PROJECT_ROOT / "heet_dataset_clean.csv"
VALENCE = ("positive", "neutral", "negative")

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


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def human_gt_row(heet_path: Path, predictions_path: Path) -> tuple[dict, int, int]:
    heet = pd.read_csv(heet_path)
    predictions = pd.read_csv(predictions_path)
    heet["ground_truth_label"] = (
        heet["ground_truth_label"].fillna("").astype(str).str.strip().str.lower()
    )
    labelled = heet[heet["ground_truth_label"].isin(VALENCE)].copy()
    labelled["filename"] = labelled["audio_path"].fillna("").map(lambda p: Path(str(p)).name)
    comparable = labelled.merge(predictions, on="filename", how="left")
    comparable = comparable[comparable["pred_valence"].isin(VALENCE)]

    cm = pd.crosstab(
        comparable["ground_truth_label"],
        comparable["pred_valence"],
    ).reindex(index=list(VALENCE), columns=list(VALENCE), fill_value=0)

    n = int(cm.values.sum())
    per_class: dict[str, dict[str, float]] = {}
    recalls = []
    f1s = []
    supports = []
    for label in VALENCE:
        tp = float(cm.loc[label, label])
        support = float(cm.loc[label].sum())
        pred_n = float(cm[label].sum())
        precision = _safe_div(tp, pred_n)
        recall = _safe_div(tp, support)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1}
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)

    metrics = {
        "accuracy": _safe_div(float(cm.values.diagonal().sum()), n),
        "macro_f1": sum(f1s) / len(f1s),
        "balanced_accuracy": sum(recalls) / len(recalls),
        "weighted_f1": _safe_div(sum(f * s for f, s in zip(f1s, supports)), sum(supports)),
        "per_class": per_class,
    }
    return (
        row_from_metrics("our_speech_corpus_cleaned_human_GT", metrics),
        len(labelled),
        len(comparable),
    )


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
    parser.add_argument(
        "--heet",
        type=Path,
        default=DEFAULT_HEET,
        help="HEET CSV containing the human ground_truth_label values",
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

    predictions_path = results_root / "our_speech_corpus_cleaned" / "predictions.csv"
    human_row, labelled_count, evaluated_count = human_gt_row(args.heet, predictions_path)
    rows.append(human_row)
    print(
        "Human-GT row: "
        f"{evaluated_count}/{labelled_count} labelled rows evaluated "
        f"({labelled_count - evaluated_count} excluded/unmatched)"
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
