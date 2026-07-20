"""
Authors:
    Steven Liem (steven.liem@sydney.edu.au)

Evaluate Empathic-Insight-Voice (EIV) with 36 emotion heads at the full emotion level (not 3-way valence).

Reuses existing per-corpus predictions.csv from
results/empathic_insight_voice_all36/ (ground-truth emotion from cleaned
filenames; prediction = top_emotion). Writes confusion-matrix PNGs under:

  results/empathic_insight_voice_all36/result_diagrams/confusion_matrix_all_emotions/

Usage:
    uv run python src/evaluators/eiv_all_emotions.py
    uv run python src/evaluators/eiv_all_emotions.py --corpus our_speech_corpus_cleaned
    uv run python src/evaluators/eiv_all_emotions.py --dry-run

References:
    https://huggingface.co/laion/Empathic-Insight-Voice-Small
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)

WORKSPACE = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = WORKSPACE / "results" / "empathic_insight_voice_all36"
DIAGRAMS_ROOT = RESULTS_ROOT / "result_diagrams" / "confusion_matrix_all_emotions"
METRICS_ROOT = RESULTS_ROOT / "all_emotions"
DEFAULT_HEET = PROJECT_ROOT / "heet_dataset_clean.csv"
VALENCE = ("positive", "neutral", "negative")

# map possibly inconsistent emotion labels to preferred order. Unknown is discarded.
EMOTION_ALIASES = {
    "fear": "fearful",
    "disgust": "disgusted",
    "unk": "unknown",
    "<unk>": "unknown",
}

PREFERRED_ORDER = (
    "angry",
    "disgusted",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised",
    "frustrated",
    "cheerful",
    "other",
    "unknown",
)

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


def normalise_emotion(label: object) -> str:
    """Lowercase an emotion label and apply spelling aliases.

    Args:
        label: Raw emotion label from a filename or prediction.

    Returns:
        str: Normalised emotion label, or ``unknown`` when empty.
    """
    text = str(label).strip().lower()
    if text.startswith("emo_"):
        text = text[4:]
    if "/" in text:
        text = text.split("/")[-1].strip()
    return EMOTION_ALIASES.get(text, text) or "unknown"


def parse_ground_truth_emotion(filename: str) -> str | None:
    """Parse the emotion from a cleaned audio filename.

    Args:
        filename: Filename ending in ``_{emotion}_{valence}.wav``.

    Returns:
        str | None: Normalised emotion, or None when the name cannot be parsed.
    """
    parts = Path(filename).stem.split("_")
    if len(parts) < 2:
        return None
    return normalise_emotion(parts[-2])


def ordered_labels(y_true: list[str], y_pred: list[str]) -> list[str]:
    """Collect labels present in the data using a stable display order.

    Args:
        y_true: Ground-truth emotion labels.
        y_pred: Predicted emotion labels.

    Returns:
        list[str]: Present labels in preferred order, followed by extras.
    """
    present = set(y_true) | set(y_pred)
    ordered = [lab for lab in PREFERRED_ORDER if lab in present]
    extras = sorted(present - set(ordered))
    return ordered + extras


def load_human_gt_filenames(heet_path: Path) -> set[str]:
    """Return wav filenames that have a human valence ground-truth label.

    Args:
        heet_path: Path to heet_dataset_clean.csv.

    Returns:
        set[str]: Filenames with positive/neutral/negative human labels.
    """
    if not heet_path.is_file():
        return set()
    heet = pd.read_csv(heet_path)
    labels = heet["ground_truth_label"].fillna("").astype(str).str.strip().str.lower()
    names = heet["audio_path"].fillna("").map(lambda p: Path(str(p)).name)
    return set(names[labels.isin(VALENCE)])


def load_predictions(corpus: str) -> pd.DataFrame:
    """Load one corpus and attach normalised true and predicted emotions.

    Args:
        corpus: Cleaned corpus directory name.

    Returns:
        pd.DataFrame: Predictions with ground-truth and predicted emotion columns.
    """
    path = RESULTS_ROOT / corpus / "predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions for {corpus}: {path}")

    df = pd.read_csv(path)
    if "ground_truth_emotion" not in df.columns or df["ground_truth_emotion"].isna().all():
        df["ground_truth_emotion"] = df["filename"].map(parse_ground_truth_emotion)
    else:
        df["ground_truth_emotion"] = df["ground_truth_emotion"].map(normalise_emotion)
    # EIV all-36 stores the top emotion head name in top_emotion.
    pred_src = "top_emotion" if "top_emotion" in df.columns else "top_emotion_stem"
    df["pred_emotion"] = df[pred_src].map(normalise_emotion)
    return df


def compute_metrics(df: pd.DataFrame) -> dict:
    """Compute full-emotion metrics (no valence binning).

    Args:
        df: Predictions containing ground-truth and predicted emotion columns.

    Returns:
        dict: Counts, labels, scalar metrics, confusion matrices, and report.
    """
    evaluated = df[
        df["ground_truth_emotion"].notna() & df["pred_emotion"].notna()
    ].copy()
    y_true = evaluated["ground_truth_emotion"].tolist()
    y_pred = evaluated["pred_emotion"].tolist()
    gt_labels = ordered_labels(y_true, y_true)
    cm_labels = ordered_labels(y_true, y_pred)

    if not y_true:
        return {
            "n_total": int(len(df)),
            "n_evaluated": 0,
            "labels": [],
            "gt_labels": [],
            "accuracy": None,
            "balanced_accuracy": None,
            "macro_f1": None,
            "weighted_f1": None,
            "confusion_matrix": [],
            "confusion_matrix_normalised": [],
            "classification_report": "No evaluated rows.",
        }

    cm = confusion_matrix(y_true, y_pred, labels=cm_labels)
    cm_norm = confusion_matrix(y_true, y_pred, labels=cm_labels, normalize="true")
    report = classification_report(
        y_true, y_pred, labels=gt_labels, digits=4, zero_division=0
    )
    return {
        "n_total": int(len(df)),
        "n_evaluated": int(len(evaluated)),
        "labels": cm_labels,
        "gt_labels": gt_labels,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            recall_score(
                y_true, y_pred, labels=gt_labels, average="macro", zero_division=0
            )
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=gt_labels, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(
                y_true, y_pred, labels=gt_labels, average="weighted", zero_division=0
            )
        ),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalised": cm_norm.tolist(),
        "classification_report": report,
        "gt_emotion_counts": evaluated["ground_truth_emotion"].value_counts().to_dict(),
        "pred_emotion_counts": evaluated["pred_emotion"].value_counts().to_dict(),
        "pred_only_labels": sorted(set(y_pred) - set(y_true)),
    }


def plot_confusion_matrices(corpus: str, metrics: dict) -> None:
    """Save count and row-normalised emotion confusion matrices.

    Args:
        corpus: Corpus name used in plot titles and filenames.
        metrics: Metrics dictionary containing labels and confusion matrices.

    Returns:
        None
    """
    cm = np.asarray(metrics.get("confusion_matrix") or [])
    cm_norm = np.asarray(metrics.get("confusion_matrix_normalised") or [])
    labels = metrics.get("labels") or []
    if cm.size == 0 or not labels:
        return

    DIAGRAMS_ROOT.mkdir(parents=True, exist_ok=True)
    n = len(labels)
    figsize = (max(6.0, 0.7 * n + 2), max(5.0, 0.65 * n + 1.5))
    title_suffix = " (human-GT subset)" if "human_gt" in corpus else ""

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted emotion")
    ax.set_ylabel("True emotion")
    ax.set_title(f"{corpus} EIV-36 emotion confusion matrix (counts){title_suffix}")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    fig.savefig(DIAGRAMS_ROOT / f"{corpus}_confusion_counts.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        vmin=0,
        vmax=1,
    )
    ax.set_xlabel("Predicted emotion")
    ax.set_ylabel("True emotion")
    ax.set_title(
        f"{corpus} EIV-36 emotion confusion matrix (row-normalised){title_suffix}"
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    fig.savefig(DIAGRAMS_ROOT / f"{corpus}_confusion_normalised.png", dpi=160)
    plt.close(fig)


def write_corpus_outputs(corpus: str, predictions: pd.DataFrame, metrics: dict) -> None:
    """Write emotion-level predictions, metrics, report, and plots.

    Args:
        corpus: Corpus name used for the output directory.
        predictions: Evaluated prediction records.
        metrics: Computed full-emotion metrics.

    Returns:
        None
    """
    out_dir = METRICS_ROOT / corpus
    out_dir.mkdir(parents=True, exist_ok=True)

    cols = [
        "corpus",
        "filename",
        "ground_truth_emotion",
        "pred_emotion",
        "top_emotion_prob",
        "prompted_valence",
        "ground_truth_valence",
        "pred_top1",
        "pred_valence_head",
        "pred_mass",
        "human_ground_truth_label",
    ]
    keep = [c for c in cols if c in predictions.columns]
    predictions[keep].to_csv(out_dir / "predictions.csv", index=False)

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    (out_dir / "classification_report.txt").write_text(
        metrics.get("classification_report", ""), encoding="utf-8"
    )
    plot_confusion_matrices(corpus, metrics)


def main() -> None:
    """Evaluate selected corpora and write per-corpus and summary outputs.

    Args:
        None

    Returns:
        None
    """
    global RESULTS_ROOT, DIAGRAMS_ROOT, METRICS_ROOT

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=RESULTS_ROOT,
        help="EIV all-36 results directory",
    )
    parser.add_argument(
        "--corpus",
        action="append",
        default=None,
        help="Evaluate only this corpus (repeatable). Default: all with predictions.",
    )
    parser.add_argument(
        "--heet",
        type=Path,
        default=DEFAULT_HEET,
        help="HEET CSV used to select the human-GT subset for our speech",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the first available corpus only",
    )
    args = parser.parse_args()

    RESULTS_ROOT = args.results_root
    DIAGRAMS_ROOT = RESULTS_ROOT / "result_diagrams" / "confusion_matrix_all_emotions"
    METRICS_ROOT = RESULTS_ROOT / "all_emotions"

    human_gt_files = load_human_gt_filenames(args.heet)
    print(f"[eiv-all-emotions] human-GT filenames = {len(human_gt_files)} from {args.heet}")

    corpora = args.corpus or list(CORPUS_DIRS)
    available = [
        c for c in corpora if (RESULTS_ROOT / c / "predictions.csv").exists()
    ]
    if args.dry_run:
        if not available:
            raise FileNotFoundError(f"No predictions under {RESULTS_ROOT}")
        available = [available[0]]

    summary_rows: list[dict] = []

    # loop through each corpus and row and compute metrics
    for corpus in available:
        print(f"[eiv-all-emotions] {corpus}")
        predictions = load_predictions(corpus)
        metrics = compute_metrics(predictions)
        write_corpus_outputs(corpus, predictions, metrics)
        summary_rows.append(
            {
                "corpus": corpus,
                "n_total": metrics["n_total"],
                "n_evaluated": metrics["n_evaluated"],
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "n_labels": len(metrics.get("labels") or []),
            }
        )
        print(
            f"  acc={metrics['accuracy']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f} "
            f"labels={metrics.get('labels')}"
        )

        if corpus == "our_speech_corpus_cleaned" and human_gt_files:
            human_pred = predictions[predictions["filename"].isin(human_gt_files)].copy()
            human_name = f"{corpus}_human_gt"
            human_metrics = compute_metrics(human_pred)
            write_corpus_outputs(human_name, human_pred, human_metrics)
            gt_counts = human_pred["ground_truth_emotion"].value_counts().to_dict()
            print(
                f"[eiv-all-emotions] {human_name}: "
                f"n_labelled_files={len(human_gt_files)} "
                f"n_matched={len(human_pred)} "
                f"n_evaluated={human_metrics['n_evaluated']} "
                f"gt_emotion_counts={gt_counts}"
            )
            print(
                f"  acc={human_metrics['accuracy']:.4f} "
                f"macro_f1={human_metrics['macro_f1']:.4f} "
                f"labels={human_metrics.get('labels')}"
            )

            # add human_gt metrics to summary_rows
            summary_rows.append(
                {
                    "corpus": human_name,
                    "n_total": human_metrics["n_total"],
                    "n_evaluated": human_metrics["n_evaluated"],
                    "accuracy": human_metrics["accuracy"],
                    "balanced_accuracy": human_metrics["balanced_accuracy"],
                    "macro_f1": human_metrics["macro_f1"],
                    "weighted_f1": human_metrics["weighted_f1"],
                    "n_labels": len(human_metrics.get("labels") or []),
                }
            )

    METRICS_ROOT.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(METRICS_ROOT / "summary.csv", index=False)
    with (METRICS_ROOT / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary_rows, file, indent=2)

    print(f"[eiv-all-emotions] diagrams -> {DIAGRAMS_ROOT}")
    print(f"[eiv-all-emotions] metrics  -> {METRICS_ROOT}")


if __name__ == "__main__":
    main()
