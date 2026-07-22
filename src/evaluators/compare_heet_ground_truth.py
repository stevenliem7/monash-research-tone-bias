#!/usr/bin/env python3
"""
Authors:
    Steven Liem (steven.liem@sydney.edu.au)

Compare human ground_truth_label in heet_dataset_clean.csv against emotion2vec predictions for our_speech_corpus_cleaned. 
Blank ground-truth cells are ignored.

Usage:
    uv run python compare_heet_ground_truth.py
    uv run python compare_heet_ground_truth.py --heet heet_dataset_clean.csv \\
        --predictions ../results/emotion2vec/our_speech_corpus_cleaned/predictions.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = Path(__file__).resolve().parents[3]
DEFAULT_HEET = PROJECT_ROOT / "heet_dataset_clean.csv"
DEFAULT_PREDICTIONS = (
    WORKSPACE / "results" / "emotion2vec" / "our_speech_corpus_cleaned" / "predictions.csv"
)
DEFAULT_DIAGRAM = (
    WORKSPACE
    / "results"
    / "emotion2vec"
    / "result_diagrams"
    / "confusion_matrix_counts"
    / "our_speech_corpus_cleaned_human_gt_confusion_counts.png"
)
VALENCE = ("positive", "neutral", "negative")


def save_confusion_matrix(cm: pd.DataFrame, output_path: Path) -> None:
    """Save the human-GT confusion matrix as a PNG diagram.

    Args:
        cm: Count matrix with ground truth as rows and predictions as columns.
        output_path: Destination PNG path.

    Returns:
        None
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted valence (emotion2vec)")
    ax.set_ylabel("Human ground-truth valence")
    ax.set_title("Our speech corpus: human GT vs emotion2vec")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main(
    heet_path: Path = DEFAULT_HEET,
    predictions_path: Path = DEFAULT_PREDICTIONS,
    diagram_path: Path = DEFAULT_DIAGRAM,
) -> None:
    """Compare and print human GT against emotion2vec valence predictions.

    Blank or invalid human labels are ignored. Predictions that do not map to
    positive, neutral, or negative are reported as excluded or unmatched.

    Args:
        heet_path: Path to heet_dataset_clean.csv with ground_truth_label.
        predictions_path: Path to emotion2vec predictions.csv.
        diagram_path: Destination path for the confusion-matrix PNG.

    Returns:
        None
    """
    heet = pd.read_csv(heet_path)
    pred = pd.read_csv(predictions_path)

    heet["ground_truth_label"] = (
        heet["ground_truth_label"].fillna("").astype(str).str.strip().str.lower()
    )
    labelled = heet[heet["ground_truth_label"].isin(VALENCE)].copy()
    labelled["filename"] = labelled["audio_path"].fillna("").map(lambda p: Path(str(p)).name)

    merged = labelled.merge(pred, on="filename", how="left")
    comparable = merged[merged["pred_valence"].isin(VALENCE)].copy()
    excluded = len(merged) - len(comparable)
    agree = (comparable["ground_truth_label"] == comparable["pred_valence"]).sum()
    total = len(comparable)
    acc = agree / total if total else 0.0

    print(f"Human-labelled rows: {len(labelled)}")
    print(f"Comparable (non-excluded emotion2vec): {total}")
    print(f"Excluded / unmatched: {excluded}")
    print(f"Agreement: {agree}/{total} = {acc:.3f} ({100 * acc:.1f}%)")
    print("\nConfusion (rows=ground_truth, cols=emotion2vec):")
    cm = pd.crosstab(
        comparable["ground_truth_label"],
        comparable["pred_valence"],
        rownames=["ground_truth"],
        colnames=["emotion2vec"],
    ).reindex(index=list(VALENCE), columns=list(VALENCE), fill_value=0)
    print(cm.to_string())
    save_confusion_matrix(cm, diagram_path)
    print(f"\nConfusion matrix diagram: {diagram_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heet", type=Path, default=DEFAULT_HEET)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--diagram", type=Path, default=DEFAULT_DIAGRAM)
    args = parser.parse_args()
    main(args.heet, args.predictions, args.diagram)
