#!/usr/bin/env python3
"""
Authors:
    Steven Liem (steven.liem@sydney.edu.au)

Train a 3-class valence head on frozen emotion2vec embeddings.
The full pipeline is as follows:
  1. Load our_speech_corpus_embeddings.npz (clip_id → 1024-d vector)
  2. Join with the locked 80/20 split manifest on clip_id
  3. Select among capacity-controlled heads by stratified CV on train only
  4. Refit the winner on all train rows
  5. Evaluate once on the held-out test set
  6. Write our_speech_corpus_valence_head_results.json

Candidate heads (chosen for small-n negatives, ~76 train negatives):
  A — PCA (32 comps) → multinomial logistic
  B — L2-regularised logistic on full 1024-d features
  C — Linear discriminant analysis (shrinkage)

Target is ground_truth_label only (positive / neutral / negative).
emotion_label and valence_label are not used as targets.

This stage is CPU-cheap (seconds locally). Kaggle GPU is not required.

Usage:
    uv run python src/our_speech_corpus_final/train_valence_head.py
    uv run python src/our_speech_corpus_final/train_valence_head.py \\
        --embeddings src/our_speech_corpus_final/our_speech_corpus_embeddings.npz \\
        --manifest data/our_speech_corpus/our_speech_corpus_manifest_split_80_20.csv

References:
    https://huggingface.co/emotion2vec/emotion2vec_plus_large
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

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import bootstrap_imports, extend_our_speech_corpus_paths

bootstrap_imports(__file__)
extend_our_speech_corpus_paths()
from config import (
    EMBEDDINGS_NPZ,
    HEAD_RESULTS_JSON,
    SPLIT_MANIFEST_CSV,
    SPLIT_SEED,
    VALENCE_CLASSES,
)

LABEL_TO_ID = {label: i for i, label in enumerate(VALENCE_CLASSES)}
ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}


def load_xy(
    embeddings_path: Path,
    manifest_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Join embeddings with the split manifest into train/test matrices.

    Args:
        embeddings_path: Path to our_speech_corpus_embeddings.npz.
        manifest_path: Path to our_speech_corpus_manifest_split_80_20.csv.

    Returns:
        tuple: (X, y, is_train_mask, labelled_frame) where X is (n, d), y is
        integer class ids, and labelled_frame is the joined labelled rows.

    Raises:
        ValueError: If clip_ids cannot be aligned or labels are invalid.
    """
    bundle = np.load(embeddings_path, allow_pickle=True)
    emb_ids = [str(x) for x in bundle["clip_id"].tolist()]
    emb_matrix = np.asarray(bundle["embeddings"], dtype=np.float32)
    emb_lookup = {cid: emb_matrix[i] for i, cid in enumerate(emb_ids)}

    manifest = pd.read_csv(manifest_path)
    labelled = manifest[manifest["is_labelled"]].copy()
    labelled = labelled[labelled["split"].isin(["train", "test"])].copy()

    vectors: list[np.ndarray] = []
    labels: list[int] = []
    keep_idx: list[int] = []
    for idx, row in labelled.iterrows():
        clip_id = str(row["clip_id"])
        gt = str(row["ground_truth_label"]).strip().lower()
        if clip_id not in emb_lookup:
            continue
        if gt not in LABEL_TO_ID:
            raise ValueError(f"Unexpected ground_truth_label={gt!r} for {clip_id}")
        vectors.append(emb_lookup[clip_id])
        labels.append(LABEL_TO_ID[gt])
        keep_idx.append(idx)

    if not vectors:
        raise ValueError("No labelled rows matched embeddings by clip_id")

    frame = labelled.loc[keep_idx].reset_index(drop=True)
    X = np.vstack(vectors)
    y = np.asarray(labels, dtype=np.int64)
    is_train = frame["split"].to_numpy() == "train"
    return X, y, is_train, frame


def build_candidates(seed: int = SPLIT_SEED) -> dict[str, Pipeline]:
    """Build the three capacity-controlled candidate heads.

    Args:
        seed: Random state for logistic solvers / PCA.

    Returns:
        dict[str, Pipeline]: Named sklearn pipelines.
    """
    return {
        "pca32_logistic": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=32, random_state=seed)),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "l2_logistic": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=0.1,
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "shrinkage_lda": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
                ),
            ]
        ),
    }


def select_by_cv(
    X_train: np.ndarray,
    y_train: np.ndarray,
    candidates: dict[str, Pipeline],
    seed: int = SPLIT_SEED,
    n_splits: int = 5,
) -> tuple[str, dict[str, float]]:
    """Pick the best head by stratified CV macro-F1 on train only.

    Args:
        X_train: Training embeddings.
        y_train: Training integer labels.
        candidates: Named model pipelines.
        seed: CV shuffle seed.
        n_splits: Number of stratified folds.

    Returns:
        tuple[str, dict[str, float]]: Winning name and CV macro-F1 per candidate.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores: dict[str, float] = {}
    for name, model in candidates.items():
        fold_scores = cross_val_score(
            model, X_train, y_train, cv=cv, scoring="f1_macro"
        )
        scores[name] = float(fold_scores.mean())
        print(f"[cv] {name}: macro-F1={scores[name]:.3f} ± {fold_scores.std():.3f}")
    winner = max(scores, key=scores.get)
    return winner, scores


def evaluate(
    model: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    split_name: str,
) -> dict:
    """Compute accuracy, macro/weighted F1, per-class metrics, and confusion matrix.

    Args:
        model: Fitted sklearn pipeline.
        X: Feature matrix.
        y: Integer labels.
        split_name: Label for logging (train/test).

    Returns:
        dict: Metrics payload.
    """
    pred = model.predict(X)
    report = classification_report(
        y,
        pred,
        labels=list(range(len(VALENCE_CLASSES))),
        target_names=list(VALENCE_CLASSES),
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y, pred, labels=list(range(len(VALENCE_CLASSES))))
    payload = {
        "split": split_name,
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "per_class": {
            label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1": float(report[label]["f1-score"]),
                "support": int(report[label]["support"]),
            }
            for label in VALENCE_CLASSES
        },
        "confusion_matrix": {
            "labels": list(VALENCE_CLASSES),
            "matrix": cm.tolist(),
        },
    }
    print(
        f"[{split_name}] n={payload['n']} acc={payload['accuracy']:.3f} "
        f"macro-F1={payload['macro_f1']:.3f} "
        f"neg_recall={payload['per_class']['negative']['recall']:.3f}"
    )
    return payload


def majority_baseline(y_train: np.ndarray, y_test: np.ndarray) -> dict:
    """Majority-class baseline (always predict the train-mode class).

    Args:
        y_train: Training labels.
        y_test: Test labels.

    Returns:
        dict: Accuracy / macro-F1 of the majority baseline on test.
    """
    majority = int(np.bincount(y_train).argmax())
    pred = np.full_like(y_test, majority)
    return {
        "predicted_class": ID_TO_LABEL[majority],
        "test_accuracy": float(accuracy_score(y_test, pred)),
        "test_macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
    }


def train_and_eval(
    embeddings_path: Path = EMBEDDINGS_NPZ,
    manifest_path: Path = SPLIT_MANIFEST_CSV,
    output_path: Path = HEAD_RESULTS_JSON,
    seed: int = SPLIT_SEED,
) -> dict:
    """Run CV selection, refit, and one-shot held-out evaluation.

    Args:
        embeddings_path: NPZ from emotion2vec extraction.
        manifest_path: Locked split manifest.
        output_path: JSON results destination.
        seed: RNG seed for CV / models.

    Returns:
        dict: Full results payload written to disk.
    """
    X, y, is_train, frame = load_xy(embeddings_path, manifest_path)
    X_train, y_train = X[is_train], y[is_train]
    X_test, y_test = X[~is_train], y[~is_train]

    print(
        f"[data] train={len(y_train)} test={len(y_test)} "
        f"dim={X.shape[1]} classes={dict(zip(VALENCE_CLASSES, np.bincount(y)))}"
    )

    candidates = build_candidates(seed=seed)
    winner_name, cv_scores = select_by_cv(X_train, y_train, candidates, seed=seed)
    print(f"[select] winner={winner_name}")

    model = candidates[winner_name]
    model.fit(X_train, y_train)

    results = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "embeddings_path": str(embeddings_path),
        "manifest_path": str(manifest_path),
        "embedding_dim": int(X.shape[1]),
        "winner": winner_name,
        "cv_macro_f1": cv_scores,
        "majority_baseline": majority_baseline(y_train, y_test),
        "train": evaluate(model, X_train, y_train, "train"),
        "test": evaluate(model, X_test, y_test, "test"),
        "test_by_source_batch": {},
    }

    test_frame = frame.loc[~is_train].reset_index(drop=True)
    for batch in sorted(test_frame["source_batch"].unique()):
        batch_mask = test_frame["source_batch"].to_numpy() == batch
        results["test_by_source_batch"][str(batch)] = evaluate(
            model, X_test[batch_mask], y_test[batch_mask], f"test:{batch}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[wrote] {output_path}")
    return results


def main() -> None:
    """CLI entrypoint for valence-head training.

    Args:
        None. Command-line flags: ``--embeddings``, ``--manifest``, ``--output``,
        and ``--seed``.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Train valence head on frozen embeddings")
    parser.add_argument("--embeddings", type=Path, default=EMBEDDINGS_NPZ)
    parser.add_argument("--manifest", type=Path, default=SPLIT_MANIFEST_CSV)
    parser.add_argument("--output", type=Path, default=HEAD_RESULTS_JSON)
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    args = parser.parse_args()

    if not args.embeddings.is_file():
        raise FileNotFoundError(f"Embeddings not found: {args.embeddings}")
    if not args.manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    train_and_eval(
        embeddings_path=args.embeddings,
        manifest_path=args.manifest,
        output_path=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
