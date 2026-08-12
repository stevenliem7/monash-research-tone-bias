#!/usr/bin/env python3
"""
Shared helpers for text-modality valence baselines vs frozen WavLM.

Used by MiniLM / DistilBERT CV harnesses and the comparison module.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import SPLIT_SEED, VALENCE_CLASSES

LABEL_TO_ID = {label: i for i, label in enumerate(VALENCE_CLASSES)}
ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}

EXPECTED_N = 854
EXPECTED_COUNTS = {"positive": 242, "neutral": 508, "negative": 104}
EXPECTED_GROUP_STATS = {
    "question": {"unique": 318, "max_size": 14},
    "response_text": {"unique": 838, "max_size": 5},
}

MATCH_TOL = 0.02
AUDIO_GAP = 0.05


def library_versions() -> dict[str, str]:
    """Collect library versions for JSON provenance.

    Returns:
        dict[str, str]: Name -> version string (missing libs omitted).
    """
    out: dict[str, str] = {}
    for name in ("numpy", "pandas", "sklearn", "torch", "transformers", "sentence_transformers"):
        try:
            mod = __import__(name if name != "sklearn" else "sklearn")
            out[name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            continue
    return out


def set_seeds(seed: int = SPLIT_SEED) -> None:
    """Seed NumPy and (if present) PyTorch RNGs.

    Args:
        seed: Integer seed.

    Returns:
        None
    """
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def normalise_text(value: object) -> str:
    """Lowercase + collapse whitespace.

    Args:
        value: Raw cell.

    Returns:
        str: Normalised text, or empty string if missing.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return " ".join(str(value).lower().split())


def load_labelled_manifest(
    manifest_path: Path,
    text_col: str = "response_text",
) -> pd.DataFrame:
    """Load labelled rows and assert corpus invariants from the spec.

    Args:
        manifest_path: Split / unified manifest CSV.
        text_col: Transcript column (must be response_text for these experiments).

    Returns:
        pd.DataFrame: Labelled rows with non-empty normalised transcripts.

    Raises:
        AssertionError / ValueError: On corpus invariant failures.
    """
    manifest = pd.read_csv(manifest_path)
    labelled = manifest[manifest["is_labelled"]].copy().reset_index(drop=True)
    assert len(labelled) == EXPECTED_N, f"expected {EXPECTED_N} labelled, got {len(labelled)}"
    if text_col not in labelled.columns:
        raise ValueError(f"Manifest missing {text_col!r}")

    texts = labelled[text_col].map(normalise_text)
    if (texts == "").any() or labelled[text_col].isna().any():
        n_bad = int(((texts == "") | labelled[text_col].isna()).sum())
        raise ValueError(f"{n_bad} empty/NaN {text_col} values after normalisation")

    labelled = labelled.copy()
    labelled["_norm_text"] = texts

    counts = (
        labelled["ground_truth_label"]
        .astype(str)
        .str.strip()
        .str.lower()
        .value_counts()
        .to_dict()
    )
    assert counts == EXPECTED_COUNTS, f"class counts {counts} != {EXPECTED_COUNTS}"
    return labelled


def labels_from_frame(frame: pd.DataFrame) -> np.ndarray:
    """Map ground_truth_label to canonical integer ids.

    Args:
        frame: Labelled manifest rows.

    Returns:
        np.ndarray: Integer class ids.
    """
    labels: list[int] = []
    for gt in frame["ground_truth_label"].astype(str).str.strip().str.lower():
        if gt not in LABEL_TO_ID:
            raise ValueError(f"Unexpected ground_truth_label={gt!r}")
        labels.append(LABEL_TO_ID[gt])
    return np.asarray(labels, dtype=np.int64)


def assert_group_stats(frame: pd.DataFrame, group_by: str) -> dict[str, int]:
    """Assert unique-group / max-size stats for leakage-check columns.

    Args:
        frame: Labelled rows.
        group_by: ``question`` or ``response_text``.

    Returns:
        dict: Observed unique count and max group size.
    """
    from train_valence_head_cv import group_labels_from_frame

    codes = group_labels_from_frame(frame, group_by)
    unique = int(len(np.unique(codes)))
    max_size = int(np.bincount(codes).max())
    expected = EXPECTED_GROUP_STATS[group_by]
    assert unique == expected["unique"], (
        f"{group_by} unique groups {unique} != {expected['unique']}"
    )
    assert max_size == expected["max_size"], (
        f"{group_by} max group {max_size} != {expected['max_size']}"
    )
    return {"unique": unique, "max_size": max_size}


def join_embeddings_by_clip_id(
    frame: pd.DataFrame,
    embeddings_path: Path,
    emb_key: str = "embeddings",
) -> np.ndarray:
    """Join NPZ embeddings to frame on clip_id (never positional index).

    Args:
        frame: Labelled rows in desired order.
        embeddings_path: NPZ with clip_id + embedding array.
        emb_key: Array name in the NPZ.

    Returns:
        np.ndarray: float32 matrix aligned to ``frame``.

    Raises:
        ValueError: If any clip_id is missing or row counts diverge.
    """
    bundle = np.load(embeddings_path, allow_pickle=True)
    if "clip_id" not in bundle.files or emb_key not in bundle.files:
        raise ValueError(
            f"{embeddings_path} must contain clip_id and {emb_key}; "
            f"has {bundle.files}"
        )
    emb_ids = [str(x) for x in bundle["clip_id"].tolist()]
    matrix = np.asarray(bundle[emb_key], dtype=np.float32)
    if len(emb_ids) != len(matrix):
        raise ValueError("clip_id / embeddings length mismatch in NPZ")
    lookup = {cid: matrix[i] for i, cid in enumerate(emb_ids)}

    vectors: list[np.ndarray] = []
    for cid in frame["clip_id"].astype(str):
        if cid not in lookup:
            raise ValueError(f"Missing embedding for clip_id={cid}")
        vectors.append(lookup[cid])
    X = np.vstack(vectors)
    if len(X) != len(frame):
        raise ValueError("Joined embedding count != frame length")
    if np.isnan(X).any():
        raise ValueError("Embeddings contain NaN")
    return X


def evaluate_embedding_grid(
    *,
    X: np.ndarray,
    y: np.ndarray,
    frame: pd.DataFrame,
    modality: str,
    encoder_id: str,
    embeddings_path: Path | None,
    manifest_path: Path,
    output_path: Path,
    folds: int,
    repeats: int,
    seed: int,
    group_by: str | None,
    only: tuple[str, ...] | None,
    extra_meta: dict[str, Any] | None = None,
) -> dict:
    """Run the audio-matched head grid on frozen text embeddings.

    Args:
        X: Embedding matrix aligned with frame.
        y: Integer labels.
        frame: Manifest rows.
        modality: Modality tag for JSON.
        encoder_id: HF / sentence-transformers model id.
        embeddings_path: Source NPZ path (optional).
        manifest_path: Manifest path for provenance.
        output_path: Destination JSON.
        folds / repeats / seed: CV protocol.
        group_by: Optional group column.
        only: Optional candidate filter.
        extra_meta: Extra top-level JSON fields.

    Returns:
        dict: Full results payload written to disk.
    """
    import train_valence_head_cv as audio_cv

    counts = {ID_TO_LABEL[i]: int(c) for i, c in enumerate(np.bincount(y))}
    assert counts == EXPECTED_COUNTS, counts
    assert len(y) == EXPECTED_N, len(y)

    groups = None
    group_stats = None
    if group_by:
        group_stats = assert_group_stats(frame, group_by)
        groups = audio_cv.group_labels_from_frame(frame, group_by)

    cv_name = (
        f"StratifiedGroupKFold(group_by={group_by})"
        if group_by
        else "RepeatedStratifiedKFold"
    )
    print(f"[data] n={len(y)} dim={X.shape[1]} modality={modality} classes={counts}")
    print(
        f"[cv]   {cv_name}  {repeats} repeats x {folds} folds = "
        f"{repeats * folds} fits/config"
    )
    if group_stats:
        print(f"[groups] {group_by}: {group_stats}")

    results: dict = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "modality": modality,
        "encoder_id": encoder_id,
        "embeddings_path": str(embeddings_path) if embeddings_path else None,
        "manifest_path": str(manifest_path),
        "embedding_dim": int(X.shape[1]),
        "n_labelled": int(len(y)),
        "class_counts": counts,
        "library_versions": library_versions(),
        "protocol": {
            "folds": folds,
            "repeats": repeats,
            "seed": seed,
            "cv": cv_name,
            "group_by": group_by,
            "group_stats": group_stats,
            "selection": "none - full grid reported, selection is a manual decision",
            "only": list(only) if only else None,
        },
        "majority_baseline": audio_cv.majority_baseline(y),
        "grid": {},
    }
    if extra_meta:
        results.update(extra_meta)

    for balanced in (True, False):
        regime = "balanced" if balanced else "empirical"
        results["grid"][regime] = {}
        candidates = audio_cv.build_candidates(balanced, seed)
        if only:
            missing = sorted(set(only) - set(candidates))
            if missing:
                raise ValueError(
                    f"Unknown candidate(s) {missing}; known={sorted(candidates)}"
                )
            candidates = {k: candidates[k] for k in only if k in candidates}
        for name, candidate in candidates.items():
            per_repeat, oof = audio_cv.run_repeated_cv(
                X, y, candidate, folds, repeats, seed, groups=groups
            )
            summary = audio_cv.summarise(per_repeat)
            summary["per_emotion_negative_recall"] = (
                audio_cv.per_emotion_negative_recall(y, oof, frame)
            )
            results["grid"][regime][name] = summary
            neg = summary["per_class"]["negative"]["recall"]
            print(
                f"[{regime:>9}] {name:<16} "
                f"acc={summary['accuracy']['mean']:.3f}±{summary['accuracy']['sd']:.3f}  "
                f"macF1={summary['macro_f1']['mean']:.3f}±{summary['macro_f1']['sd']:.3f}  "
                f"negR={neg['mean']:.3f}±{neg['sd']:.3f}"
            )
            mac = summary["macro_f1"]["mean"]
            if mac > 0.90:
                print(
                    f"[sanity] WARNING {regime}/{name} macro-F1={mac:.3f} > 0.90 "
                    "— possible leakage or label bug"
                )
            if mac <= 0.249:
                print(
                    f"[sanity] WARNING {regime}/{name} macro-F1={mac:.3f} ≤ majority "
                    "— pipeline may be broken"
                )
        print()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    import json

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[wrote] {output_path}")
    return results
