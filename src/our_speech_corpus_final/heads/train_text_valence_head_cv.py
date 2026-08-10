#!/usr/bin/env python3
"""
Authors:
    Steven Liem (steven.liem@sydney.edu.au)

Text-only valence baseline on Gemini Live transcripts (response_text).

Motivation
----------
WavLM encodes spoken lexical content; valence recovery often rides on implicit
linguistics. This harness trains the same style of classical heads on TF-IDF
features of the spoken transcript only (no audio). If text matches WavLM under
the matched 5x10 group-CV protocol, the instrument is largely reading words.

Protocol mirrors train_valence_head_cv.py:
  - 5 folds x 10 repeats, seed 42
  - balanced / empirical regimes
  - optional StratifiedGroupKFold(--group-by question|response_text)
  - target = ground_truth_label

Usage:
    uv run python src/our_speech_corpus_final/heads/train_text_valence_head_cv.py \\
        --group-by question \\
        --compare-wavlm data/our_speech_corpus_wavlm/reports/valence_head_cv_wavlm_group_question.json
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
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVC, LinearSVC

from config import bootstrap_imports, extend_our_speech_corpus_paths

bootstrap_imports(__file__)
extend_our_speech_corpus_paths()
from config import (  # noqa: E402
    SPLIT_MANIFEST_CSV,
    SPLIT_SEED,
    TEXT_CV_RESULTS_JSON,
    VALENCE_CLASSES,
    WAVLM_CV_GROUP_QUESTION_JSON,
)

# Reuse CV plumbing from the audio valence harness.
import train_valence_head_cv as audio_cv  # noqa: E402

LABEL_TO_ID = {label: i for i, label in enumerate(VALENCE_CLASSES)}
ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}

# Map text candidate names onto WavLM grid names for side-by-side Δ.
TEXT_TO_WAVLM_NAME = {
    "svd32_logistic": "pca32_logistic",
    "l2_logistic": "l2_logistic",
    "linear_svc": "linear_svc",
    "rbf_svc": "rbf_svc",
    "hist_gbm": "hist_gbm",
}

# Decision thresholds (macro-F1 gap text vs WavLM), from the phase plan.
MATCH_TOL = 0.02
AUDIO_GAP = 0.05


def load_labelled_texts(
    manifest_path: Path,
    text_col: str = "response_text",
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load labelled rows and spoken transcripts.

    Args:
        manifest_path: Split / unified manifest CSV.
        text_col: Transcript column (default response_text).

    Returns:
        tuple: (texts, y, frame) with texts as 1-d object array of strings.

    Raises:
        ValueError: If the text column is missing, empty, or labels are invalid.
    """
    manifest = pd.read_csv(manifest_path)
    labelled = manifest[manifest["is_labelled"]].copy()
    if text_col not in labelled.columns:
        raise ValueError(f"Manifest missing {text_col!r}")

    texts: list[str] = []
    labels: list[int] = []
    keep: list[int] = []
    for idx, row in labelled.iterrows():
        gt = str(row["ground_truth_label"]).strip().lower()
        if gt not in LABEL_TO_ID:
            raise ValueError(f"Unexpected ground_truth_label={gt!r} for {row['clip_id']}")
        text = audio_cv.normalise_text(row[text_col])
        if not text:
            raise ValueError(f"Empty {text_col} for clip_id={row['clip_id']}")
        texts.append(text)
        labels.append(LABEL_TO_ID[gt])
        keep.append(idx)

    if not texts:
        raise ValueError("No labelled rows with usable transcripts")

    frame = labelled.loc[keep].reset_index(drop=True)
    return (
        np.asarray(texts, dtype=object),
        np.asarray(labels, dtype=np.int64),
        frame,
    )


def make_tfidf() -> FeatureUnion:
    """Word + char_wb TF-IDF union (fit inside each CV fold via Pipeline).

    Returns:
        FeatureUnion: Parallel word (1–2) and char_wb (3–5) TF-IDF vectorizers.
    """
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=20_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=20_000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )


def build_text_candidates(balanced: bool, seed: int) -> dict[str, dict]:
    """Build TF-IDF classical heads at one explicit operating point.

    Sparse-friendly linear models take TF-IDF directly. Dense models
    (RBF-SVM, HistGBM) sit on TruncatedSVD(32), the PCA analogue for sparse
    text features.

    Args:
        balanced: If True, class_weight="balanced"; else empirical.
        seed: Random state for SVD / solvers.

    Returns:
        dict[str, dict]: name -> {"model": pipeline, "oversample": False}.
    """
    class_weight = "balanced" if balanced else None
    return {
        "svd32_logistic": {
            "model": Pipeline(
                [
                    ("tfidf", make_tfidf()),
                    ("svd", TruncatedSVD(n_components=32, random_state=seed)),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=2000,
                            class_weight=class_weight,
                            random_state=seed,
                        ),
                    ),
                ]
            ),
            "oversample": False,
        },
        "l2_logistic": {
            "model": Pipeline(
                [
                    ("tfidf", make_tfidf()),
                    (
                        "clf",
                        LogisticRegression(
                            C=0.1,
                            max_iter=2000,
                            class_weight=class_weight,
                            random_state=seed,
                            solver="saga",
                        ),
                    ),
                ]
            ),
            "oversample": False,
        },
        "linear_svc": {
            "model": Pipeline(
                [
                    ("tfidf", make_tfidf()),
                    (
                        "clf",
                        LinearSVC(
                            C=0.1,
                            class_weight=class_weight,
                            dual="auto",
                            max_iter=5000,
                            random_state=seed,
                        ),
                    ),
                ]
            ),
            "oversample": False,
        },
        "rbf_svc": {
            "model": Pipeline(
                [
                    ("tfidf", make_tfidf()),
                    ("svd", TruncatedSVD(n_components=32, random_state=seed)),
                    (
                        "clf",
                        SVC(
                            kernel="rbf",
                            C=1.0,
                            gamma="scale",
                            class_weight=class_weight,
                            random_state=seed,
                        ),
                    ),
                ]
            ),
            "oversample": False,
        },
        "hist_gbm": {
            "model": Pipeline(
                [
                    ("tfidf", make_tfidf()),
                    ("svd", TruncatedSVD(n_components=32, random_state=seed)),
                    (
                        "clf",
                        HistGradientBoostingClassifier(
                            max_depth=3,
                            learning_rate=0.05,
                            max_iter=200,
                            class_weight=class_weight,
                            random_state=seed,
                        ),
                    ),
                ]
            ),
            "oversample": False,
        },
    }


def _metric_pair(summary: dict) -> tuple[float, float, float, float]:
    """Extract macro-F1 / neg-recall means and sds from a CV summary cell.

    Args:
        summary: One candidate's summarise() payload.

    Returns:
        tuple: (mac_f1_mean, mac_f1_sd, neg_r_mean, neg_r_sd).
    """
    return (
        float(summary["macro_f1"]["mean"]),
        float(summary["macro_f1"]["sd"]),
        float(summary["per_class"]["negative"]["recall"]["mean"]),
        float(summary["per_class"]["negative"]["recall"]["sd"]),
    )


def compare_to_wavlm(
    text_results: dict,
    wavlm_path: Path,
    headline_candidate: str = "rbf_svc",
    headline_regime: str = "balanced",
) -> dict:
    """Side-by-side text vs WavLM metrics with an explicit match/gap verdict.

    Args:
        text_results: Full text-only CV payload.
        wavlm_path: Existing WavLM CV JSON (typically group=question).
        headline_candidate: Text candidate used for the primary verdict.
        headline_regime: Operating-point regime for the primary verdict.

    Returns:
        dict: Comparison table rows plus verdict metadata.
    """
    with wavlm_path.open(encoding="utf-8") as f:
        wavlm = json.load(f)

    rows: list[dict] = []
    for regime in ("balanced", "empirical"):
        text_grid = text_results["grid"].get(regime, {})
        wavlm_grid = wavlm.get("grid", {}).get(regime, {})
        for text_name, wavlm_name in TEXT_TO_WAVLM_NAME.items():
            if text_name not in text_grid or wavlm_name not in wavlm_grid:
                continue
            t_mac, t_mac_sd, t_neg, t_neg_sd = _metric_pair(text_grid[text_name])
            w_mac, w_mac_sd, w_neg, w_neg_sd = _metric_pair(wavlm_grid[wavlm_name])
            rows.append(
                {
                    "regime": regime,
                    "text_candidate": text_name,
                    "wavlm_candidate": wavlm_name,
                    "text_macro_f1": t_mac,
                    "text_macro_f1_sd": t_mac_sd,
                    "wavlm_macro_f1": w_mac,
                    "wavlm_macro_f1_sd": w_mac_sd,
                    "delta_macro_f1_text_minus_wavlm": t_mac - w_mac,
                    "text_neg_recall": t_neg,
                    "text_neg_recall_sd": t_neg_sd,
                    "wavlm_neg_recall": w_neg,
                    "wavlm_neg_recall_sd": w_neg_sd,
                    "delta_neg_recall_text_minus_wavlm": t_neg - w_neg,
                }
            )

    headline = next(
        (
            r
            for r in rows
            if r["regime"] == headline_regime
            and r["text_candidate"] == headline_candidate
        ),
        None,
    )
    if headline is None:
        verdict = {
            "status": "unavailable",
            "reason": (
                f"Missing headline cell {headline_regime}/{headline_candidate} "
                "in comparison rows"
            ),
        }
    else:
        delta = headline["delta_macro_f1_text_minus_wavlm"]
        abs_delta = abs(delta)
        if abs_delta <= MATCH_TOL:
            status = "match"
            interpretation = (
                "Text-only macro-F1 matches WavLM within "
                f"±{MATCH_TOL:.2f}. Instrument is largely reading words "
                "(transcript-sentiment proxy); de-emphasise prosodic claims."
            )
        elif delta <= -AUDIO_GAP:
            status = "audio_contribution"
            interpretation = (
                f"WavLM leads text by {abs_delta:.3f} macro-F1 (≥{AUDIO_GAP:.2f}). "
                "Genuine audio contribution beyond lexical content."
            )
        elif delta >= AUDIO_GAP:
            status = "text_leads"
            interpretation = (
                f"Text leads WavLM by {delta:.3f} macro-F1. Labels may be "
                "content-driven; WavLM is not adding useful paralinguistic signal."
            )
        else:
            status = "inconclusive"
            interpretation = (
                f"|Δ macro-F1|={abs_delta:.3f} sits between match (±{MATCH_TOL:.2f}) "
                f"and clear-gap ({AUDIO_GAP:.2f}) thresholds — report both numbers."
            )
        verdict = {
            "status": status,
            "headline_regime": headline_regime,
            "headline_text_candidate": headline_candidate,
            "headline_wavlm_candidate": headline["wavlm_candidate"],
            "text_macro_f1": headline["text_macro_f1"],
            "wavlm_macro_f1": headline["wavlm_macro_f1"],
            "delta_macro_f1_text_minus_wavlm": delta,
            "text_neg_recall": headline["text_neg_recall"],
            "wavlm_neg_recall": headline["wavlm_neg_recall"],
            "match_tolerance": MATCH_TOL,
            "audio_gap_threshold": AUDIO_GAP,
            "interpretation": interpretation,
        }

    comparison = {
        "wavlm_path": str(wavlm_path),
        "text_majority_baseline": text_results.get("majority_baseline"),
        "wavlm_majority_baseline": wavlm.get("majority_baseline"),
        "rows": rows,
        "verdict": verdict,
    }
    return comparison


def print_comparison(comparison: dict) -> None:
    """Print a compact text-vs-WavLM table and the verdict.

    Args:
        comparison: Payload from compare_to_wavlm.

    Returns:
        None
    """
    print("\n[compare] text-only vs WavLM")
    print(
        f"{'regime':<10} {'text':<16} {'wavlm':<16} "
        f"{'t_mac':>7} {'w_mac':>7} {'d_mac':>7} "
        f"{'t_negR':>7} {'w_negR':>7}"
    )
    for row in comparison["rows"]:
        print(
            f"{row['regime']:<10} {row['text_candidate']:<16} "
            f"{row['wavlm_candidate']:<16} "
            f"{row['text_macro_f1']:7.3f} {row['wavlm_macro_f1']:7.3f} "
            f"{row['delta_macro_f1_text_minus_wavlm']:7.3f} "
            f"{row['text_neg_recall']:7.3f} {row['wavlm_neg_recall']:7.3f}"
        )
    verdict = comparison["verdict"]
    print(f"\n[verdict] status={verdict.get('status')}")
    if "interpretation" in verdict:
        print(f"          {verdict['interpretation']}")
    elif "reason" in verdict:
        print(f"          {verdict['reason']}")


def evaluate_all(
    manifest_path: Path = SPLIT_MANIFEST_CSV,
    output_path: Path = TEXT_CV_RESULTS_JSON,
    folds: int = 5,
    repeats: int = 10,
    seed: int = SPLIT_SEED,
    only: tuple[str, ...] | None = None,
    group_by: str | None = None,
    text_col: str = "response_text",
    compare_wavlm: Path | None = None,
) -> dict:
    """Run the text-only grid and optionally compare to WavLM JSON.

    Args:
        manifest_path: Manifest with labels and transcripts.
        output_path: Destination JSON.
        folds: Stratified folds per repeat.
        repeats: CV repetitions.
        seed: Random state.
        only: Optional candidate-name filter.
        group_by: Optional StratifiedGroupKFold column.
        text_col: Transcript column.
        compare_wavlm: If set, attach and print WavLM comparison.

    Returns:
        dict: Full results payload written to disk.
    """
    texts, y, frame = load_labelled_texts(manifest_path, text_col=text_col)
    counts = {ID_TO_LABEL[i]: int(c) for i, c in enumerate(np.bincount(y))}
    groups = audio_cv.group_labels_from_frame(frame, group_by) if group_by else None
    cv_name = (
        f"StratifiedGroupKFold(group_by={group_by})"
        if group_by
        else "RepeatedStratifiedKFold"
    )
    print(f"[data] n={len(y)} modality=text/{text_col} classes={counts}")
    print(
        f"[cv]   {cv_name}  {repeats} repeats x {folds} folds = "
        f"{repeats * folds} fits/config"
    )
    print("[note] development estimates; matched protocol vs WavLM audio grid\n")

    results: dict = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "modality": "text_only",
        "text_column": text_col,
        "feature": "tfidf_word_1_2_char_wb_3_5",
        "manifest_path": str(manifest_path),
        "n_labelled": int(len(y)),
        "class_counts": counts,
        "protocol": {
            "folds": folds,
            "repeats": repeats,
            "seed": seed,
            "cv": cv_name,
            "group_by": group_by,
            "selection": "none - full grid reported, selection is a manual decision",
            "only": list(only) if only else None,
        },
        "majority_baseline": audio_cv.majority_baseline(y),
        "grid": {},
    }

    for balanced in (True, False):
        regime = "balanced" if balanced else "empirical"
        results["grid"][regime] = {}
        candidates = build_text_candidates(balanced, seed)
        if only:
            missing = sorted(set(only) - set(candidates))
            if missing:
                raise ValueError(
                    f"Unknown candidate(s) {missing}; known={sorted(candidates)}"
                )
            candidates = {k: candidates[k] for k in only if k in candidates}
        for name, candidate in candidates.items():
            per_repeat, oof = audio_cv.run_repeated_cv(
                texts, y, candidate, folds, repeats, seed, groups=groups
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
        print()

    if compare_wavlm is not None:
        if not compare_wavlm.is_file():
            raise FileNotFoundError(f"WavLM results not found: {compare_wavlm}")
        comparison = compare_to_wavlm(results, compare_wavlm)
        results["comparison_to_wavlm"] = comparison
        print_comparison(comparison)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[wrote] {output_path}")
    return results


def main() -> None:
    """CLI entrypoint for the text-only valence CV baseline.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description="Repeated CV text-only valence baseline on response_text"
    )
    parser.add_argument("--manifest", type=Path, default=SPLIT_MANIFEST_CSV)
    parser.add_argument("--output", type=Path, default=TEXT_CV_RESULTS_JSON)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="CV repetitions (default 10). Not 10-fold CV.",
    )
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    parser.add_argument(
        "--text-col",
        default="response_text",
        help="Transcript column (default response_text — spoken AI text).",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Restrict to these candidate names",
    )
    parser.add_argument(
        "--group-by",
        choices=("response_text", "question"),
        default=None,
        help=(
            "StratifiedGroupKFold on normalised text. "
            "question = headline WavLM protocol."
        ),
    )
    parser.add_argument(
        "--compare-wavlm",
        type=Path,
        default=None,
        nargs="?",
        const=WAVLM_CV_GROUP_QUESTION_JSON,
        help=(
            "Compare to a WavLM CV JSON. Pass flag alone to use the default "
            "group=question WavLM report path."
        ),
    )
    args = parser.parse_args()

    if not args.manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    evaluate_all(
        manifest_path=args.manifest,
        output_path=args.output,
        folds=args.folds,
        repeats=args.repeats,
        seed=args.seed,
        only=tuple(args.only) if args.only else None,
        group_by=args.group_by,
        text_col=args.text_col,
        compare_wavlm=args.compare_wavlm,
    )


if __name__ == "__main__":
    main()
