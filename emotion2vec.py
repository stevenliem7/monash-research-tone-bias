#!/usr/bin/env python3
"""
Authors:
    Steven Liem (steven.liem@sydney.edu.au)

Perform SER classification using emotion2vec_plus_large on all 8 corpora. Each cleaned corpus under corpora_cleaned/ is evaluated independently.
No training or cross-validation: the pretrained emotion2vec_plus_large classifier is used off-the-shelf

The full pipeline is as follows:
  1. Discover WAVs in each cleaned corpus directory
  2. Parse temporary ground-truth valence from filenames (optional CSV override)
  3. Run utterance-level emotion2vec_plus_large inference (cached per file)
  4. Map predictions to positive/neutral/negative; exclude other/surprised/unknown
  5. Write valence predictions, metrics, and confusion-matrix plots

Usage:
    uv run python emotion2vec.py --dry-run
    uv run python emotion2vec.py
    uv run python emotion2vec.py --corpus emovoice_cleaned --limit 50
    uv run python emotion2vec.py --from-cache --workers 8
    uv run python emotion2vec.py --labels-csv path/to/our_speech_labels.csv

Optional labels CSV columns:
    filename,ground_truth_valence

References:
    https://huggingface.co/emotion2vec/emotion2vec_plus_large
    https://github.com/ddlBoJack/emotion2vec
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

WORKSPACE = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parent
CORPORA_ROOT = WORKSPACE / "corpora_cleaned"  # Directory of the cleaned corpora
RESULTS_ROOT = WORKSPACE / "results" / "emotion2vec"  # Directory to save the results
CACHE_ROOT = RESULTS_ROOT / "cache"  # Directory to save the cached predictions
DIAGRAMS_ROOT = RESULTS_ROOT / "result_diagrams"  # Directory to save the confusion matrices

MODEL_ID = "iic/emotion2vec_plus_large"
VALENCE_CLASSES = ("positive", "neutral", "negative")
EXCLUDED_PREDICTIONS = frozenset({"other", "surprised", "unknown"})

PRED_TO_VALENCE = {
    "happy": "positive",
    "neutral": "neutral",
    "angry": "negative",
    "disgusted": "negative",
    "fearful": "negative",
    "sad": "negative",
    "other": None,
    "surprised": None,
    "unknown": None,
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


def pick_device(explicit: str | None = None) -> str:
    """Choose CUDA if available, otherwise CPU.

    Args:
        explicit: Optional device string such as ``cuda:0`` or ``cpu``. If
            omitted or None, CUDA is used when available.

    Returns:
        str: Device identifier for FunASR ``AutoModel``.
    """
    if explicit:
        return explicit
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    return "cpu"


def parse_ground_truth_valence(path: Path) -> str | None:
    """Parse valence from cleaned filename stem: ``..._{emotion}_{valence}``.

    Args:
        path: Path to a cleaned corpus WAV file.

    Returns:
        str | None: One of positive/neutral/negative, or None if the stem
        does not end with a recognised valence label.
    """
    parts = path.stem.split("_")
    if len(parts) < 2:
        return None
    valence = parts[-1].strip().lower()
    return valence if valence in VALENCE_CLASSES else None


def load_label_overrides(path: Path | None) -> dict[str, str]:
    """Load optional filename --> ground_truth_valence overrides from CSV.

    Args:
        path: Optional CSV path with columns ``filename`` and
            ``ground_truth_valence``. Missing path returns an empty mapping.

    Returns:
        dict[str, str]: Lookup keyed by filename, basename, and stem.

    Raises:
        ValueError: If the CSV exists but is missing required columns.
    """
    if path is None or not path.exists():
        return {}
    df = pd.read_csv(path)
    required = {"filename", "ground_truth_valence"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"labels CSV missing columns: {sorted(missing)}")

    overrides: dict[str, str] = {}
    for _, row in df.iterrows():
        name = str(row["filename"]).strip()
        valence = str(row["ground_truth_valence"]).strip().lower()
        if valence not in VALENCE_CLASSES:
            continue
        overrides[name] = valence
        overrides[Path(name).name] = valence
        overrides[Path(name).stem] = valence
    return overrides


def resolve_ground_truth(path: Path, overrides: dict[str, str]) -> str | None:
    """Resolve ground-truth valence from override CSV or filename.

    Args:
        path: Path to a cleaned corpus WAV file.
        overrides: Mapping from ``load_label_overrides``.

    Returns:
        str | None: Ground-truth valence, or None if unresolved.
    """
    for key in (path.name, path.stem, str(path)):
        if key in overrides:
            return overrides[key]
    return parse_ground_truth_valence(path)


def list_corpus_wavs(corpus_dir: Path) -> list[Path]:
    """List WAV files in a cleaned corpus directory.

    Args:
        corpus_dir: Directory containing cleaned ``*.wav`` files.

    Returns:
        list[Path]: Sorted WAV paths.
    """
    return sorted(corpus_dir.glob("*.wav"))


def cache_path_for(corpus: str, wav: Path) -> Path:
    """Return JSON cache path for one WAV prediction.

    Args:
        corpus: Corpus directory name (e.g. ``emovoice_cleaned``).
        wav: Path to the source WAV file.

    Returns:
        Path: Cache JSON path under ``CACHE_ROOT``.
    """
    return CACHE_ROOT / corpus / f"{wav.stem}.json"


def load_cached(cache_path: Path) -> dict | None:
    """Load a cached prediction if present.

    Args:
        cache_path: Path to a per-file prediction JSON cache.

    Returns:
        dict | None: Cached payload, or None if the file does not exist.
    """
    if not cache_path.exists():
        return None
    with cache_path.open(encoding="utf-8") as file:
        return json.load(file)


def save_cached(cache_path: Path, payload: dict) -> None:
    """Write one prediction to the resume cache.

    Args:
        cache_path: Destination JSON path.
        payload: Prediction dict to serialise.

    Returns:
        None
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file)


def normalise_emotion_label(label: object) -> str:
    """Normalise FunASR labels (incl. bilingual ``开心/happy``) to English.

    Args:
        label: Raw model label (string, list/tuple of labels, or other).

    Returns:
        str: Lowercase English emotion name, or ``unknown``.
    """
    if isinstance(label, (list, tuple)) and label:
        label = label[0]
    text = str(label).strip().lower()
    if text.startswith("emo_"):
        text = text[4:]
    if "/" in text:
        text = text.split("/")[-1].strip()
    return text or "unknown"


def prediction_from_scores(score_map: dict[str, float], native_label: str | None = None) -> dict:
    """Map emotion scores to native label, valence, and exclusion flag.

    Args:
        score_map: Emotion label --> score (keys may be bilingual).
        native_label: Fallback top label if ``score_map`` is empty.

    Returns:
        dict: Keys ``native_label``, ``native_score``, ``pred_valence``,
        ``excluded``, and normalised English ``scores``.
    """
    cleaned: dict[str, float] = {}
    for label, score in (score_map or {}).items():
        key = normalise_emotion_label(label)
        cleaned[key] = max(float(score), cleaned.get(key, float("-inf")))
    if cleaned:
        top_label = max(cleaned, key=cleaned.get)
        top_score = float(cleaned[top_label])
    else:
        top_label = normalise_emotion_label(native_label or "unknown")
        top_score = 1.0
        cleaned = {top_label: top_score}
    valence = PRED_TO_VALENCE.get(top_label)
    return {
        "native_label": top_label,
        "native_score": top_score,
        "pred_valence": valence,
        "excluded": top_label in EXCLUDED_PREDICTIONS or valence is None,
        "scores": cleaned,
    }


def remap_cached_prediction(pred: dict) -> dict:
    """Recompute valence fields from cached scores (ignore stale labels).

    Args:
        pred: Cached prediction dict with a ``scores`` mapping.

    Returns:
        dict: Remapped prediction from ``prediction_from_scores``.
    """
    scores = pred.get("scores") or {}
    if isinstance(scores, str):
        scores = json.loads(scores)
    return prediction_from_scores(scores, native_label=pred.get("native_label"))


def parse_generate_result(result: object) -> tuple[str, float, dict[str, float]]:
    """Parse FunASR ``generate()`` output into top label, score, and score map.

    Args:
        result: Raw FunASR generate return value (dict or list of dicts).

    Returns:
        tuple[str, float, dict[str, float]]: Top English label, its score,
        and the full label-->score map.

    Raises:
        TypeError: If the result shape is not a dict (or list of dicts).
    """
    item = result[0] if isinstance(result, list) and result else result
    if not isinstance(item, dict):
        raise TypeError(f"Unexpected emotion2vec result type: {type(result)}")

    labels = item.get("labels") or item.get("label") or []
    scores = item.get("scores") or item.get("score") or []
    if isinstance(labels, str):
        labels = [labels]
    if isinstance(scores, (int, float)):
        scores = [float(scores)]

    score_map: dict[str, float] = {}
    for label, score in zip(labels, scores):
        score_map[normalise_emotion_label(label)] = float(score)

    if score_map:
        top_label = max(score_map, key=score_map.get)
        return top_label, score_map[top_label], score_map

    pred = item.get("preds") or item.get("text") or labels
    top_label = normalise_emotion_label(pred)
    return top_label, 1.0, {top_label: 1.0}


def load_model(device: str):
    """Load emotion2vec_plus_large via FunASR.

    Args:
        device: Device string such as ``cuda:0`` or ``cpu``.

    Returns:
        FunASR AutoModel instance for utterance-level emotion classification.
    """
    from funasr import AutoModel

    hub = os.environ.get("EMOTION2VEC_HUB", "hf")
    print(f"[emotion2vec] loading {MODEL_ID} on {device} (hub={hub})")
    return AutoModel(model=MODEL_ID, hub=hub, device=device, disable_update=True)


def predict_one(model, wav: Path) -> dict:
    """Run utterance-level emotion2vec inference on one WAV.

    Args:
        model: Loaded FunASR AutoModel.
        wav: Path to the WAV file.

    Returns:
        dict: Prediction payload from ``prediction_from_scores``.
    """
    result = model.generate(
        input=str(wav),
        granularity="utterance",
        extract_embedding=False,
    )
    native_label, native_score, score_map = parse_generate_result(result)
    return prediction_from_scores(score_map, native_label=native_label)


def evaluate_corpus(
    corpus: str,
    model,
    overrides: dict[str, str],
    limit: int | None,
    workers: int = 1,
    from_cache: bool = False,
) -> pd.DataFrame:
    """Evaluate one cleaned corpus (from cache and/or model inference).

    Args:
        corpus: Corpus directory name under ``CORPORA_ROOT``.
        model: Loaded FunASR model, or None when ``from_cache`` is True.
        overrides: Optional ground-truth valence overrides.
        limit: Optional max number of WAVs to evaluate.
        workers: Thread workers for cache I/O/remap. Model ``generate`` stays
            serialised under a lock when workers > 1.
        from_cache: If True, require cache hits and do not run inference.

    Returns:
        pd.DataFrame: Per-file predictions with ground truth and exclusion flags.
    """
    corpus_dir = CORPORA_ROOT / corpus
    if not corpus_dir.is_dir():
        print(f"[emotion2vec] skip missing corpus: {corpus_dir}")
        return pd.DataFrame()

    wavs = list_corpus_wavs(corpus_dir)
    if limit is not None:
        wavs = wavs[:limit]
    print(f"[emotion2vec] {corpus}: {len(wavs)} wavs")
    model_lock = threading.Lock()

    def process_one(wav: Path) -> dict:
        """Load/remap cache or run inference for one WAV.

        Args:
            wav: Path to a corpus WAV file.

        Returns:
            dict: One prediction row for the results DataFrame.
        """
        cache_path = cache_path_for(corpus, wav)
        cached = load_cached(cache_path)
        if cached is not None:
            pred = remap_cached_prediction(cached)
            save_cached(cache_path, pred)  # rewrite fixed labels
        elif from_cache:
            raise FileNotFoundError(f"Missing cache for {wav.name}; run without --from-cache first")
        else:
            with model_lock:
                pred = predict_one(model, wav)
            save_cached(cache_path, pred)

        truth = resolve_ground_truth(wav, overrides)
        return {
            "corpus": corpus,
            "filename": wav.name,
            "audio_path": str(wav),
            "ground_truth_valence": truth,
            "native_label": pred["native_label"],
            "native_score": pred["native_score"],
            "pred_valence": pred["pred_valence"],
            "excluded": bool(pred["excluded"]),
            "scores_json": json.dumps(pred.get("scores") or {}),
        }

    rows: list[dict] = []
    if workers <= 1:
        for index, wav in enumerate(wavs, start=1):
            rows.append(process_one(wav))
            if index % 50 == 0 or index == len(wavs):
                print(f"[emotion2vec] {corpus}: {index}/{len(wavs)}")
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(process_one, wav): wav for wav in wavs}
            done = 0
            for future in as_completed(futures):
                rows.append(future.result())
                done += 1
                if done % 50 == 0 or done == len(wavs):
                    print(f"[emotion2vec] {corpus}: {done}/{len(wavs)}")
        rows.sort(key=lambda r: r["filename"])

    return pd.DataFrame(rows)


def compute_metrics(df: pd.DataFrame) -> dict:
    """Compute valence metrics after excluding ambiguous predictions.

    Args:
        df: Per-file predictions DataFrame from ``evaluate_corpus``.

    Returns:
        dict: Counts, accuracy/F1, confusion matrices, and classification report.
    """
    total = len(df)
    with_truth = df[df["ground_truth_valence"].isin(VALENCE_CLASSES)].copy()
    evaluated = with_truth[
        ~with_truth["excluded"] & with_truth["pred_valence"].isin(VALENCE_CLASSES)
    ].copy()
    excluded = with_truth[
        with_truth["excluded"] | ~with_truth["pred_valence"].isin(VALENCE_CLASSES)
    ]

    y_true = evaluated["ground_truth_valence"].tolist()
    y_pred = evaluated["pred_valence"].tolist()
    labels = list(VALENCE_CLASSES)

    if y_true:
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_norm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, zero_division=0
        )
        report = classification_report(
            y_true, y_pred, labels=labels, digits=4, zero_division=0
        )
        return {
            "n_total": int(total),
            "n_with_truth": int(len(with_truth)),
            "n_evaluated": int(len(evaluated)),
            "n_excluded": int(len(excluded)),
            "coverage": float(len(evaluated) / len(with_truth)) if len(with_truth) else 0.0,
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "macro_f1": float(
                f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
            ),
            "weighted_f1": float(
                f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
            ),
            "per_class": {
                label: {
                    "precision": float(precision[i]),
                    "recall": float(recall[i]),
                    "f1": float(f1[i]),
                    "support": int(support[i]),
                }
                for i, label in enumerate(labels)
            },
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_normalised": cm_norm.tolist(),
            "classification_report": report,
            "excluded_native_counts": (
                excluded["native_label"].value_counts().to_dict() if len(excluded) else {}
            ),
        }

    return {
        "n_total": int(total),
        "n_with_truth": int(len(with_truth)),
        "n_evaluated": 0,
        "n_excluded": int(len(excluded)),
        "coverage": 0.0,
        "accuracy": None,
        "balanced_accuracy": None,
        "macro_f1": None,
        "weighted_f1": None,
        "per_class": {},
        "confusion_matrix": [],
        "confusion_matrix_normalised": [],
        "classification_report": "No evaluated rows.",
        "excluded_native_counts": (
            excluded["native_label"].value_counts().to_dict() if len(excluded) else {}
        ),
    }


def plot_confusion_matrices(corpus: str, metrics: dict) -> None:
    """Save count and normalised confusion-matrix PNGs under result_diagrams/.

    Args:
        corpus: Corpus name used in titles and filenames.
        metrics: Metrics dict from ``compute_metrics``.

    Returns:
        None
    """
    cm = np.asarray(metrics.get("confusion_matrix") or [])
    cm_norm = np.asarray(
        metrics.get("confusion_matrix_normalised")
        or metrics.get("confusion_matrix_normalised")
        or []
    )
    if cm.size == 0:
        return

    counts_dir = DIAGRAMS_ROOT / "confusion_matrix_counts"
    normalised_dir = DIAGRAMS_ROOT / "confusion_matrix_normalised"
    counts_dir.mkdir(parents=True, exist_ok=True)
    normalised_dir.mkdir(parents=True, exist_ok=True)
    labels = list(VALENCE_CLASSES)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted valence")
    ax.set_ylabel("True valence")
    ax.set_title(f"{corpus} confusion matrix (counts)")
    fig.tight_layout()
    fig.savefig(counts_dir / f"{corpus}_confusion_counts.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
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
    ax.set_xlabel("Predicted valence")
    ax.set_ylabel("True valence")
    ax.set_title(f"{corpus} confusion matrix (row-normalised)")
    fig.tight_layout()
    fig.savefig(normalised_dir / f"{corpus}_confusion_normalised.png", dpi=160)
    plt.close(fig)


def write_corpus_outputs(corpus: str, predictions: pd.DataFrame, metrics: dict) -> None:
    """Write prediction CSV, metrics JSON, report text, and plots.

    Args:
        corpus: Corpus name (output subdirectory under ``RESULTS_ROOT``).
        predictions: Per-file predictions DataFrame.
        metrics: Metrics dict from ``compute_metrics``.

    Returns:
        None
    """
    out_dir = RESULTS_ROOT / corpus
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions.to_csv(out_dir / "predictions.csv", index=False)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    (out_dir / "classification_report.txt").write_text(
        metrics.get("classification_report", ""), encoding="utf-8"
    )
    plot_confusion_matrices(corpus, metrics)


def write_summary(summary_rows: list[dict]) -> None:
    """Write aggregate summary CSV/JSON across corpora.

    Args:
        summary_rows: One summary dict per evaluated corpus.

    Returns:
        None
    """
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RESULTS_ROOT / "summary.csv", index=False)
    with (RESULTS_ROOT / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary_rows, file, indent=2)
    print(f"[emotion2vec] wrote summary: {RESULTS_ROOT / 'summary.csv'}")


def main() -> None:
    """Run independent zero-shot evaluation for each cleaned corpus.

    Returns:
        None
    """
    global CORPORA_ROOT

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--corpora-root",
        type=Path,
        default=CORPORA_ROOT,
        help="Root containing cleaned corpus directories",
    )
    parser.add_argument(
        "--corpus",
        action="append",
        default=None,
        help="Evaluate only this corpus (repeatable). Default: all known corpora.",
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=None,
        help="Optional CSV override: filename,ground_truth_valence",
    )
    parser.add_argument("--device", type=str, default=None, help="cuda:0 / cpu / auto")
    parser.add_argument("--limit", type=int, default=None, help="Max files per corpus")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run 5 files from the first available corpus only",
    )
    parser.add_argument(
        "--from-cache",
        action="store_true",
        help="Rebuild metrics from cache only (no model load)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Thread workers for cache I/O/remap (generate stays locked)",
    )

    args = parser.parse_args()

    CORPORA_ROOT = args.corpora_root

    corpora = args.corpus or list(CORPUS_DIRS)
    limit = args.limit
    if args.dry_run:
        available = [c for c in corpora if (CORPORA_ROOT / c).is_dir()]
        if not available:
            raise FileNotFoundError(f"No corpora found under {CORPORA_ROOT}")
        corpora = [available[0]]
        limit = 5 if limit is None else limit

    overrides = load_label_overrides(args.labels_csv)
    model = None
    if not args.from_cache:
        device = pick_device(None if args.device in (None, "auto") else args.device)
        model = load_model(device)

    summary_rows: list[dict] = []
    for corpus in corpora:
        predictions = evaluate_corpus(
            corpus,
            model,
            overrides,
            limit,
            workers=max(1, args.workers),
            from_cache=args.from_cache,
        )
        if predictions.empty:
            continue
        metrics = compute_metrics(predictions)
        write_corpus_outputs(corpus, predictions, metrics)
        summary_rows.append(
            {
                "corpus": corpus,
                "n_total": metrics["n_total"],
                "n_with_truth": metrics["n_with_truth"],
                "n_evaluated": metrics["n_evaluated"],
                "n_excluded": metrics["n_excluded"],
                "coverage": metrics["coverage"],
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
            }
        )
        print(
            f"[emotion2vec] {corpus}: "
            f"acc={metrics['accuracy']} "
            f"macro_f1={metrics['macro_f1']} "
            f"coverage={metrics['coverage']:.3f} "
            f"(excluded {metrics['n_excluded']})"
        )

    write_summary(summary_rows)
    print(f"[emotion2vec] done. results under {RESULTS_ROOT}")


if __name__ == "__main__":
    main()
