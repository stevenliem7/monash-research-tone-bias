"""
Shared human-GT set for cross-instrument valence eval.

Human HEET labels alone are not enough for fair emotion2vec vs EIV comparison:
emotion2vec drops unmappable preds (<unk>/surprised/other), while EIV always
emits a valence. The shared set is:

  human-labelled ∩ emotion2vec-mappable (pred_valence in {pos,neu,neg})

Every instrument must be scored on this same filename set (n=229) and counting.
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


from pathlib import Path

import pandas as pd

from config import (
    HEET_CLEAN_CSV,
    RESULTS_EMOTION2VEC,
    VALENCE_CLASSES,
    bootstrap_imports,
)

bootstrap_imports(__file__)

VALENCE = VALENCE_CLASSES

DEFAULT_HEET = HEET_CLEAN_CSV
DEFAULT_E2V_PREDS = (
    RESULTS_EMOTION2VEC / "our_speech_corpus_cleaned" / "predictions.csv"
)


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def load_human_labelled(heet_path: Path) -> pd.DataFrame:
    """Load HEET rows with a valid human valence label.

    Args:
        heet_path: Path to heet_dataset_clean.csv.

    Returns:
        pd.DataFrame: Columns filename, human_ground_truth_label.
    """
    heet = pd.read_csv(heet_path)
    heet["human_ground_truth_label"] = (
        heet["ground_truth_label"].fillna("").astype(str).str.strip().str.lower()
    )
    labelled = heet[heet["human_ground_truth_label"].isin(VALENCE)].copy()
    labelled["filename"] = labelled["audio_path"].fillna("").map(lambda p: Path(str(p)).name)
    return labelled[["filename", "human_ground_truth_label"]].drop_duplicates("filename")


def build_comparable(
    heet_path: Path,
    e2v_predictions_path: Path,
) -> tuple[pd.DataFrame, dict]:
    """Build the shared human-GT ∩ emotion2vec-mappable frame.

    Args:
        heet_path: HEET CSV with ground_truth_label.
        e2v_predictions_path: emotion2vec predictions.csv (needs pred_valence).

    Returns:
        tuple[pd.DataFrame, dict]: Comparable rows and coverage stats.
    """
    labelled = load_human_labelled(heet_path)
    e2v = pd.read_csv(e2v_predictions_path)
    if "pred_valence" not in e2v.columns:
        raise KeyError(f"Missing pred_valence in {e2v_predictions_path}")

    merged = labelled.merge(
        e2v[["filename", "pred_valence", "native_label"]].rename(
            columns={
                "pred_valence": "e2v_pred_valence",
                "native_label": "e2v_native_label",
            }
        ),
        on="filename",
        how="left",
        indicator=True,
    )
    mappable = merged["e2v_pred_valence"].isin(VALENCE)
    comparable = merged[mappable].drop(columns=["_merge"]).copy()
    excluded = merged[~mappable].copy()

    labelled_support = labelled["human_ground_truth_label"].value_counts().to_dict()
    support = comparable["human_ground_truth_label"].value_counts().to_dict()
    excl_support = excluded["human_ground_truth_label"].value_counts().to_dict()
    excl_native = (
        excluded["e2v_native_label"].fillna("<unmatched>").astype(str).value_counts().to_dict()
        if len(excluded)
        else {}
    )

    stats = {
        "n_labelled": int(len(labelled)),
        "n": int(len(comparable)),
        "n_excluded": int(len(excluded)),
        "labelled_support": labelled_support,
        "support": support,
        "excluded_support": excl_support,
        "excluded_e2v_native": excl_native,
        "neg_labelled": int(labelled_support.get("negative", 0)),
        "neg_support": int(support.get("negative", 0)),
        "filenames": set(comparable["filename"]),
    }
    return comparable, stats


def print_coverage(stats: dict) -> None:
    """Print human-GT ∩ e2v-mappable coverage.

    Args:
        stats: Coverage dictionary from build_comparable.

    Returns:
        None
    """
    print("\n[human_GT] Human-labelled ∩ emotion2vec-mappable")
    print(f"  n_labelled   = {stats['n_labelled']}")
    print(f"  n            = {stats['n']}")
    print(f"  n_excluded   = {stats['n_excluded']}")
    ls = stats["labelled_support"]
    cs = stats["support"]
    print(
        "  labelled support  "
        f"pos={ls.get('positive', 0)} neu={ls.get('neutral', 0)} neg={stats['neg_labelled']}"
    )
    print(
        "  comparable support "
        f"pos={cs.get('positive', 0)} neu={cs.get('neutral', 0)} neg={stats['neg_support']}"
    )
    if stats["n_excluded"]:
        print(f"  excluded human-GT support = {stats['excluded_support']}")
        print(f"  excluded e2v native_label = {stats['excluded_e2v_native']}")
    else:
        print("  excluded breakdown = (none)")


def metrics_from_truth_pred(truth: pd.Series, pred: pd.Series) -> tuple[dict, pd.DataFrame]:
    """Compute valence metrics + confusion matrix from aligned series.

    Args:
        truth: Human (or other) valence labels.
        pred: Model valence predictions.

    Returns:
        tuple[dict, pd.DataFrame]: Metrics dict and reindexed confusion matrix.
    """
    cm = pd.crosstab(truth, pred).reindex(index=list(VALENCE), columns=list(VALENCE), fill_value=0)
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
        "n": n,
        "neg_support": int(supports[VALENCE.index("negative")]),
    }
    return metrics, cm


def evaluate_human_gt(
    comparable: pd.DataFrame,
    predictions: pd.DataFrame,
    pred_col: str,
    instrument: str,
) -> tuple[dict, pd.DataFrame, dict]:
    """Score one instrument on the shared human-GT filename set only.

    Args:
        comparable: Frame with filename + human_ground_truth_label.
        predictions: Instrument predictions including filename and pred_col.
        pred_col: Prediction column to evaluate.
        instrument: Name used in coverage logs.

    Returns:
        tuple[dict, pd.DataFrame, dict]: Metrics, confusion matrix, coverage.
    """
    if pred_col not in predictions.columns:
        raise KeyError(f"{instrument}: missing prediction column {pred_col!r}")

    merged = comparable[["filename", "human_ground_truth_label"]].merge(
        predictions[["filename", pred_col]],
        on="filename",
        how="left",
        indicator=True,
    )
    pred_ok = merged[pred_col].isin(VALENCE)
    usable = merged[pred_ok].copy()
    dropped = merged[~pred_ok]

    coverage = {
        "instrument": instrument,
        "n": int(len(comparable)),
        "n_evaluated": int(len(usable)),
        "n_dropped": int(len(dropped)),
        "neg_support": int(
            (usable["human_ground_truth_label"] == "negative").sum() if len(usable) else 0
        ),
        "dropped_pred_values": (
            dropped[pred_col].fillna("<missing>").astype(str).value_counts().to_dict()
            if len(dropped)
            else {}
        ),
    }
    print(
        f"[{instrument}] human_GT: "
        f"n={coverage['n']} "
        f"n_evaluated={coverage['n_evaluated']} "
        f"n_dropped={coverage['n_dropped']} "
        f"neg_support={coverage['neg_support']}"
    )
    if coverage["n_dropped"]:
        print(f"  dropped {pred_col} values = {coverage['dropped_pred_values']}")
        raise RuntimeError(
            f"{instrument}: dropped {coverage['n_dropped']} human-GT "
            "rows — cross-instrument set is no longer shared"
        )

    metrics, cm = metrics_from_truth_pred(
        usable["human_ground_truth_label"], usable[pred_col]
    )
    return metrics, cm, coverage
