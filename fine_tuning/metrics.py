"""Metrics, scalar-mix diagnostics, and paired group-aware inference."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from fine_tuning.paths import ID_TO_LABEL, NEG_ID, VALENCE_CLASSES


def classification_report_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Macro-F1-centred report for 3-class valence."""
    labels = list(range(len(VALENCE_CLASSES)))
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {}
    for i, name in enumerate(VALENCE_CLASSES):
        per_class[name] = {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f[i]),
        }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "negative_f1": float(f[NEG_ID]),
        "negative_recall": float(r[NEG_ID]),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "predicted_prevalence": {
            ID_TO_LABEL[i]: float(np.mean(y_pred == i)) for i in labels
        },
    }


def scalar_weight_diagnostics(logits: np.ndarray) -> dict:
    """Entropy / KL / effective-layer diagnostics for softmax layer weights."""
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max()
    p = np.exp(z)
    p = p / p.sum()
    n = len(p)
    entropy = float(-(p * np.log(np.maximum(p, 1e-12))).sum())
    log_n = float(np.log(n))
    return {
        "weights": p.tolist(),
        "entropy": entropy,
        "normalized_entropy": float(entropy / log_n) if log_n > 0 else 0.0,
        "kl_from_uniform": float(log_n - entropy) if log_n > 0 else 0.0,
        "effective_layers": float(np.exp(entropy)),
        "max_weight": float(p.max()),
        "l1_from_uniform": float(np.abs(p - 1.0 / n).sum()),
    }


def mean_sd(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return {"mean": 0.0, "sd": 0.0, "n": 0}
    return {
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1) if len(arr) > 1 else 0.0),
        "n": int(len(arr)),
    }


def _group_indices(groups: np.ndarray) -> dict[str, np.ndarray]:
    buckets: dict[str, list[int]] = defaultdict(list)
    for i, g in enumerate(groups.tolist()):
        buckets[str(g)].append(i)
    return {k: np.asarray(v, dtype=np.int64) for k, v in buckets.items()}


def paired_group_bootstrap(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    groups: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict:
    """Cluster-bootstrap paired delta Macro-F1 and delta negative-F1."""
    rng = np.random.default_rng(seed)
    by_group = _group_indices(groups)
    keys = np.asarray(list(by_group.keys()))
    base = classification_report_dict(y_true, pred_a)
    ft = classification_report_dict(y_true, pred_b)
    point_d_macro = ft["macro_f1"] - base["macro_f1"]
    point_d_neg = ft["negative_f1"] - base["negative_f1"]

    d_macros: list[float] = []
    d_negs: list[float] = []
    for _ in range(n_boot):
        sample_keys = rng.choice(keys, size=len(keys), replace=True)
        idx = np.concatenate([by_group[k] for k in sample_keys])
        a = classification_report_dict(y_true[idx], pred_a[idx])
        b = classification_report_dict(y_true[idx], pred_b[idx])
        d_macros.append(b["macro_f1"] - a["macro_f1"])
        d_negs.append(b["negative_f1"] - a["negative_f1"])

    lo_m, hi_m = np.quantile(d_macros, [0.025, 0.975])
    lo_n, hi_n = np.quantile(d_negs, [0.025, 0.975])
    return {
        "baseline": base,
        "candidate": ft,
        "delta_macro_f1": {
            "point": float(point_d_macro),
            "ci95": [float(lo_m), float(hi_m)],
            "excludes_zero": bool(lo_m > 0 or hi_m < 0),
        },
        "delta_negative_f1": {
            "point": float(point_d_neg),
            "ci95": [float(lo_n), float(hi_n)],
            "excludes_loss_worse_than_minus_0_02": bool(lo_n > -0.02),
        },
    }


def paired_group_randomization(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    groups: np.ndarray,
    *,
    n_perm: int = 2000,
    seed: int = 42,
) -> dict:
    """Swap A/B predictions within question groups; recompute delta Macro-F1."""
    rng = np.random.default_rng(seed)
    by_group = _group_indices(groups)
    observed = (
        classification_report_dict(y_true, pred_b)["macro_f1"]
        - classification_report_dict(y_true, pred_a)["macro_f1"]
    )
    count = 0
    for _ in range(n_perm):
        a = pred_a.copy()
        b = pred_b.copy()
        for idx in by_group.values():
            if rng.random() < 0.5:
                a[idx], b[idx] = b[idx].copy(), a[idx].copy()
        delta = (
            classification_report_dict(y_true, b)["macro_f1"]
            - classification_report_dict(y_true, a)["macro_f1"]
        )
        if abs(delta) >= abs(observed):
            count += 1
    p = (count + 1) / (n_perm + 1)
    return {"observed_delta_macro_f1": float(observed), "p_value": float(p)}


def mcnemar_secondary(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """Secondary paired correctness test (accuracy-oriented, not Macro-F1)."""
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true
    b01 = int(np.sum(~correct_a & correct_b))
    b10 = int(np.sum(correct_a & ~correct_b))
    n = b01 + b10
    if n == 0:
        return {"b01": b01, "b10": b10, "p_value": 1.0, "note": "no discordant pairs"}
    # Exact two-sided binomial mid-p approximation via normal continuity correction
    # when n is large; for small n use exact binomial.
    from math import comb

    p = 0.0
    for k in range(n + 1):
        if abs(k - n / 2) >= abs(b01 - n / 2):
            p += comb(n, k) * (0.5**n)
    return {"b01": b01, "b10": b10, "p_value": float(min(1.0, p))}


def evidence_level(
    boot: dict,
    rand: dict,
    seed_deltas: list[float],
) -> str:
    """Map paired results to demonstrated / directional / not_supported."""
    d = boot["delta_macro_f1"]["point"]
    neg_ok = boot["delta_negative_f1"]["excludes_loss_worse_than_minus_0_02"]
    seeds_agree = all(x > 0 for x in seed_deltas) if seed_deltas else False
    if (
        d >= 0.02
        and boot["delta_macro_f1"]["excludes_zero"]
        and d > 0
        and rand["p_value"] < 0.05
        and seeds_agree
        and neg_ok
    ):
        return "demonstrated"
    if d > 0 and neg_ok:
        return "directional"
    return "not_supported"


def earliest_within_one_se(scores: dict[int, list[float]]) -> int:
    """Select earliest layer within 1 SE of the best mean Macro-F1.

    SE = SD / sqrt(n_folds) of the best layer's fold scores. Using the raw SD
    instead of SE is ~sqrt(n) times too aggressive and was the previous bug.
    """
    means = {k: float(np.mean(v)) for k, v in scores.items()}
    best = max(means.values())
    best_layer = max(means, key=means.get)
    n = len(scores[best_layer])
    best_sd = float(np.std(scores[best_layer], ddof=1 if n > 1 else 0))
    best_se = best_sd / float(np.sqrt(max(n, 1)))
    threshold = best - best_se
    candidates = sorted(k for k, m in means.items() if m >= threshold)
    return int(candidates[0])
