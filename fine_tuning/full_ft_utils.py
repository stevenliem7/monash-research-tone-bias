"""Pure helpers for dense full fine-tuning."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


def atomic_write_json(path: Path, payload: dict) -> None:
    """Atomically replace JSON so concurrent readers never see partial data."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def stable_fingerprint(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score_index_hash(indices: Iterable[int]) -> str:
    values = np.sort(np.asarray(list(indices), dtype=np.int64))
    return hashlib.sha256(values.tobytes()).hexdigest()


def fold_progress_path(root: Path, fold: int) -> Path:
    return Path(root) / "full_ft_progress" / f"fold{int(fold)}.json"


def empty_fold_progress(fold: int, identity: dict) -> dict:
    return {
        "fold": int(fold),
        "status": "running",
        "identity": identity,
        "score_idx": None,
        "score_hash": None,
        "tuned_frozen": None,
        "selection_runs": {},
        "arms": {},
    }


def load_fold_progress(root: Path, fold: int, identity: dict) -> dict:
    path = fold_progress_path(root, fold)
    if not path.is_file():
        return empty_fold_progress(fold, identity)
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("identity") != identity:
        raise RuntimeError(
            f"fold {fold}: stale progress identity; move/delete {path} before rerun"
        )
    stored = state.get("score_idx")
    if stored is not None and state.get("score_hash") != score_index_hash(stored):
        raise RuntimeError(f"fold {fold}: stored score hash does not match score_idx")
    return state


def save_fold_progress(root: Path, fold: int, state: dict) -> None:
    atomic_write_json(fold_progress_path(root, fold), state)


def check_partition(score_sets: Sequence[Sequence[int]], n_rows: int) -> None:
    counts = np.zeros(n_rows, dtype=np.int64)
    for values in score_sets:
        idx = np.asarray(values, dtype=np.int64)
        if np.any(idx < 0) or np.any(idx >= n_rows):
            raise ValueError("outer score index outside manifest range")
        counts[idx] += 1
    if np.any(counts > 1):
        raise ValueError(f"{int(np.sum(counts > 1))} rows scored by multiple folds")
    if np.any(counts == 0):
        raise ValueError(f"{int(np.sum(counts == 0))} rows never scored")


def missing_fold_items(
    state: dict, arms: Sequence[str], seeds: Sequence[int]
) -> list[str]:
    missing: list[str] = []
    if state.get("status") != "done":
        missing.append(f"status={state.get('status')!r}")
    if not state.get("tuned_frozen"):
        missing.append("tuned_frozen")
    if state.get("score_idx") is None:
        missing.append("score_idx")
    n_score = len(state.get("score_idx") or [])
    tuned = state.get("tuned_frozen")
    if tuned:
        predictions = tuned.get("pred")
        if not isinstance(predictions, list) or len(predictions) != n_score:
            missing.append("tuned_frozen.pred length")
        elif any(int(value) not in (0, 1, 2) for value in predictions):
            missing.append("tuned_frozen.pred classes")
        if "meta" not in tuned:
            missing.append("tuned_frozen.meta")
        if not tuned.get("audit"):
            missing.append("tuned_frozen.audit")
    for arm in arms:
        arm_state = state.get("arms", {}).get(arm)
        if not arm_state or "selected" not in arm_state:
            missing.append(f"{arm}.selected")
            continue
        selected = arm_state["selected"]
        if not all(
            key in selected
            for key in ("encoder_lr", "head_lr", "epoch", "mean_inner_macro_f1")
        ):
            missing.append(f"{arm}.selected fields")
        if not arm_state.get("selection_audit"):
            missing.append(f"{arm}.selection_audit")
        for seed in seeds:
            seed_state = arm_state.get("seeds", {}).get(str(seed))
            if not seed_state:
                missing.append(f"{arm}.seed{seed}")
            elif "pred_full" not in seed_state or "pred_matched_frozen" not in seed_state:
                missing.append(f"{arm}.seed{seed}.predictions")
            else:
                for field in ("pred_full", "pred_matched_frozen"):
                    predictions = seed_state[field]
                    if not isinstance(predictions, list) or len(predictions) != n_score:
                        missing.append(f"{arm}.seed{seed}.{field} length")
                    elif any(int(value) not in (0, 1, 2) for value in predictions):
                        missing.append(f"{arm}.seed{seed}.{field} classes")
                if "meta" not in seed_state:
                    missing.append(f"{arm}.seed{seed}.meta")
                if "diagnostics" not in seed_state:
                    missing.append(f"{arm}.seed{seed}.diagnostics")
                if not seed_state.get("audit"):
                    missing.append(f"{arm}.seed{seed}.audit")
    return missing


def steps_per_epoch(n: int, batch_size: int) -> int:
    return max(1, math.ceil(n / batch_size))


def matched_encoder_epochs(
    *,
    inner_encoder_epochs: float,
    n_inner_fit: int,
    n_outer_fit: int,
    batch_size: int,
) -> int:
    target = inner_encoder_epochs * steps_per_epoch(n_inner_fit, batch_size)
    return max(1, int(round(target / steps_per_epoch(n_outer_fit, batch_size))))


def label_aware_duration_batches(
    lengths: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    *,
    minority_label: int,
) -> list[np.ndarray]:
    """Length-local positions, repaired to include minority clips where possible."""
    lengths = np.asarray(lengths)
    labels = np.asarray(labels)
    if len(lengths) != len(labels):
        raise ValueError("lengths and labels must align")
    order = np.argsort(lengths, kind="stable")
    batches = [
        list(order[start : start + batch_size])
        for start in range(0, len(order), batch_size)
    ]

    def minority_count(batch: list[int]) -> int:
        return int(np.sum(labels[batch] == minority_label))

    for batch in batches:
        if minority_count(batch):
            continue
        target = float(np.median(lengths[batch]))
        best: tuple[float, list[int], int] | None = None
        for donor in batches:
            if donor is batch or minority_count(donor) < 2:
                continue
            for position in donor:
                if labels[position] != minority_label:
                    continue
                gap = abs(float(lengths[position]) - target)
                if best is None or gap < best[0]:
                    best = (gap, donor, position)
        if best is None:
            break
        _, donor, minority_position = best
        replacement = min(
            batch,
            key=lambda position: abs(
                float(lengths[position]) - float(lengths[minority_position])
            ),
        )
        donor[donor.index(minority_position)] = replacement
        batch[batch.index(replacement)] = minority_position

    result = [np.asarray(batch, dtype=np.int64) for batch in batches]
    rng.shuffle(result)
    return result


def minority_coverage(
    batches: Sequence[np.ndarray], labels: np.ndarray, minority_label: int
) -> float:
    if not batches:
        return 0.0
    return float(
        np.mean([np.any(np.asarray(labels)[batch] == minority_label) for batch in batches])
    )


def evidence_criteria(
    boot: dict,
    rand: dict,
    seed_deltas: Sequence[float],
    *,
    point_margin: float,
    negative_f1_margin: float,
) -> dict:
    """Practical point threshold plus evidence that improvement exceeds zero."""
    point = float(boot["delta_macro_f1"]["point"])
    macro_lo = float(boot["delta_macro_f1"]["ci95"][0])
    negative_lo = float(boot["delta_negative_f1"]["ci95"][0])
    criteria = {
        "point_at_least_practical_margin": point >= point_margin,
        "ci_lower_above_zero": macro_lo > 0.0,
        "randomization_p_below_0_05": float(rand["p_value"]) < 0.05,
        "all_predeclared_seeds_positive": bool(seed_deltas)
        and all(float(delta) > 0 for delta in seed_deltas),
        "negative_f1_non_inferior": negative_lo > negative_f1_margin,
    }
    return {
        "pass": all(criteria.values()),
        "criteria": criteria,
        "point": point,
        "ci95": [macro_lo, float(boot["delta_macro_f1"]["ci95"][1])],
        "negative_f1_ci95": [
            negative_lo,
            float(boot["delta_negative_f1"]["ci95"][1]),
        ],
        "seed_deltas": [float(delta) for delta in seed_deltas],
        "point_margin": float(point_margin),
        "negative_f1_margin": float(negative_f1_margin),
        "margin_interpretation": (
            "adoption threshold on point estimate; CI/randomization test improvement over zero"
        ),
    }


def evidence_label(result: dict) -> str:
    if result["pass"]:
        return "demonstrated"
    if result["point"] > 0 and result["criteria"]["negative_f1_non_inferior"]:
        return "directional"
    return "not_supported"


def decide_headline(*, a_pass: bool, b_pass: bool, c_pass: bool) -> str:
    if a_pass and b_pass:
        return "full_ft"
    if a_pass and not b_pass and c_pass:
        return "frozen_uniform_25"
    return "partial_ft"


HEADLINE_RULE = (
    "full_ft iff A and B pass; frozen_uniform_25 iff A passes, B fails, and C "
    "passes; otherwise partial_ft"
)


def locked_config_from_folds(fold_rows: Sequence[dict], arm: str) -> dict:
    """Lock modal whole (encoder LR, head LR, epoch) triple with deterministic ties."""
    triples: dict[tuple[float, float, int], list[float]] = {}
    for row in fold_rows:
        selected = row[arm]["selected"]
        triple = (
            float(selected["encoder_lr"]),
            float(selected["head_lr"]),
            int(selected["epoch"]),
        )
        triples.setdefault(triple, []).append(float(selected["mean_inner_macro_f1"]))
    if not triples:
        return {
            "pool_mode": arm,
            "encoder_lr": None,
            "head_lr": None,
            "epoch": None,
        }
    # Count first, mean inner score second, then earlier/lower values.
    best = max(
        triples,
        key=lambda triple: (
            len(triples[triple]),
            float(np.mean(triples[triple])),
            -triple[2],
            -triple[0],
            -triple[1],
        ),
    )
    return {
        "pool_mode": arm,
        "encoder_lr": best[0],
        "head_lr": best[1],
        "epoch": best[2],
        "n_folds_selecting_whole_triple": len(triples[best]),
        "tie_break": "count, mean inner macro-F1, earlier epoch, lower encoder/head LR",
    }


def duration_summary(sample_lengths: Sequence[int], sr: int = 16_000) -> dict:
    seconds = np.asarray(sample_lengths, dtype=np.float64) / sr
    if not len(seconds):
        return {"n": 0}
    p50, p90, p99 = np.quantile(seconds, [0.5, 0.9, 0.99])
    return {
        "n": int(len(seconds)),
        "min_s": round(float(seconds.min()), 3),
        "p50_s": round(float(p50), 3),
        "p90_s": round(float(p90), 3),
        "p99_s": round(float(p99), 3),
        "max_s": round(float(seconds.max()), 3),
        "total_hours": round(float(seconds.sum() / 3600), 3),
    }


def anchor_check(value: float, reference: float, tolerance: float) -> dict:
    difference = float(value) - float(reference)
    return {
        "value": float(value),
        "reference": float(reference),
        "diff": round(difference, 4),
        "tolerance": float(tolerance),
        "reproduced": abs(difference) <= tolerance,
    }


def leakage_verdict(
    ci95: Sequence[float], *, safe_below: float, concern_above: float
) -> str:
    lower, upper = map(float, ci95)
    if lower > concern_above:
        return "memorisation_concern"
    if upper < safe_below:
        return "no_memorisation_signal"
    return "inconclusive"


__all__ = [
    "HEADLINE_RULE",
    "anchor_check",
    "atomic_write_json",
    "check_partition",
    "decide_headline",
    "duration_summary",
    "evidence_criteria",
    "evidence_label",
    "file_sha256",
    "fold_progress_path",
    "label_aware_duration_batches",
    "leakage_verdict",
    "load_fold_progress",
    "locked_config_from_folds",
    "matched_encoder_epochs",
    "minority_coverage",
    "missing_fold_items",
    "save_fold_progress",
    "score_index_hash",
    "stable_fingerprint",
]
