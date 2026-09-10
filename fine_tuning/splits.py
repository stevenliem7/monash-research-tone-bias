"""Leak-proof fold primitives.

The bug this module exists to prevent: a training helper that takes
``(X_tr, y_tr, X_va, y_va)``, early-stops on the second pair, then predicts the
second pair. Callers cannot use such a function correctly, because the signature
does not distinguish "data I may select on" from "data I must not touch".

Here every split carries three disjoint index sets.

    fit     indices whose labels update weights
    select  indices whose labels choose epochs, hyperparameters, checkpoints
    score   indices that are predicted once and never read during training

An outer refit is built with an empty ``select`` set. A trainer asked to early
stop on an empty select set raises. The leak becomes impossible to write rather
than merely discouraged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator, Sequence

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


class LeakageError(RuntimeError):
    """Raised when a scoring index reaches training or model selection."""


def _as_idx(values: Iterable[int]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.int64)
    if arr.ndim != 1:
        raise ValueError(f"index array must be 1-D, got shape {arr.shape}")
    if len(np.unique(arr)) != len(arr):
        raise ValueError("index array contains duplicates")
    return np.sort(arr)


@dataclass(frozen=True)
class FoldSpec:
    """One fold, with the three roles kept apart.

    ``select`` is empty for a refit. That is the whole point: a refit has
    nothing to early stop on, so it must run a predetermined number of steps.
    """

    fit: np.ndarray
    score: np.ndarray
    select: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    repeat: int = 0
    fold: int = 0
    label: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "fit", _as_idx(self.fit))
        object.__setattr__(self, "score", _as_idx(self.score))
        object.__setattr__(self, "select", _as_idx(self.select))
        if len(self.fit) == 0:
            raise ValueError("fit set is empty")
        if len(self.score) == 0:
            raise ValueError("score set is empty")
        pairs = (
            ("fit", "select", self.fit, self.select),
            ("fit", "score", self.fit, self.score),
            ("select", "score", self.select, self.score),
        )
        for a_name, b_name, a, b in pairs:
            overlap = np.intersect1d(a, b)
            if len(overlap) > 0:
                raise LeakageError(
                    f"{a_name} and {b_name} overlap on {len(overlap)} indices, "
                    f"first={int(overlap[0])}"
                )

    @property
    def can_select(self) -> bool:
        return len(self.select) > 0

    @property
    def train_all(self) -> np.ndarray:
        """Every index a model is allowed to learn from, fit plus select."""
        return np.sort(np.concatenate([self.fit, self.select]))

    def require_select(self, who: str) -> np.ndarray:
        if not self.can_select:
            raise LeakageError(
                f"{who} asked for a selection set on a fold that has none. "
                "This fold is a refit. Pass a predetermined step count instead "
                "of early stopping."
            )
        return self.select

    def describe(self) -> dict:
        return {
            "label": self.label,
            "repeat": self.repeat,
            "fold": self.fold,
            "n_fit": int(len(self.fit)),
            "n_select": int(len(self.select)),
            "n_score": int(len(self.score)),
        }


class LabelAccessGuard:
    """Records which labels a trainer reads, then checks the score set is clean.

    Trainers read labels through ``guard.labels(y, idx, purpose)`` instead of
    ``y[idx]``. At the end of a fold, ``assert_clean()`` fails if any scoring
    index was read for anything other than final scoring.
    """

    def __init__(self, spec: FoldSpec):
        self.spec = spec
        self._score = set(spec.score.tolist())
        self.reads: list[tuple[str, int]] = []
        self._touched: set[int] = set()

    def labels(self, y: np.ndarray, idx: np.ndarray, purpose: str) -> np.ndarray:
        idx = np.asarray(idx, dtype=np.int64)
        if purpose not in {"fit", "select", "final_score"}:
            raise ValueError(f"unknown label purpose {purpose!r}")
        if purpose != "final_score":
            self._touched.update(idx.tolist())
        self.reads.append((purpose, int(len(idx))))
        return y[idx]

    def assert_clean(self) -> None:
        bad = sorted(self._touched & self._score)
        if bad:
            raise LeakageError(
                f"{len(bad)} scoring labels were read during fit or select, "
                f"first={bad[0]}. The fold is contaminated."
            )

    def summary(self) -> dict:
        by_purpose: dict[str, int] = {}
        for purpose, n in self.reads:
            by_purpose[purpose] = by_purpose.get(purpose, 0) + n
        return {"label_reads": by_purpose, **self.spec.describe()}


def grouped_splits(
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [
        (np.asarray(tr, dtype=np.int64), np.asarray(te, dtype=np.int64))
        for tr, te in cv.split(np.zeros(len(y)), y, groups)
    ]


def outer_folds(
    y: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int = 5,
    n_repeats: int = 1,
    seed: int = 42,
) -> Iterator[FoldSpec]:
    """Repeated grouped folds. Every outer fold is a refit, so select is empty.

    Repeats reduce partition noise. They do not add data, and the reported
    standard error across repeats is not an independent-sample standard error.
    """
    for repeat in range(n_repeats):
        for fold, (tr, te) in enumerate(
            grouped_splits(y, groups, n_splits, seed + repeat)
        ):
            yield FoldSpec(
                fit=tr,
                score=te,
                select=np.empty(0, dtype=np.int64),
                repeat=repeat,
                fold=fold,
                label=f"outer[r{repeat}f{fold}]",
            )


def inner_folds(
    outer: FoldSpec,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int = 3,
    seed: int = 42,
) -> list[FoldSpec]:
    """Selection folds carved from the outer fit partition only.

    Each inner fold scores on a slice of the outer training data. The outer
    score set never appears. Inner folds return numbers, never weights: the
    caller uses them to pick an epoch count or hyperparameter, then refits.
    """
    train_idx = outer.fit
    y_tr = y[train_idx]
    g_tr = groups[train_idx]
    specs: list[FoldSpec] = []
    for i, (itr, iva) in enumerate(grouped_splits(y_tr, g_tr, n_splits, seed)):
        specs.append(
            FoldSpec(
                fit=train_idx[itr],
                score=train_idx[iva],
                select=np.empty(0, dtype=np.int64),
                repeat=outer.repeat,
                fold=i,
                label=f"{outer.label}.inner[{i}]",
            )
        )
    for spec in specs:
        overlap = np.intersect1d(spec.train_all, outer.score)
        if len(overlap) > 0:
            raise LeakageError("inner fold leaked outer scoring indices")
    return specs


def leaky_outer_folds(
    y: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int = 5,
    n_repeats: int = 1,
    seed: int = 42,
) -> Iterator[FoldSpec]:
    """Deliberately contaminated folds, for calibration only.

    These reproduce the old behaviour, where the scoring fold was also the early
    stopping fold. Running both variants gives the leaked minus honest gap,
    which measures how much the old harness inflated its own numbers. Never
    report a leaky number on its own.
    """
    for repeat in range(n_repeats):
        for fold, (tr, te) in enumerate(
            grouped_splits(y, groups, n_splits, seed + repeat)
        ):
            spec = FoldSpec.__new__(FoldSpec)
            object.__setattr__(spec, "fit", _as_idx(tr))
            object.__setattr__(spec, "score", _as_idx(te))
            object.__setattr__(spec, "select", _as_idx(te))
            object.__setattr__(spec, "repeat", repeat)
            object.__setattr__(spec, "fold", fold)
            object.__setattr__(spec, "label", f"LEAKY[r{repeat}f{fold}]")
            yield spec


def steps_per_epoch(n: int, batch_size: int) -> int:
    return max(1, (n + batch_size - 1) // batch_size)


def epochs_for_matched_steps(
    *,
    inner_epochs: float,
    n_inner_fit: int,
    n_outer_fit: int,
    batch_size: int,
) -> int:
    """Convert an epoch count chosen on a small partition to the larger one.

    Inner folds train on roughly 455 clips, the outer refit on roughly 683. The
    same epoch count is 1.5 times the optimizer steps. Match steps, not epochs.
    """
    target_steps = inner_epochs * steps_per_epoch(n_inner_fit, batch_size)
    outer_spe = steps_per_epoch(n_outer_fit, batch_size)
    return max(1, int(round(target_steps / outer_spe)))


def audit_fold_family(specs: Sequence[FoldSpec], n_rows: int) -> dict:
    """Cheap sanity report over a set of folds."""
    covered = np.zeros(n_rows, dtype=np.int64)
    for spec in specs:
        covered[spec.score] += 1
    return {
        "n_folds": len(specs),
        "rows_never_scored": int(np.sum(covered == 0)),
        "max_times_scored": int(covered.max()) if n_rows else 0,
        "any_select_sets": any(s.can_select for s in specs),
    }
