"""Tests for the invariant the old suite missed.

Disjoint train/test indices are not enough. Scoring labels must never be read
for fitting or model selection.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fine_tuning.splits import (
    FoldSpec,
    LabelAccessGuard,
    LeakageError,
    audit_fold_family,
    epochs_for_matched_steps,
    inner_folds,
    leaky_outer_folds,
    outer_folds,
)

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def toy_corpus(n_groups: int = 60, per_group: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_groups), per_group).astype(str)
    n = len(groups)
    y = rng.choice([0, 1, 2], size=n, p=[0.28, 0.60, 0.12])
    for cls in range(3):
        if not np.any(y == cls):
            y[cls] = cls
    return y, groups


class TestFoldSpec(unittest.TestCase):
    def test_overlap_is_rejected(self):
        idx = np.arange(20)
        with self.assertRaises(LeakageError):
            FoldSpec(fit=idx[:15], select=idx[15:], score=idx[15:])

    def test_duplicate_indices_rejected(self):
        with self.assertRaises(ValueError):
            FoldSpec(fit=np.array([1, 1, 2]), score=np.array([3]))

    def test_empty_score_rejected(self):
        with self.assertRaises(ValueError):
            FoldSpec(fit=np.arange(5), score=np.array([], dtype=np.int64))

    def test_refit_cannot_early_stop(self):
        spec = FoldSpec(fit=np.arange(10), score=np.arange(10, 15))
        self.assertFalse(spec.can_select)
        with self.assertRaises(LeakageError):
            spec.require_select("trainer")

    def test_train_all_is_fit_plus_select(self):
        spec = FoldSpec(
            fit=np.arange(10), select=np.arange(10, 13), score=np.arange(13, 20)
        )
        np.testing.assert_array_equal(spec.train_all, np.arange(13))


class TestFoldConstruction(unittest.TestCase):
    def setUp(self):
        self.y, self.groups = toy_corpus()

    def test_outer_folds_have_no_select_set(self):
        for spec in outer_folds(self.y, self.groups, n_splits=5, n_repeats=2, seed=42):
            self.assertFalse(spec.can_select, f"{spec.label} can early stop")

    def test_every_row_scored_once_per_repeat(self):
        specs = list(outer_folds(self.y, self.groups, n_splits=5, n_repeats=3, seed=42))
        audit = audit_fold_family(specs, len(self.y))
        self.assertEqual(audit["rows_never_scored"], 0)
        self.assertEqual(audit["max_times_scored"], 3)
        self.assertFalse(audit["any_select_sets"])

    def test_inner_folds_stay_inside_the_outer_training_partition(self):
        for outer in outer_folds(self.y, self.groups, n_splits=5, seed=42):
            for inner in inner_folds(outer, self.y, self.groups, n_splits=3, seed=42):
                self.assertEqual(len(np.intersect1d(inner.train_all, outer.score)), 0)
                self.assertEqual(len(np.intersect1d(inner.score, outer.score)), 0)

    def test_groups_never_span_a_fold_boundary(self):
        for outer in outer_folds(self.y, self.groups, n_splits=5, seed=42):
            train_groups = set(self.groups[outer.fit].tolist())
            score_groups = set(self.groups[outer.score].tolist())
            self.assertEqual(train_groups & score_groups, set())

    def test_leaky_folds_are_labelled_as_such(self):
        specs = list(leaky_outer_folds(self.y, self.groups, n_splits=5, seed=42))
        self.assertTrue(all(s.label.startswith("LEAKY") for s in specs))
        self.assertTrue(all(s.can_select for s in specs))
        for spec in specs:
            np.testing.assert_array_equal(spec.select, spec.score)


class TestLabelGuard(unittest.TestCase):
    def setUp(self):
        self.y, self.groups = toy_corpus()
        self.outer = next(iter(outer_folds(self.y, self.groups, n_splits=5, seed=42)))
        self.inners = inner_folds(self.outer, self.y, self.groups, n_splits=3, seed=42)

    def test_honest_trainer_passes(self):
        guard = LabelAccessGuard(self.outer)
        for inner in self.inners:
            guard.labels(self.y, inner.fit, "fit")
            guard.labels(self.y, inner.score, "select")
        guard.labels(self.y, self.outer.fit, "fit")
        guard.labels(self.y, self.outer.score, "final_score")
        guard.assert_clean()

    def test_old_call_pattern_is_caught(self):
        guard = LabelAccessGuard(self.outer)
        guard.labels(self.y, self.outer.fit, "fit")
        guard.labels(self.y, self.outer.score, "select")
        with self.assertRaises(LeakageError) as ctx:
            guard.assert_clean()
        self.assertIn("contaminated", str(ctx.exception))

    def test_partial_leak_of_a_single_clip_is_caught(self):
        guard = LabelAccessGuard(self.outer)
        sneaky = np.concatenate([self.outer.fit, self.outer.score[:1]])
        guard.labels(self.y, sneaky, "fit")
        with self.assertRaises(LeakageError):
            guard.assert_clean()

    def test_summary_counts_reads(self):
        guard = LabelAccessGuard(self.outer)
        guard.labels(self.y, self.outer.fit, "fit")
        summary = guard.summary()
        self.assertEqual(summary["label_reads"]["fit"], len(self.outer.fit))
        self.assertEqual(summary["n_select"], 0)


class TestEpochTransfer(unittest.TestCase):
    def test_more_data_means_fewer_epochs_for_the_same_steps(self):
        got = epochs_for_matched_steps(
            inner_epochs=9, n_inner_fit=455, n_outer_fit=683, batch_size=16
        )
        self.assertEqual(got, 6)

    def test_identical_partitions_are_a_no_op(self):
        got = epochs_for_matched_steps(
            inner_epochs=7, n_inner_fit=500, n_outer_fit=500, batch_size=16
        )
        self.assertEqual(got, 7)

    def test_never_returns_zero(self):
        got = epochs_for_matched_steps(
            inner_epochs=1, n_inner_fit=50, n_outer_fit=5000, batch_size=16
        )
        self.assertGreaterEqual(got, 1)


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestMaskingAndBatching(unittest.TestCase):
    def test_wavlm_mask_contract_is_2d_binary(self):
        """Transformers 5.0 WavLM expects (B, T) with 1=valid, not a 4D additive mask."""
        frame_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.long)
        self.assertEqual(tuple(frame_mask.shape), (2, 4))
        self.assertTrue(set(frame_mask.unique().tolist()).issubset({0, 1}))
        # key_padding_mask = attention_mask.ne(1)
        key_padding = frame_mask.ne(1)
        self.assertTrue(bool(key_padding[0, 2]))
        self.assertFalse(bool(key_padding[1, 3]))

    def test_duration_batches_cover_every_row_exactly_once(self):
        from fine_tuning.optimise import duration_batches

        rng = np.random.default_rng(0)
        rows = np.arange(40)
        lengths = rng.integers(100, 900, size=200)
        batches = duration_batches(rows, lengths, 8, rng)
        seen = np.sort(np.concatenate(batches))
        np.testing.assert_array_equal(seen, np.arange(40))

    def test_duration_batches_are_length_contiguous_before_shuffle(self):
        from fine_tuning.optimise import duration_batches

        rng = np.random.default_rng(0)
        rows = np.arange(16)
        lengths = np.arange(16) * 10
        # Force deterministic shuffle seed; check each batch is contiguous in
        # sorted-length order by reconstructing without shuffle.
        order = np.argsort(lengths[rows], kind="stable")
        sorted_rows = rows[order]
        expected = [sorted_rows[s : s + 4] for s in range(0, 16, 4)]
        batches = duration_batches(rows, lengths, 4, rng)
        # Same multiset of batches (order shuffled).
        self.assertEqual(
            sorted(tuple(b.tolist()) for b in batches),
            sorted(tuple(b.tolist()) for b in expected),
        )


if __name__ == "__main__":
    unittest.main()
