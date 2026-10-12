"""Fast regression tests for dense full-FT protocol helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from fine_tuning.full_ft_utils import (
    check_partition,
    decide_headline,
    evidence_criteria,
    label_aware_duration_batches,
    leakage_verdict,
    load_fold_progress,
    locked_config_from_folds,
    matched_encoder_epochs,
    missing_fold_items,
    save_fold_progress,
    score_index_hash,
    stable_fingerprint,
)


class TestProgressIdentity(unittest.TestCase):
    def test_stale_identity_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = load_fold_progress(root, 0, {"fingerprint": "a"})
            state["score_idx"] = [2, 1]
            state["score_hash"] = score_index_hash(state["score_idx"])
            save_fold_progress(root, 0, state)
            with self.assertRaises(RuntimeError):
                load_fold_progress(root, 0, {"fingerprint": "b"})

    def test_stored_hash_recomputed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            identity = {"fingerprint": "a"}
            state = load_fold_progress(root, 0, identity)
            state["score_idx"] = [1, 2]
            state["score_hash"] = score_index_hash([1, 3])
            save_fold_progress(root, 0, state)
            with self.assertRaises(RuntimeError):
                load_fold_progress(root, 0, identity)

    def test_fingerprint_order_independent(self):
        self.assertEqual(stable_fingerprint({"a": 1, "b": 2}), stable_fingerprint({"b": 2, "a": 1}))

    def test_completed_fold_rejects_malformed_predictions(self):
        state = {
            "status": "done",
            "score_idx": [0, 1],
            "tuned_frozen": {"pred": [0], "audit": {"ok": True}},
            "arms": {
                "uniform": {
                    "selected": {
                        "encoder_lr": 1e-5,
                        "head_lr": 1e-4,
                        "epoch": 3,
                        "mean_inner_macro_f1": 0.5,
                    },
                    "selection_audit": {"ok": True},
                    "seeds": {
                        str(seed): {
                            "pred_full": [0, 4],
                            "pred_matched_frozen": [0, 1],
                            "audit": {"ok": True},
                        }
                        for seed in (42, 43, 44)
                    },
                }
            },
        }
        missing = missing_fold_items(state, ("uniform",), (42, 43, 44))
        self.assertIn("tuned_frozen.pred length", missing)
        self.assertIn("uniform.seed42.pred_full classes", missing)


class TestProtocolDecisions(unittest.TestCase):
    def test_headline_truth_table(self):
        for a in (False, True):
            for b in (False, True):
                for c in (False, True):
                    expected = (
                        "full_ft"
                        if a and b
                        else "frozen_uniform_25"
                        if a and not b and c
                        else "partial_ft"
                    )
                    self.assertEqual(
                        decide_headline(a_pass=a, b_pass=b, c_pass=c), expected
                    )

    def test_margin_is_point_threshold(self):
        boot = {
            "delta_macro_f1": {"point": 0.011, "ci95": [0.001, 0.02]},
            "delta_negative_f1": {"point": 0.0, "ci95": [-0.01, 0.01]},
        }
        result = evidence_criteria(
            boot,
            {"p_value": 0.01},
            [0.01, 0.02, 0.03],
            point_margin=0.01,
            negative_f1_margin=-0.02,
        )
        self.assertTrue(result["pass"])
        self.assertIn("point estimate", result["margin_interpretation"])

    def test_whole_triple_locked_not_component_modes(self):
        rows = [
            {"uniform": {"selected": {"encoder_lr": 1e-5, "head_lr": 1e-4, "epoch": 3, "mean_inner_macro_f1": 0.6}}},
            {"uniform": {"selected": {"encoder_lr": 1e-5, "head_lr": 1e-4, "epoch": 3, "mean_inner_macro_f1": 0.61}}},
            {"uniform": {"selected": {"encoder_lr": 3e-5, "head_lr": 1e-3, "epoch": 10, "mean_inner_macro_f1": 0.7}}},
        ]
        locked = locked_config_from_folds(rows, "uniform")
        self.assertEqual(
            (locked["encoder_lr"], locked["head_lr"], locked["epoch"]),
            (1e-5, 1e-4, 3),
        )


class TestFoldsAndBatches(unittest.TestCase):
    def test_partition(self):
        check_partition([[0, 1], [2, 3]], 4)
        with self.assertRaises(ValueError):
            check_partition([[0, 1], [1, 2, 3]], 4)

    def test_encoder_step_matching(self):
        epochs = matched_encoder_epochs(
            inner_encoder_epochs=5,
            n_inner_fit=455,
            n_outer_fit=683,
            batch_size=16,
        )
        self.assertGreaterEqual(epochs, 1)

    def test_label_aware_batches_cover_rows_once(self):
        lengths = np.arange(1, 65)
        labels = np.array([2] * 8 + [0] * 24 + [1] * 32)
        batches = label_aware_duration_batches(
            lengths,
            labels,
            8,
            np.random.default_rng(42),
            minority_label=2,
        )
        np.testing.assert_array_equal(
            np.sort(np.concatenate(batches)), np.arange(len(labels))
        )

    def test_leakage_verdict_uses_ci(self):
        self.assertEqual(
            leakage_verdict([0.04, 0.08], safe_below=0.01, concern_above=0.03),
            "memorisation_concern",
        )
        self.assertEqual(
            leakage_verdict([-0.02, 0.005], safe_below=0.01, concern_above=0.03),
            "no_memorisation_signal",
        )
        self.assertEqual(
            leakage_verdict([0.0, 0.04], safe_below=0.01, concern_above=0.03),
            "inconclusive",
        )


if __name__ == "__main__":
    unittest.main()
