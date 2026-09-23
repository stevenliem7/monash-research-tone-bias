"""Unit tests for fine_tuning (no WavLM download required)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fine_tuning.cache import build_pooled_cache, load_pooled_cache, validate_meta
from fine_tuning.data import (
    class_weights_from_y,
    make_grouped_folds,
    manifest_fingerprint,
    remap_audio_paths,
)
from fine_tuning.metrics import (
    earliest_within_one_se,
    evidence_level,
    paired_group_bootstrap,
    paired_group_randomization,
    scalar_weight_diagnostics,
)
from fine_tuning.model import (
    FreezePlan,
    FrozenPooledClassifier,
    PartialWavLMClassifier,
    SuffixMixClassifier,
    ValenceHead,
    _encoder_layer_accepts_index,
    call_wavlm_encoder_layer,
    freeze_except_suffix,
    mask_aware_mean_pool,
    seed_everything,
)
from fine_tuning.optimise import (
    TrainConfig,
    duration_batches,
    predict_frozen_pooled,
    train_frozen_pooled,
)
from fine_tuning.paths import CACHE_FORMAT_VERSION, PROTOCOL_VERSION
from fine_tuning.splits import FoldSpec, LabelAccessGuard, LeakageError, outer_folds
from fine_tuning.stages import ExperimentConfig, stage_frozen_gate, stage_nested_cv


class FakeEncoder(torch.nn.Module):
    def __init__(self, n_layers: int = 4, hidden: int = 16):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [FakeLayer(hidden) for _ in range(n_layers)]
        )
        self.layer_norm = torch.nn.LayerNorm(hidden)


class FakeAttention(torch.nn.Module):
    def __init__(self, hidden: int = 16, n_heads: int = 2):
        super().__init__()
        self.num_heads = n_heads
        self.hidden = hidden

    def compute_bias(self, q_len: int, k_len: int) -> torch.Tensor:
        return torch.zeros(self.num_heads, q_len, k_len)


class FakeLayer(torch.nn.Module):
    def __init__(self, hidden: int = 16):
        super().__init__()
        self.linear = torch.nn.Linear(hidden, hidden)
        self.attention = FakeAttention(hidden)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        output_attentions=False,
        index=None,
    ):
        # Mask contract: binary (B, T) with 1=valid.
        if attention_mask is not None:
            assert attention_mask.dim() == 2
            assert set(attention_mask.unique().tolist()).issubset({0, 1})
        return self.linear(hidden_states), position_bias


class FakeWavLM(torch.nn.Module):
    def __init__(self, n_layers: int = 4, hidden: int = 16):
        super().__init__()
        self.encoder = FakeEncoder(n_layers, hidden)


class TestDataAndMetrics(unittest.TestCase):
    def test_class_weights(self):
        y = np.array([0, 0, 1, 2, 2, 2])
        w = class_weights_from_y(y, 3)
        self.assertEqual(w.shape, (3,))
        self.assertGreater(w[1], w[0])

    def test_grouped_folds_disjoint_groups(self):
        y = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0, 1, 2])
        groups = np.array(["a", "a", "b", "b", "c", "c", "d", "d", "e", "e", "f", "f"])
        folds = make_grouped_folds(y, groups, n_splits=3, seed=42)
        for tr, va in folds:
            self.assertEqual(len(set(groups[tr]) & set(groups[va])), 0)

    def test_remap_audio_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio_dir = Path(tmp)
            df = pd.DataFrame({"clip_id": ["A_1", "B_2"], "is_labelled": [True, True]})
            (audio_dir / "A_1.wav").write_bytes(b"x")
            (audio_dir / "B_2.wav").write_bytes(b"x")
            out = remap_audio_paths(df, audio_dir)
            self.assertTrue(out["audio_path"].str.endswith("A_1.wav").iloc[0])

    def test_scalar_diagnostics_uniform(self):
        diag = scalar_weight_diagnostics(np.zeros(5))
        self.assertAlmostEqual(diag["normalized_entropy"], 1.0, places=5)

    def test_earliest_within_one_se_uses_se_not_sd(self):
        # best mean=0.6 at layer 12; sd≈0.115, se≈0.058 → SE thresh≈0.542
        # layer 8 mean=0.52 is above SD thresh (~0.48) but below SE thresh.
        scores = {
            4: [0.40, 0.40, 0.40, 0.40],
            8: [0.52, 0.52, 0.52, 0.52],
            12: [0.70, 0.50, 0.70, 0.50],
        }
        chosen = earliest_within_one_se(scores)
        self.assertEqual(chosen, 12)

    def test_paired_bootstrap_runs(self):
        y = np.array([0, 1, 2, 0, 1, 2, 0, 1])
        a = np.array([0, 1, 1, 0, 1, 2, 0, 2])
        b = np.array([0, 1, 2, 0, 1, 2, 0, 1])
        g = np.array(["q1", "q1", "q2", "q2", "q3", "q3", "q4", "q4"])
        boot = paired_group_bootstrap(y, a, b, g, n_boot=50, seed=0)
        rand = paired_group_randomization(y, a, b, g, n_perm=50, seed=0)
        level = evidence_level(boot, rand, [0.05, 0.04, 0.03])
        self.assertIn(level, {"demonstrated", "directional", "not_supported"})


class TestModel(unittest.TestCase):
    def test_head_shape(self):
        head = ValenceHead(hidden_size=32)
        self.assertEqual(head(torch.randn(4, 32)).shape, (4, 3))

    def test_mask_aware_pool(self):
        h = torch.ones(2, 5, 4)
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])
        pooled = mask_aware_mean_pool(h, mask)
        self.assertEqual(pooled.shape, (2, 4))

    def test_freeze_plan(self):
        m = FakeWavLM(n_layers=6, hidden=8)
        plan = FreezePlan(active_start=4, active_end=6, retained_endpoint=6)
        names = freeze_except_suffix(m, plan)
        self.assertTrue(any("layers.4" in n for n in names))

    def test_encoder_layer_call_without_index(self):
        class LayerNoIndex(torch.nn.Module):
            def forward(
                self,
                hidden_states,
                attention_mask=None,
                position_bias=None,
                output_attentions=False,
            ):
                return hidden_states + 1.0, position_bias

        layer = LayerNoIndex()
        out, bias = call_wavlm_encoder_layer(
            layer,
            torch.zeros(2, 3, 4),
            attention_mask=torch.ones(2, 3, dtype=torch.long),
            position_bias=None,
            layer_idx=0,
        )
        self.assertEqual(out.shape, (2, 3, 4))
        self.assertFalse(_encoder_layer_accepts_index(LayerNoIndex))

    def test_encoder_layer_call_with_index(self):
        class LayerWithIndex(torch.nn.Module):
            def forward(
                self,
                hidden_states,
                attention_mask=None,
                position_bias=None,
                output_attentions=False,
                index=0,
            ):
                return hidden_states + float(index), position_bias

        layer = LayerWithIndex()
        self.assertTrue(_encoder_layer_accepts_index(LayerWithIndex))
        out, _ = call_wavlm_encoder_layer(
            layer,
            torch.zeros(1, 2, 3),
            attention_mask=torch.ones(1, 2, dtype=torch.long),
            position_bias=None,
            layer_idx=5,
        )
        self.assertTrue(torch.allclose(out, torch.full((1, 2, 3), 5.0)))

    def test_partial_classifier_head_on_wavlm_device(self):
        wavlm = FakeWavLM(n_layers=6, hidden=8)
        plan = FreezePlan(active_start=4, active_end=6, retained_endpoint=6)
        clf = PartialWavLMClassifier(wavlm, plan, n_mix_layers=1, hidden_size=8)
        wavlm_device = next(wavlm.parameters()).device
        self.assertEqual(next(clf.mix.parameters()).device, wavlm_device)
        self.assertEqual(next(clf.head.parameters()).device, wavlm_device)

    def test_live_mix_shape_and_final_norm(self):
        wavlm = FakeWavLM(n_layers=4, hidden=8)
        clf = SuffixMixClassifier(
            wavlm, active_start=2, active_end=4, n_frozen_lower=0, hidden_size=8
        )
        hidden = torch.randn(2, 5, 8)
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.long)
        logits = clf(hidden, mask)
        self.assertEqual(logits.shape, (2, 3))
        self.assertEqual(clf.mix.logits.numel(), 2)

    def test_seed_before_construction_is_deterministic(self):
        seed_everything(0)
        a = ValenceHead(hidden_size=8)
        seed_everything(0)
        b = ValenceHead(hidden_size=8)
        for pa, pb in zip(a.parameters(), b.parameters()):
            self.assertTrue(torch.allclose(pa, pb))

    def test_frozen_classifier_train(self):
        n, L, H = 40, 5, 16
        X = np.random.randn(n, L, H).astype(np.float32)
        y = np.array([i % 3 for i in range(n)], dtype=np.int64)
        model = FrozenPooledClassifier(n_layers=L, hidden_size=H, learn_mix=True)
        cfg = TrainConfig(max_epochs=3, min_epochs=1, patience=2, seed=0)
        out = train_frozen_pooled(
            model,
            X[:30],
            y[:30],
            class_weights_from_y(y[:30]),
            cfg=cfg,
            device="cpu",
            X_val=X[30:],
            y_val=y[30:],
        )
        self.assertIn("state_dict", out)
        pred = predict_frozen_pooled(model, X[30:], "cpu")
        self.assertEqual(pred.shape, (10,))

    def test_contiguous_duration_buckets(self):
        rng = np.random.default_rng(1)
        rows = np.arange(20)
        lengths = np.arange(20)
        batches = duration_batches(rows, lengths, 5, rng)
        self.assertEqual(sum(len(b) for b in batches), 20)


class TestOuterLabelIsolation(unittest.TestCase):
    def test_guard_rejects_outer_score_during_fit(self):
        y = np.arange(30) % 3
        groups = np.repeat(np.arange(10), 3).astype(str)
        outer = next(iter(outer_folds(y, groups, n_splits=5, seed=0)))
        guard = LabelAccessGuard(outer)
        guard.labels(y, outer.fit, "fit")
        guard.labels(y, outer.score, "select")
        with self.assertRaises(LeakageError):
            guard.assert_clean()


class TestGate0ArmsAndSkipPath(unittest.TestCase):
    def test_uniform_vec_is_mean_not_concat(self):
        from fine_tuning.honest_gate0 import FEATURES

        X = np.arange(2 * 25 * 4, dtype=np.float32).reshape(2, 25, 4)
        vec = FEATURES["uniform_vec"](X)
        self.assertEqual(vec.shape, (2, 1, 4))
        np.testing.assert_allclose(vec[:, 0, :], X.mean(axis=1))

    def test_svc_rejects_flattened_stack(self):
        from fine_tuning.honest_gate0 import run_svc_fixed
        from fine_tuning.splits import FoldSpec, LabelAccessGuard

        X = np.random.randn(20, 25 * 1024).astype(np.float32)
        y = np.array([i % 3 for i in range(20)])
        outer = FoldSpec(fit=np.arange(15), score=np.arange(15, 20))
        guard = LabelAccessGuard(outer)
        with self.assertRaises(ValueError):
            run_svc_fixed(X, y, outer, guard)

    def test_nested_skips_without_force_but_runs_with_ignore_gate0(self):
        """prefer_frozen_mix skips nested-cv alone; ignore_gate0 / stage=all does not."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rows = []
            for i in range(36):
                rows.append(
                    {
                        "clip_id": f"c{i}",
                        "source_batch": "corpusA_1200",
                        "question": f"q{i // 3}",
                        "emotion_label": "neutral",
                        "valence_label": "neutral",
                        "audio_path": f"/tmp/c{i}.wav",
                        "response_text": "x",
                        "ground_truth_label": ["positive", "neutral", "negative"][i % 3],
                        "is_labelled": True,
                        "duration_s": 3.0,
                        "split": "train",
                    }
                )
            manifest = tmp_path / "manifest.csv"
            pd.DataFrame(rows).to_csv(manifest, index=False)
            out_dir = tmp_path / "out"
            cache_dir = out_dir / "cache"
            reports = out_dir / "reports"
            reports.mkdir(parents=True)
            build_pooled_cache(
                clip_ids=[r["clip_id"] for r in rows],
                pooled=np.random.randn(len(rows), 25, 8).astype(np.float32),
                frame_lengths=np.full(len(rows), 10, dtype=np.int32),
                cache_dir=cache_dir,
                manifest_fingerprint=manifest_fingerprint(manifest),
                hidden_size=8,
                n_transformer_layers=24,
            )
            (reports / "gate0_frozen.json").write_text(
                json.dumps(
                    {
                        "prefer_frozen_mix": True,
                        "selected_scalar_lrs": [0.001],
                        "honest": {"folds": []},
                        "protocol_version": PROTOCOL_VERSION,
                    }
                ),
                encoding="utf-8",
            )
            from fine_tuning.stages import (
                stage_leakage_check,
                stage_nested_cv,
                stage_train_full,
            )

            cfg_skip = ExperimentConfig(
                manifest_path=manifest,
                output_dir=out_dir,
                stage="nested-cv",
                outer_folds=3,
                inner_folds=2,
                seed=0,
                use_partial_ft=False,
                nested_inner_max_epochs=1,
                ignore_gate0=False,
            )
            nested = stage_nested_cv(cfg_skip)
            self.assertTrue(nested["skipped"])
            leak = stage_leakage_check(cfg_skip)
            self.assertTrue(leak["skipped"])
            trained = stage_train_full(cfg_skip)
            self.assertEqual(trained["locked"]["mode"], "frozen_scalar_mix")

            cfg_force = ExperimentConfig(
                manifest_path=manifest,
                output_dir=out_dir,
                stage="nested-cv",
                outer_folds=3,
                inner_folds=2,
                seed=0,
                use_partial_ft=False,
                nested_inner_max_epochs=1,
                ignore_gate0=True,
            )
            nested2 = stage_nested_cv(cfg_force)
            self.assertFalse(nested2.get("skipped", False))
            self.assertIn("locked_config", nested2)


class TestCacheAndStages(unittest.TestCase):
    def test_pooled_cache_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            clip_ids = [f"c{i}" for i in range(6)]
            pooled = np.random.randn(6, 5, 8).astype(np.float32)
            lengths = np.arange(6, dtype=np.int32) + 1
            build_pooled_cache(
                clip_ids=clip_ids,
                pooled=pooled,
                frame_lengths=lengths,
                cache_dir=cache_dir,
                manifest_fingerprint="abc",
                hidden_size=8,
                n_transformer_layers=4,
            )
            expected = {
                "cache_format_version": CACHE_FORMAT_VERSION,
                "model_id": "microsoft/wavlm-large",
                "manifest_fingerprint": "abc",
                "hidden_size": 8,
                "n_transformer_layers": 4,
            }
            bundle = load_pooled_cache(cache_dir, expected)
            self.assertEqual(len(bundle["clip_id"]), 6)
            with self.assertRaises(ValueError):
                validate_meta({"cache_format_version": 999}, expected)

    def test_gate0_and_nested_with_fake_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rows = []
            for i in range(60):
                rows.append(
                    {
                        "clip_id": f"c{i}",
                        "source_batch": "corpusA_1200",
                        "question": f"q{i // 3}",
                        "emotion_label": "neutral",
                        "valence_label": "neutral",
                        "audio_path": f"/tmp/c{i}.wav",
                        "response_text": "x",
                        "ground_truth_label": ["positive", "neutral", "negative"][i % 3],
                        "is_labelled": True,
                        "duration_s": 3.0 + (i % 5) * 0.2,
                        "split": "train",
                    }
                )
            manifest = tmp_path / "manifest.csv"
            pd.DataFrame(rows).to_csv(manifest, index=False)
            out_dir = tmp_path / "out"
            cache_dir = out_dir / "cache"
            clip_ids = [r["clip_id"] for r in rows]
            pooled = np.random.randn(len(rows), 25, 16).astype(np.float32)
            build_pooled_cache(
                clip_ids=clip_ids,
                pooled=pooled,
                frame_lengths=np.full(len(rows), 10, dtype=np.int32),
                cache_dir=cache_dir,
                manifest_fingerprint=manifest_fingerprint(manifest),
                hidden_size=16,
                n_transformer_layers=24,
            )
            cfg = ExperimentConfig(
                manifest_path=manifest,
                output_dir=out_dir,
                outer_folds=3,
                inner_folds=2,
                seed=42,
                ignore_gate0=True,
                use_partial_ft=False,
                gate0_max_epochs=2,
                nested_inner_max_epochs=2,
                repeats=1,
            )
            gate = stage_frozen_gate(cfg)
            self.assertEqual(gate.get("protocol_version"), PROTOCOL_VERSION)
            self.assertIn("final_mlp", gate["honest"]["pooled"])
            self.assertIn("final_svc_fixed", gate["honest"]["pooled"])
            self.assertIn("final_svc_tuned", gate["honest"]["pooled"])
            self.assertIn("uniform_svc_fixed", gate["honest"]["pooled"])
            self.assertIn("uniform_svc_tuned", gate["honest"]["pooled"])
            self.assertIn("fixed_svc_source", gate)
            nested = stage_nested_cv(cfg)
            self.assertEqual(nested.get("protocol_version"), PROTOCOL_VERSION)
            self.assertIn("locked_config", nested)


@unittest.skipUnless(
    os.environ.get("WAVLM_PARITY_TEST") == "1",
    "Set WAVLM_PARITY_TEST=1 to download WavLM and run suffix parity",
)
class TestWavLMParityIntegration(unittest.TestCase):
    def test_padded_suffix_matches_full_forward(self):
        from fine_tuning.model import load_wavlm
        from fine_tuning.optimise import eval_suffix_parity
        from fine_tuning.paths import WAVLM_MODEL_ID

        device = "cpu"
        processor, model = load_wavlm(WAVLM_MODEL_ID, device)
        waves = [
            np.zeros(16_000, dtype=np.float32),
            np.zeros(24_000, dtype=np.float32),
        ]
        result = eval_suffix_parity(
            model, processor, waves, active_start=20, device=device
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["mask_contract"], "2d_binary_1_valid")


if __name__ == "__main__":
    unittest.main()
