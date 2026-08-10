#!/usr/bin/env python3
"""
Authors:
    Steven Liem (steven.liem@sydney.edu.au)

Compare a text-modality CV JSON against frozen WavLM group=question results.

Reports matched-pair deltas, best-vs-best verdict (primary), fixed-head
headline verdict (secondary), per-class F1 gaps, and negative-class
recall/precision calibration (predicted_count/true_count = R/P).

Usage:
    uv run python src/our_speech_corpus_final/heads/compare_text_vs_wavlm.py \\
        --text data/our_speech_corpus_text/reports/valence_head_cv_minilm_group_question.json \\
        --wavlm data/our_speech_corpus_wavlm/reports/valence_head_cv_wavlm_group_question.json
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
from typing import Any

from config import bootstrap_imports, extend_our_speech_corpus_paths

bootstrap_imports(__file__)
extend_our_speech_corpus_paths()
from config import (  # noqa: E402
    VALENCE_CLASSES,
    WAVLM_CV_GROUP_QUESTION_JSON,
)
from text_baseline_common import AUDIO_GAP, MATCH_TOL  # noqa: E402


def _best_head(grid_regime: dict) -> tuple[str, dict]:
    """Pick the head with highest mean macro-F1 in one regime.

    Args:
        grid_regime: Mapping candidate -> summary.

    Returns:
        tuple: (name, summary).
    """
    return max(
        grid_regime.items(),
        key=lambda kv: float(kv[1]["macro_f1"]["mean"]),
    )


def _verdict_from_delta(delta: float) -> dict[str, Any]:
    """Map Δ macro-F1 (text − WavLM) to status + interpretation.

    Args:
        delta: text_macro_f1 - wavlm_macro_f1.

    Returns:
        dict: status, delta, interpretation.
    """
    abs_delta = abs(delta)
    if abs_delta <= MATCH_TOL:
        status = "match"
        interpretation = (
            f"Text matches WavLM within ±{MATCH_TOL:.2f} macro-F1. "
            "Instrument is largely reading words."
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
            f"Text leads WavLM by {delta:.3f} macro-F1. "
            "Reframe toward lexical (+ tonal) skew."
        )
    else:
        status = "inconclusive"
        interpretation = (
            f"|Δ macro-F1|={abs_delta:.3f} between match (±{MATCH_TOL:.2f}) "
            f"and clear-gap ({AUDIO_GAP:.2f}) thresholds."
        )
    return {
        "status": status,
        "delta_macro_f1_text_minus_wavlm": float(delta),
        "match_tolerance": MATCH_TOL,
        "audio_gap_threshold": AUDIO_GAP,
        "interpretation": interpretation,
    }


def _class_f1(summary: dict, label: str) -> float:
    """Mean per-class F1 from a CV summary cell.

    Args:
        summary: Candidate summary.
        label: Class name.

    Returns:
        float: Mean F1.
    """
    return float(summary["per_class"][label]["f1"]["mean"])


def _neg_rp(summary: dict) -> tuple[float, float, float]:
    """Negative-class recall, precision, and R/P calibration ratio.

    Args:
        summary: Candidate summary.

    Returns:
        tuple: (recall, precision, recall/precision).
    """
    neg = summary["per_class"]["negative"]
    r = float(neg["recall"]["mean"])
    p = float(neg["precision"]["mean"])
    ratio = (r / p) if p > 1e-12 else float("inf")
    return r, p, ratio


def compare_text_vs_wavlm(
    text_results: dict,
    wavlm_results: dict,
    headline_regime: str = "balanced",
    headline_cell: str = "rbf_svc",
) -> dict:
    """Build matched-pair, best-vs-best, per-class, and calibration comparison.

    Args:
        text_results: Text CV JSON payload.
        wavlm_results: WavLM CV JSON payload.
        headline_regime: Regime for primary reporting.
        headline_cell: Fixed head name for secondary verdict (WavLM headline).

    Returns:
        dict: Full comparison payload.
    """
    rows: list[dict] = []
    calibration: list[dict] = []

    for regime in ("balanced", "empirical"):
        text_grid = text_results.get("grid", {}).get(regime, {})
        wavlm_grid = wavlm_results.get("grid", {}).get(regime, {})
        shared = sorted(set(text_grid) & set(wavlm_grid))
        for name in shared:
            ts, ws = text_grid[name], wavlm_grid[name]
            t_mac = float(ts["macro_f1"]["mean"])
            w_mac = float(ws["macro_f1"]["mean"])
            row = {
                "regime": regime,
                "head": name,
                "text_macro_f1": t_mac,
                "wavlm_macro_f1": w_mac,
                "delta_macro_f1_text_minus_wavlm": t_mac - w_mac,
                "text_neg_recall": float(ts["per_class"]["negative"]["recall"]["mean"]),
                "wavlm_neg_recall": float(ws["per_class"]["negative"]["recall"]["mean"]),
                "per_class_f1_delta": {
                    label: _class_f1(ts, label) - _class_f1(ws, label)
                    for label in VALENCE_CLASSES
                },
                "text_per_class_f1": {label: _class_f1(ts, label) for label in VALENCE_CLASSES},
                "wavlm_per_class_f1": {
                    label: _class_f1(ws, label) for label in VALENCE_CLASSES
                },
            }
            rows.append(row)

            t_r, t_p, t_ratio = _neg_rp(ts)
            w_r, w_p, w_ratio = _neg_rp(ws)
            calibration.append(
                {
                    "regime": regime,
                    "head": name,
                    "text_neg_recall": t_r,
                    "text_neg_precision": t_p,
                    "text_neg_pred_over_true": t_ratio,
                    "wavlm_neg_recall": w_r,
                    "wavlm_neg_precision": w_p,
                    "wavlm_neg_pred_over_true": w_ratio,
                }
            )

    text_best_name, text_best = _best_head(
        text_results["grid"][headline_regime]
    )
    wavlm_best_name, wavlm_best = _best_head(
        wavlm_results["grid"][headline_regime]
    )
    best_delta = float(text_best["macro_f1"]["mean"]) - float(
        wavlm_best["macro_f1"]["mean"]
    )
    best_vs_best = {
        "regime": headline_regime,
        "text_head": text_best_name,
        "wavlm_head": wavlm_best_name,
        "text_macro_f1": float(text_best["macro_f1"]["mean"]),
        "wavlm_macro_f1": float(wavlm_best["macro_f1"]["mean"]),
        "delta_macro_f1_text_minus_wavlm": best_delta,
        "text_per_class_f1": {
            label: _class_f1(text_best, label) for label in VALENCE_CLASSES
        },
        "wavlm_per_class_f1": {
            label: _class_f1(wavlm_best, label) for label in VALENCE_CLASSES
        },
        "per_class_f1_delta": {
            label: _class_f1(text_best, label) - _class_f1(wavlm_best, label)
            for label in VALENCE_CLASSES
        },
        "verdict": _verdict_from_delta(best_delta),
    }

    headline_verdict: dict[str, Any]
    t_cell = text_results["grid"][headline_regime].get(headline_cell)
    w_cell = wavlm_results["grid"][headline_regime].get(headline_cell)
    if t_cell is None or w_cell is None:
        headline_verdict = {
            "status": "unavailable",
            "reason": f"Missing {headline_regime}/{headline_cell} in one or both grids",
        }
    else:
        h_delta = float(t_cell["macro_f1"]["mean"]) - float(w_cell["macro_f1"]["mean"])
        headline_verdict = {
            "regime": headline_regime,
            "head": headline_cell,
            "text_macro_f1": float(t_cell["macro_f1"]["mean"]),
            "wavlm_macro_f1": float(w_cell["macro_f1"]["mean"]),
            **_verdict_from_delta(h_delta),
        }

    return {
        "text_modality": text_results.get("modality"),
        "text_encoder_id": text_results.get("encoder_id"),
        "wavlm_embeddings_path": wavlm_results.get("embeddings_path"),
        "matched_pair_rows": rows,
        "best_vs_best": best_vs_best,
        "headline_cell_verdict": headline_verdict,
        "negative_calibration": calibration,
        "note": (
            "Primary verdict is best_vs_best. headline_cell_verdict is secondary "
            "(fixed head, often overstates the gap when that head is WavLM-best / text-worst)."
        ),
    }


def print_comparison(comparison: dict) -> None:
    """Print comparison tables and verdicts to stdout.

    Args:
        comparison: Payload from compare_text_vs_wavlm.

    Returns:
        None
    """
    print("\n[compare] matched-pair (balanced first)")
    print(
        f"{'regime':<10} {'head':<16} {'t_mac':>7} {'w_mac':>7} {'d_mac':>7} "
        f"{'d_posF1':>8} {'d_neuF1':>8} {'d_negF1':>8}"
    )
    for row in comparison["matched_pair_rows"]:
        d = row["per_class_f1_delta"]
        print(
            f"{row['regime']:<10} {row['head']:<16} "
            f"{row['text_macro_f1']:7.3f} {row['wavlm_macro_f1']:7.3f} "
            f"{row['delta_macro_f1_text_minus_wavlm']:7.3f} "
            f"{d['positive']:8.3f} {d['neutral']:8.3f} {d['negative']:8.3f}"
        )

    bvb = comparison["best_vs_best"]
    print(
        f"\n[best-vs-best] regime={bvb['regime']} "
        f"text={bvb['text_head']} ({bvb['text_macro_f1']:.3f}) vs "
        f"wavlm={bvb['wavlm_head']} ({bvb['wavlm_macro_f1']:.3f}) "
        f"Δ={bvb['delta_macro_f1_text_minus_wavlm']:+.3f}"
    )
    print(f"  per-class ΔF1: {bvb['per_class_f1_delta']}")
    print(f"  verdict: {bvb['verdict']['status']} — {bvb['verdict']['interpretation']}")

    hv = comparison["headline_cell_verdict"]
    print(f"\n[headline-cell] {hv}")

    print("\n[calibration] neg pred/true ≈ R/P (balanced)")
    for row in comparison["negative_calibration"]:
        if row["regime"] != "balanced":
            continue
        print(
            f"  {row['head']:<16} "
            f"text R/P={row['text_neg_recall']:.3f}/{row['text_neg_precision']:.3f} "
            f"→ {row['text_neg_pred_over_true']:.2f}x  |  "
            f"wavlm R/P={row['wavlm_neg_recall']:.3f}/{row['wavlm_neg_precision']:.3f} "
            f"→ {row['wavlm_neg_pred_over_true']:.2f}x"
        )


def main() -> None:
    """CLI entrypoint for text vs WavLM comparison.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Compare text CV JSON vs WavLM")
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--wavlm", type=Path, default=WAVLM_CV_GROUP_QUESTION_JSON)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--regime", default="balanced")
    parser.add_argument("--headline-cell", default="rbf_svc")
    args = parser.parse_args()

    with args.text.open(encoding="utf-8") as f:
        text_results = json.load(f)
    with args.wavlm.open(encoding="utf-8") as f:
        wavlm_results = json.load(f)

    comparison = compare_text_vs_wavlm(
        text_results,
        wavlm_results,
        headline_regime=args.regime,
        headline_cell=args.headline_cell,
    )
    print_comparison(comparison)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2)
        print(f"[wrote] {args.output}")


if __name__ == "__main__":
    main()
