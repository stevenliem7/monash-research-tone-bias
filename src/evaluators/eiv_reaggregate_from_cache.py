"""
Authors:
    Steven Liem (steven.liem@sydney.edu.au)

One-shot: re-aggregate EIV valence predictions from cached raw_* scores.

Use this after changing EMOTION_HEADS polarity / head set, or the mass rule
(count-normalized polarity argmax). Does NOT re-run Whisper or MLPs.

Updates per corpus:
  predictions.csv  — pred_top1, pred_mass, pos/neu/neg mass, top_emotion, probs
  metrics.json     — valence_head / top1 / mass metrics vs prompted_valence

Then run:
  uv run python src/evaluators/eiv_summary_table.py
  uv run python src/evaluators/eiv_all_emotions.py

Usage:
    uv run python src/evaluators/eiv_reaggregate_from_cache.py
    uv run python src/evaluators/eiv_reaggregate_from_cache.py --corpus our_speech_corpus_cleaned
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
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

from config import EIV_ALL36_SCRIPT, RESULTS_EIV, bootstrap_imports

bootstrap_imports(__file__)

DEFAULT_RESULTS = RESULTS_EIV
ALL36_PATH = EIV_ALL36_SCRIPT

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


def load_all36():
    """Import empathic_insight_voice_all36_kaggle.py by path.

    Args:
        None

    Returns:
        module: Loaded all36 module with EMOTION_HEADS + aggregate helpers.
    """
    spec = importlib.util.spec_from_file_location("eiv_all36", ALL36_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {ALL36_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["eiv_all36"] = mod
    spec.loader.exec_module(mod)
    return mod


def reaggregate_frame(
    df: pd.DataFrame,
    eiv,
    neutral_band: float,
) -> pd.DataFrame:
    """Rewrite aggregation columns from raw_* using current EMOTION_HEADS.

    Args:
        df: Cached predictions with raw_<stem> columns.
        eiv: Loaded all36 module.
        neutral_band: Band for pred_valence_head from raw_valence.

    Returns:
        pd.DataFrame: Updated predictions.
    """
    stems = list(eiv.EMOTION_HEADS.keys())
    missing = [s for s in stems if f"raw_{s}" not in df.columns]
    if missing:
        raise KeyError(
            f"predictions.csv missing raw columns for current heads: {missing[:5]}..."
        )

    out = df.copy()
    # Drop stale per-head prob columns from removed heads; rewrite below.
    drop_probs = [c for c in out.columns if c.startswith("prob_")]
    out = out.drop(columns=drop_probs, errors="ignore")

    rows = []
    for record in out.to_dict(orient="records"):
        score_row = {s: float(record[f"raw_{s}"]) for s in stems}
        agg = eiv.aggregate_emotions(score_row)
        rows.append(agg)

    agg_df = pd.DataFrame(rows)
    for col in agg_df.columns:
        out[col] = agg_df[col].to_numpy()

    if "raw_valence" in out.columns:
        out["pred_valence_head"] = [
            eiv.bin_valence(float(v), neutral_band) for v in out["raw_valence"]
        ]

    # Prefer prompted_valence naming; keep legacy alias if present.
    if "prompted_valence" not in out.columns and "ground_truth_valence" in out.columns:
        out = out.rename(columns={"ground_truth_valence": "prompted_valence"})

    return out


def main() -> None:
    """Re-aggregate all (or selected) corpora from cached raw scores.

    Args:
        None

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--corpus", action="append", default=None)
    parser.add_argument("--neutral-band", type=float, default=0.5)
    args = parser.parse_args()

    eiv = load_all36()
    print(
        f"[reagg] EMOTION_HEADS = {len(eiv.EMOTION_HEADS)} "
        f"(pos={len(eiv.POS_HEADS)} neu={len(eiv.NEU_HEADS)} neg={len(eiv.NEG_HEADS)})"
    )
    print("[reagg] mass rule = count_normalized_argmax")

    corpora = args.corpus or list(CORPUS_DIRS)
    for corpus in corpora:
        pred_path = args.results_root / corpus / "predictions.csv"
        if not pred_path.exists():
            print(f"[skip] missing {pred_path}")
            continue

        print(f"[reagg] {corpus}")
        df = pd.read_csv(pred_path)
        df = reaggregate_frame(df, eiv, args.neutral_band)
        if "corpus" not in df.columns:
            df.insert(0, "corpus", corpus)

        metrics = eiv.summarize_corpus(df, args.neutral_band)
        out_dir = args.results_root / corpus
        df.to_csv(pred_path, index=False)
        with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"[reagg] wrote {pred_path}")
        print(f"[reagg] wrote {out_dir / 'metrics.json'}")

    print("\n[reagg] done. Next:")
    print("  uv run python src/evaluators/eiv_summary_table.py")
    print("  uv run python src/evaluators/eiv_all_emotions.py")


if __name__ == "__main__":
    main()
