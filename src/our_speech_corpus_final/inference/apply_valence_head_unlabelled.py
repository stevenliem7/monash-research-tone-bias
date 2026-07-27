#!/usr/bin/env python3
"""
Authors:
    Steven Liem (steven.liem@sydney.edu.au)

Apply valence head(s) to the unlabelled pool — reported by stratum.

Do NOT pool Corpus A + Corpus B into one distribution:
  - Corpus A unlabelled (mixed prompt tones) → baseline tone bias / RQ2
  - Corpus B unlabelled (negative-conditioned prompts) → override rate at scale

Default heads (post group-CV handoff): rbf_svc balanced (primary extend) and
l2_logistic balanced (robustness check; different family, avoids pca32 nesting).

Requires embeddings covering unlabeled_pool. Extract on Kaggle:

    !python .../run_ssl_backbone_kaggle.py \\
      --model wavlm-large \\
      --corpora-root .../our_speech_corpus_final/our_speech_corpus_final \\
      --manifest .../our_speech_corpus_manifest_split_80_20.csv \\
      --splits unlabeled_pool \\
      --output-dir /kaggle/working/ssl_wavlm_pool

Usage:
    python apply_valence_head_unlabelled.py \\
      --embeddings labelled.npz pool.npz \\
      --manifest .../our_speech_corpus_manifest_split_80_20.csv \\
      --candidates rbf_svc l2_logistic --balanced
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
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

from config import bootstrap_imports, extend_our_speech_corpus_paths

bootstrap_imports(__file__)
extend_our_speech_corpus_paths()
from config import (  # noqa: E402
    EMBEDDINGS_NPZ,
    SOURCE_BATCH_A,
    SOURCE_BATCH_B,
    SPLIT_MANIFEST_CSV,
    SPLIT_SEED,
    VALENCE_CLASSES,
)
from train_valence_head_cv import (  # noqa: E402
    ID_TO_LABEL,
    LABEL_TO_ID,
    build_candidates,
    oversample_to_balance,
)


def load_embedding_lookup(paths: list[Path]) -> dict[str, np.ndarray]:
    """Merge one or more NPZs into a clip_id-to-vector lookup.

    Args:
        paths: List of embedding NPZ files (later files overwrite clip_ids).

    Returns:
        dict[str, np.ndarray]: Mapping from clip_id to float32 embedding vector.
    """
    lookup: dict[str, np.ndarray] = {}
    for path in paths:
        bundle = np.load(path, allow_pickle=True)
        ids = [str(x) for x in bundle["clip_id"].tolist()]
        mat = np.asarray(bundle["embeddings"], dtype=np.float32)
        for i, cid in enumerate(ids):
            lookup[cid] = mat[i]
        print(f"[emb] {path.name}: {len(ids)} vectors (lookup now {len(lookup)})")
    return lookup


def stratum_summary(frame: pd.DataFrame, preds: np.ndarray) -> dict:
    """Summarise predicted valence distribution for one stratum.

    Args:
        frame: Manifest rows for the stratum (may include emotion_label).
        preds: Integer predicted class ids aligned with ``frame``.

    Returns:
        dict: Counts, proportions, and optional per-emotion breakdown.
    """
    n = len(preds)
    counts = {lab: int((preds == LABEL_TO_ID[lab]).sum()) for lab in VALENCE_CLASSES}
    props = {lab: (counts[lab] / n if n else 0.0) for lab in VALENCE_CLASSES}
    return {
        "n": n,
        "predicted_counts": counts,
        "predicted_proportions": props,
        "by_emotion_label": _by_emotion(frame, preds),
    }


def _by_emotion(frame: pd.DataFrame, preds: np.ndarray) -> dict:
    """Break down predicted valence counts by emotion_label within a stratum.

    Args:
        frame: Manifest rows with optional ``emotion_label`` column.
        preds: Integer predicted class ids aligned with ``frame``.

    Returns:
        dict: Per-emotion counts and proportions, or empty if column missing.
    """
    if "emotion_label" not in frame.columns:
        return {}
    out: dict = {}
    emotions = frame["emotion_label"].fillna("unknown").astype(str).to_numpy()
    for emo in sorted(set(emotions)):
        mask = emotions == emo
        sub = preds[mask]
        n = int(mask.sum())
        counts = {lab: int((sub == LABEL_TO_ID[lab]).sum()) for lab in VALENCE_CLASSES}
        out[emo] = {
            "n": n,
            "predicted_counts": counts,
            "predicted_proportions": {
                lab: (counts[lab] / n if n else 0.0) for lab in VALENCE_CLASSES
            },
        }
    return out


def apply_one(
    name: str,
    cand: dict,
    X_lab: np.ndarray,
    y_lab: np.ndarray,
    pool: pd.DataFrame,
    pool_vecs: np.ndarray,
    balanced: bool,
    seed: int,
    output_dir: Path,
) -> dict:
    """Fit one candidate on labelled data and predict the unlabelled pool.

    Args:
        name: Candidate head name (e.g. ``rbf_svc``).
        cand: Entry from :func:`build_candidates`.
        X_lab: Labelled training embeddings.
        y_lab: Labelled training integer labels.
        pool: Unlabelled pool manifest rows.
        pool_vecs: Embeddings aligned with ``pool``.
        balanced: Whether balanced class weighting was used.
        seed: RNG seed for optional oversampling.
        output_dir: Directory for per-candidate prediction CSV.

    Returns:
        dict: Stratum summaries and paths for the candidate run.
    """
    X_fit, y_fit = X_lab, y_lab
    if cand["oversample"]:
        X_fit, y_fit = oversample_to_balance(
            X_fit, y_fit, np.random.default_rng(seed)
        )
    model = clone(cand["model"])
    model.fit(X_fit, y_fit)
    pred = model.predict(pool_vecs)
    pred_label = np.array([ID_TO_LABEL[i] for i in pred])

    out_frame = pool.copy()
    out_frame["predicted_valence"] = pred_label
    out_frame["candidate"] = name
    out_frame["balanced"] = balanced
    csv_path = output_dir / f"unlabelled_pool_predictions_{name}.csv"
    out_frame.to_csv(csv_path, index=False)

    strata = {
        "corpusA_unlabelled_mixed_prompts": pool["source_batch"] == SOURCE_BATCH_A,
        "corpusB_unlabelled_negative_conditioned": pool["source_batch"]
        == SOURCE_BATCH_B,
    }
    report: dict = {
        "candidate": name,
        "balanced": balanced,
        "predictions_csv": str(csv_path),
        "strata": {},
        "pooled_all_unlabelled_DO_NOT_USE_FOR_RQ2": stratum_summary(pool, pred),
    }
    print(f"\n[{name}] strata (report these, not the pooled row)")
    for sname, mask in strata.items():
        sub = pool.loc[mask].reset_index(drop=True)
        sub_pred = pred[mask.to_numpy()]
        summary = stratum_summary(sub, sub_pred)
        report["strata"][sname] = {
            "source_batch": SOURCE_BATCH_A if "corpusA" in sname else SOURCE_BATCH_B,
            "role": (
                "RQ2 baseline tone bias (mixed prompt tones)"
                if "corpusA" in sname
                else "override finding at scale (all negative-conditioned)"
            ),
            **summary,
        }
        props = summary["predicted_proportions"]
        print(
            f"  {sname}: n={summary['n']}  "
            f"pos={props['positive']:.3f}  neu={props['neutral']:.3f}  "
            f"neg={props['negative']:.3f}"
        )
    print(f"[wrote] {csv_path}")
    return report


def main() -> None:
    """CLI entrypoint: fit on labelled clips and predict the unlabelled pool.

    Args:
        None. Command-line flags: ``--embeddings``, ``--manifest``, ``--candidates``,
        ``--balanced``, ``--seed``, and ``--output-dir``.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description="Apply valence head(s) to unlabelled pool (stratified)"
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        nargs="+",
        default=[EMBEDDINGS_NPZ],
        help="NPZs covering labelled + unlabeled_pool",
    )
    parser.add_argument("--manifest", type=Path, default=SPLIT_MANIFEST_CSV)
    parser.add_argument(
        "--candidates",
        nargs="+",
        default=["rbf_svc", "l2_logistic"],
        help="Heads to apply (default: rbf_svc l2_logistic)",
    )
    parser.add_argument(
        "--balanced",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use balanced class weights (default: true)",
    )
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/our_speech_corpus/unlabelled_pool_preds"),
    )
    args = parser.parse_args()

    for p in args.embeddings:
        if not p.is_file():
            raise FileNotFoundError(f"Embeddings not found: {p}")
    if not args.manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    lookup = load_embedding_lookup(list(args.embeddings))
    manifest = pd.read_csv(args.manifest)
    labelled = manifest[manifest["is_labelled"]].copy()
    lab_vecs, lab_y = [], []
    for _, row in labelled.iterrows():
        cid = str(row["clip_id"])
        gt = str(row["ground_truth_label"]).strip().lower()
        if cid not in lookup or gt not in LABEL_TO_ID:
            continue
        lab_vecs.append(lookup[cid])
        lab_y.append(LABEL_TO_ID[gt])
    if not lab_vecs:
        raise ValueError("No labelled rows matched embeddings")
    X_lab = np.vstack(lab_vecs)
    y_lab = np.asarray(lab_y, dtype=np.int64)
    print(
        f"[fit] labelled n={len(y_lab)}  "
        f"classes={dict(Counter(ID_TO_LABEL[i] for i in y_lab))}"
    )

    pool = manifest[manifest["split"] == "unlabeled_pool"].copy()
    missing = [str(c) for c in pool["clip_id"] if str(c) not in lookup]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)}/{len(pool)} unlabeled_pool clips lack embeddings "
            f"(e.g. {missing[:3]}). Extract the pool first:\n"
            "  python extract_hf_ssl_embeddings.py --model wavlm-large \\\n"
            "    --manifest ... --splits unlabeled_pool \\\n"
            "    --output wavlm_unlabelled_pool.npz\n"
            "Then: --embeddings labelled.npz pool.npz"
        )

    pool_vecs = np.vstack([lookup[str(c)] for c in pool["clip_id"]])
    cands = build_candidates(args.balanced, args.seed)
    unknown = [c for c in args.candidates if c not in cands]
    if unknown:
        raise ValueError(f"Unknown candidate(s) {unknown}; known={sorted(cands)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "embeddings": [str(p) for p in args.embeddings],
        "manifest_path": str(args.manifest),
        "balanced": args.balanced,
        "n_labelled_fit": int(len(y_lab)),
        "n_pool": int(len(pool)),
        "note": (
            "Human-labelled rates remain primary; classifier extends to the "
            "unlabelled pool. Report Corpus A and Corpus B strata separately — "
            "never pool (B is ~71% and entirely negative-conditioned)."
        ),
        "candidates": {},
    }
    for name in args.candidates:
        payload["candidates"][name] = apply_one(
            name,
            cands[name],
            X_lab,
            y_lab,
            pool,
            pool_vecs,
            args.balanced,
            args.seed,
            args.output_dir,
        )

    json_path = args.output_dir / "unlabelled_pool_strata_report.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[wrote] {json_path}")


if __name__ == "__main__":
    main()
