#!/usr/bin/env python3
"""
Authors:
    Steven Liem (steven.liem@sydney.edu.au)

Experiment 3 — frozen all-MiniLM-L6-v2 text encoder + audio-matched valence head CV.

Fair like-for-like comparison: general-purpose frozen text encoder vs frozen WavLM.

Usage:
    uv run python src/our_speech_corpus_final/heads/train_minilm_valence_head_cv.py \\
        --manifest data/our_speech_corpus_manifest_split_80_20.csv \\
        --group-by question \\
        --compare-wavlm \\
        --output data/our_speech_corpus_text/reports/valence_head_cv_minilm_group_question.json
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

import numpy as np

from config import bootstrap_imports, extend_our_speech_corpus_paths

bootstrap_imports(__file__)
extend_our_speech_corpus_paths()
from config import (  # noqa: E402
    MINILM_CV_GROUP_QUESTION_JSON,
    MINILM_EMBEDDINGS_NPZ,
    MINILM_MODEL_ID,
    SPLIT_MANIFEST_CSV,
    SPLIT_SEED,
    WAVLM_CV_GROUP_QUESTION_JSON,
)
from text_baseline_common import (  # noqa: E402
    evaluate_embedding_grid,
    join_embeddings_by_clip_id,
    labels_from_frame,
    load_labelled_manifest,
    set_seeds,
)


def extract_minilm_embeddings(
    frame,
    output_path: Path,
    model_id: str = MINILM_MODEL_ID,
    batch_size: int = 64,
) -> Path:
    """Encode response_text with SentenceTransformer; write NPZ.

    Args:
        frame: Labelled manifest with ``_norm_text`` and ``clip_id``.
        output_path: Destination NPZ.
        model_id: sentence-transformers model id.
        batch_size: Encode batch size.

    Returns:
        Path: ``output_path``.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_id)
    texts = frame["_norm_text"].tolist()
    emb = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=True,
    )
    emb = np.asarray(emb, dtype=np.float32)
    assert emb.shape == (len(frame), 384), emb.shape
    clip_ids = frame["clip_id"].astype(str).to_numpy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, clip_id=clip_ids, embeddings=emb)
    print(f"[wrote] {output_path} shape={emb.shape}")
    return output_path


def main() -> None:
    """CLI for MiniLM embedding extract + matched valence-head CV.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description="Frozen MiniLM text encoder valence-head CV"
    )
    parser.add_argument("--manifest", type=Path, default=SPLIT_MANIFEST_CSV)
    parser.add_argument("--embeddings", type=Path, default=MINILM_EMBEDDINGS_NPZ)
    parser.add_argument("--output", type=Path, default=MINILM_CV_GROUP_QUESTION_JSON)
    parser.add_argument("--model-id", default=MINILM_MODEL_ID)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    parser.add_argument(
        "--group-by",
        choices=("response_text", "question"),
        default=None,
    )
    parser.add_argument("--only", nargs="+", default=None)
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Only write embeddings NPZ; skip CV",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Reuse existing --embeddings NPZ",
    )
    parser.add_argument(
        "--compare-wavlm",
        type=Path,
        nargs="?",
        const=WAVLM_CV_GROUP_QUESTION_JSON,
        default=None,
    )
    args = parser.parse_args()

    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)

    set_seeds(args.seed)
    frame = load_labelled_manifest(args.manifest)
    y = labels_from_frame(frame)

    if not args.skip_extract:
        extract_minilm_embeddings(
            frame,
            args.embeddings,
            model_id=args.model_id,
            batch_size=args.batch_size,
        )
    elif not args.embeddings.is_file():
        raise FileNotFoundError(args.embeddings)

    if args.extract_only:
        return

    X = join_embeddings_by_clip_id(frame, args.embeddings, emb_key="embeddings")
    results = evaluate_embedding_grid(
        X=X,
        y=y,
        frame=frame,
        modality="text_minilm",
        encoder_id=args.model_id,
        embeddings_path=args.embeddings,
        manifest_path=args.manifest,
        output_path=args.output,
        folds=args.folds,
        repeats=args.repeats,
        seed=args.seed,
        group_by=args.group_by,
        only=tuple(args.only) if args.only else None,
        extra_meta={
            "comparison_note": (
                "Fair comparison: general-purpose frozen text encoder vs "
                "general-purpose frozen WavLM. MiniLM is the headline text baseline."
            ),
        },
    )

    if args.compare_wavlm is not None:
        from compare_text_vs_wavlm import compare_text_vs_wavlm, print_comparison

        if not args.compare_wavlm.is_file():
            print(
                f"[compare] SKIP — WavLM JSON not found: {args.compare_wavlm}\n"
                "  Upload valence_head_cv_wavlm_group_question.json to the dataset, "
                "or run compare_text_vs_wavlm.py later. CV results already written."
            )
        else:
            with args.compare_wavlm.open(encoding="utf-8") as f:
                wavlm = json.load(f)
            comparison = compare_text_vs_wavlm(results, wavlm)
            print_comparison(comparison)
            results["comparison_to_wavlm"] = comparison
            with args.output.open("w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            print(f"[updated] {args.output} with comparison_to_wavlm")


if __name__ == "__main__":
    main()
