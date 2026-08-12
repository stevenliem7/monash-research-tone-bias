#!/usr/bin/env python3
"""
Authors:
    Steven Liem (steven.liem@sydney.edu.au)

Experiment 1 — DistilBERT-SST2 frozen encoder (discard SST-2 head) + valence CV.

Hardest text baseline: sentiment-specialised transformer embeddings vs general-
purpose WavLM. Mean-pool is primary; CLS is a robustness check.

Usage:
    uv run python src/our_speech_corpus_final/heads/train_distilbert_valence_head_cv.py \\
        --manifest data/our_speech_corpus_manifest_split_80_20.csv \\
        --group-by question --pooling mean --compare-wavlm \\
        --output data/our_speech_corpus_text/reports/valence_head_cv_distilbert_group_question.json
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
    DISTILBERT_CV_GROUP_QUESTION_JSON,
    DISTILBERT_EMBEDDINGS_NPZ,
    DISTILBERT_SST2_MODEL_ID,
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


def mean_pool(last_hidden: "torch.Tensor", attention_mask: "torch.Tensor") -> "torch.Tensor":
    """Mean-pool last hidden state over non-padding tokens.

    Args:
        last_hidden: (batch, seq, hidden).
        attention_mask: (batch, seq) 1=token, 0=pad.

    Returns:
        torch.Tensor: (batch, hidden).
    """
    import torch

    mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def extract_distilbert_embeddings(
    frame,
    output_path: Path,
    model_id: str = DISTILBERT_SST2_MODEL_ID,
    batch_size: int = 32,
    max_length: int = 128,
    device: str | None = None,
) -> dict:
    """Extract mean-pooled and CLS embeddings; discard classification head.

    Args:
        frame: Labelled manifest with ``_norm_text``.
        output_path: Destination NPZ.
        model_id: HF DistilBERT-SST2 id (loaded as AutoModel).
        batch_size: Inference batch size.
        max_length: Tokenizer max length.
        device: Optional torch device.

    Returns:
        dict: truncation_rate and shape metadata.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(model_id)
    # AutoModel loads the encoder without the SST-2 classification head.
    model = AutoModel.from_pretrained(model_id)
    model.to(device)
    model.eval()

    texts = frame["_norm_text"].tolist()
    n = len(texts)
    hidden = int(model.config.hidden_size)
    mean_emb = np.zeros((n, hidden), dtype=np.float32)
    cls_emb = np.zeros((n, hidden), dtype=np.float32)
    n_truncated = 0

    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch = texts[start : start + batch_size]
            # Length check without truncation for rate logging
            for t in batch:
                ids = tok.encode(t, add_special_tokens=True, truncation=False)
                if len(ids) > max_length:
                    n_truncated += 1
            enc = tok(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            last = out.last_hidden_state
            pooled = mean_pool(last, enc["attention_mask"])
            cls = last[:, 0, :]
            mean_emb[start : start + len(batch)] = pooled.cpu().numpy()
            cls_emb[start : start + len(batch)] = cls.cpu().numpy()

    trunc_rate = n_truncated / n
    print(
        f"[truncation] {n_truncated}/{n} ({100 * trunc_rate:.2f}%) exceed "
        f"max_length={max_length}"
    )
    if trunc_rate >= 0.01:
        print("[truncation] WARNING: truncation rate ≥ 1%")

    clip_ids = frame["clip_id"].astype(str).to_numpy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        clip_id=clip_ids,
        embeddings_mean=mean_emb,
        embeddings_cls=cls_emb,
    )
    print(
        f"[wrote] {output_path} mean={mean_emb.shape} cls={cls_emb.shape}"
    )
    return {
        "truncation_rate": float(trunc_rate),
        "n_truncated": int(n_truncated),
        "max_length": max_length,
        "hidden_size": hidden,
    }


def main() -> None:
    """CLI for DistilBERT encoder extract + matched valence-head CV.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description="Frozen DistilBERT-SST2 encoder valence-head CV (hardest text baseline)"
    )
    parser.add_argument("--manifest", type=Path, default=SPLIT_MANIFEST_CSV)
    parser.add_argument("--embeddings", type=Path, default=DISTILBERT_EMBEDDINGS_NPZ)
    parser.add_argument("--output", type=Path, default=DISTILBERT_CV_GROUP_QUESTION_JSON)
    parser.add_argument("--model-id", default=DISTILBERT_SST2_MODEL_ID)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument(
        "--pooling",
        choices=("mean", "cls"),
        default="mean",
        help="Primary=mean; cls is robustness check",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    parser.add_argument(
        "--group-by",
        choices=("response_text", "question"),
        default=None,
    )
    parser.add_argument("--only", nargs="+", default=None)
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--skip-extract", action="store_true")
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

    extract_meta: dict = {}
    if not args.skip_extract:
        extract_meta = extract_distilbert_embeddings(
            frame,
            args.embeddings,
            model_id=args.model_id,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
    elif not args.embeddings.is_file():
        raise FileNotFoundError(args.embeddings)

    if args.extract_only:
        return

    emb_key = "embeddings_mean" if args.pooling == "mean" else "embeddings_cls"
    X = join_embeddings_by_clip_id(frame, args.embeddings, emb_key=emb_key)
    results = evaluate_embedding_grid(
        X=X,
        y=y,
        frame=frame,
        modality=f"text_distilbert_{args.pooling}",
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
            "pooling": args.pooling,
            "reported_primary_pooling": "mean",
            "extract_meta": extract_meta,
            "comparison_note": (
                "Hardest text baseline: DistilBERT is sentiment-tuned; WavLM is "
                "general-purpose. Slightly unfair to WavLM. MiniLM is the fair comparison."
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
