#!/usr/bin/env python3
"""
Authors:
    Steven Liem (steven.liem@sydney.edu.au)

Extract and cache utterance-level emotion2vec embeddings for our_speech_corpus.
The full pipeline is as follows:
  1. Load our_speech_corpus_manifest_split_80_20.csv
  2. Filter to requested splits (default: train + test)
  3. Load emotion2vec_plus_large via FunASR
  4. Extract utterance embeddings with extract_embedding=True
  5. Write our_speech_corpus_embeddings.npz keyed by clip_id
  6. Refresh our_speech_corpus_qc_report.json with an embedding_qc section

Requires Python 3.12 (funasr does not support 3.14). Example setup:

    uv python install 3.12
    uv venv --python 3.12 .venv312
    uv pip install --python .venv312/bin/python funasr torch torchaudio pandas numpy
    .venv312/bin/python src/our_speech_corpus_final/extract_emotion2vec_embeddings.py

Usage:
    .venv312/bin/python src/our_speech_corpus_final/extract_emotion2vec_embeddings.py
    .venv312/bin/python src/our_speech_corpus_final/extract_emotion2vec_embeddings.py --limit 50
    .venv312/bin/python src/our_speech_corpus_final/extract_emotion2vec_embeddings.py --splits train test

References:
    https://huggingface.co/emotion2vec/emotion2vec_plus_large
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
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import bootstrap_imports, extend_our_speech_corpus_paths

bootstrap_imports(__file__)
extend_our_speech_corpus_paths()
from our_speech_corpus_manifest_report import write_qc_report
from config import (
    EMBEDDINGS_NPZ,
    MODEL_ID,
    QC_REPORT_JSON,
    SPLIT_MANIFEST_CSV,
    SPLIT_COLUMN,
    UNIFIED_MANIFEST_CSV,
)


def pick_device(explicit: str | None = None) -> str:
    """Choose CUDA if available, otherwise CPU.

    Args:
        explicit: Optional device string such as ``cuda:0`` or ``cpu``. If
            omitted or None, CUDA is used when available.

    Returns:
        str: Device identifier for FunASR ``AutoModel``.
    """
    if explicit:
        return explicit
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    return "cpu"


def load_model(device: str):
    """Load emotion2vec_plus_large via FunASR.

    Args:
        device: Device string such as ``cuda:0`` or ``cpu``.

    Returns:
        FunASR AutoModel instance for utterance-level embedding extraction.
    """
    from funasr import AutoModel

    hub = os.environ.get("EMOTION2VEC_HUB", "hf")
    print(f"[embeddings] loading {MODEL_ID} on {device} (hub={hub})")
    return AutoModel(model=MODEL_ID, hub=hub, device=device, disable_update=True)


def parse_embedding(result: object) -> np.ndarray | None:
    """Parse FunASR ``generate()`` output into a flat float32 embedding vector.

    Args:
        result: Raw FunASR generate return value (dict or list of dicts).

    Returns:
        np.ndarray | None: 1-D embedding, or None if no embedding field is found.
    """
    item = result[0] if isinstance(result, list) and result else result
    if not isinstance(item, dict):
        return None

    for key in ("embedding", "embeddings", "utt_embedding", "feats"):
        if key in item and item[key] is not None:
            arr = np.asarray(item[key], dtype=np.float32)
            return arr.reshape(-1)

    return None


def extract_one(model, wav_path: Path) -> np.ndarray | None:
    """Run utterance-level emotion2vec embedding extraction on one WAV.

    Args:
        model: Loaded FunASR AutoModel.
        wav_path: Path to the WAV file.

    Returns:
        np.ndarray | None: Flat embedding vector, or None on parse failure.
    """
    result = model.generate(
        input=str(wav_path),
        granularity="utterance",
        extract_embedding=True,
    )
    return parse_embedding(result)


def extract_embeddings(
    manifest_path: Path = SPLIT_MANIFEST_CSV,
    output_path: Path = EMBEDDINGS_NPZ,
    device: str | None = None,
    limit: int | None = None,
    splits: tuple[str, ...] | None = None,
    unified_path: Path | None = None,
    qc_report_path: Path | None = None,
) -> dict:
    """Extract embeddings for selected splits and write a compressed NPZ cache.

    Args:
        manifest_path: Split manifest CSV (must include clip_id, audio_path, split).
        output_path: Destination for our_speech_corpus_embeddings.npz.
        device: Optional device override for FunASR.
        limit: Optional max number of rows (for smoke tests).
        splits: Split values to include (e.g. ``("train", "test")``). None keeps all.
        unified_path: Optional unified manifest for QC (defaults to pipeline_config).
        qc_report_path: Optional QC JSON path (defaults to pipeline_config).

    Returns:
        dict: Embedding QC summary (counts, dim, failure samples).

    Raises:
        RuntimeError: If no embeddings are successfully extracted.
    """
    df = pd.read_csv(manifest_path)
    if splits:
        df = df[df[SPLIT_COLUMN].isin(splits)].copy()

    if limit is not None:
        df = df.head(limit)

    model = load_model(pick_device(device))

    clip_ids: list[str] = []
    embeddings: list[np.ndarray] = []
    missing_audio: list[str] = []
    failed: list[str] = []

    for _, row in df.iterrows():
        clip_id = str(row["clip_id"])
        wav = Path(str(row["audio_path"]))
        if not wav.is_file():
            missing_audio.append(clip_id)
            continue

        try:
            emb = extract_one(model, wav)
        except Exception:
            failed.append(clip_id)
            continue

        if emb is None or emb.size == 0 or not np.isfinite(emb).all():
            failed.append(clip_id)
            continue

        clip_ids.append(clip_id)
        embeddings.append(emb)

        if (len(clip_ids) % 50) == 0:
            print(f"[embeddings] processed {len(clip_ids)}/{len(df)}")

    if not embeddings:
        raise RuntimeError("No embeddings extracted")

    emb_matrix = np.vstack(embeddings)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        clip_id=np.array(clip_ids, dtype=object),
        embeddings=emb_matrix,
        embedding_dim=np.array([emb_matrix.shape[1]], dtype=np.int32),
        model_id=np.array([MODEL_ID], dtype=object),
    )

    summary = {
        "embeddings_file": str(output_path),
        "model_id": MODEL_ID,
        "requested_rows": int(len(df)),
        "extracted": int(len(clip_ids)),
        "missing_audio": int(len(missing_audio)),
        "failed": int(len(failed)),
        "embedding_dim": int(emb_matrix.shape[1]),
        "missing_audio_clip_ids_sample": missing_audio[:20],
        "failed_clip_ids_sample": failed[:20],
    }

    qc_out = qc_report_path or QC_REPORT_JSON
    split_for_qc = manifest_path
    unified_for_qc = unified_path or UNIFIED_MANIFEST_CSV
    try:
        write_qc_report(
            output_path=qc_out,
            unified_path=unified_for_qc,
            split_path=split_for_qc,
            embedding_summary=summary,
        )
    except FileNotFoundError as exc:
        print(f"[embeddings] skipped QC refresh ({exc})")

    print(
        f"[embeddings] wrote {output_path} "
        f"({summary['extracted']} vectors, dim={summary['embedding_dim']})"
    )
    return summary


def main() -> None:
    """CLI entrypoint: extract embeddings for selected splits.

    Args:
        None. Command-line flags: ``--manifest``, ``--unified``, ``--output``,
        ``--qc-report``, ``--device``, ``--limit``, and ``--splits``.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Extract emotion2vec embeddings")
    parser.add_argument("--manifest", type=Path, default=SPLIT_MANIFEST_CSV)
    parser.add_argument("--unified", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=EMBEDDINGS_NPZ)
    parser.add_argument("--qc-report", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--splits",
        nargs="*",
        default=["train", "test"],
        help="Split values to include (default: train test)",
    )
    args = parser.parse_args()

    extract_embeddings(
        manifest_path=args.manifest,
        output_path=args.output,
        device=args.device,
        limit=args.limit,
        splits=tuple(args.splits) if args.splits else None,
        unified_path=args.unified,
        qc_report_path=args.qc_report,
    )


if __name__ == "__main__":
    main()
