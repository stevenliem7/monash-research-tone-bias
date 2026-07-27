#!/usr/bin/env python3
"""
Authors:
    Steven Liem (steven.liem@sydney.edu.au)

Extract utterance-level embeddings from HuggingFace SSL speech encoders
(wav2vec2-large-robust, WavLM-large) for the our_speech_corpus valence head.

The full pipeline is as follows:
  1. Load the split manifest (clip_id, audio_path, split)
  2. Filter to requested splits (default: train + test = labelled set)
  3. Load the HF model + feature extractor via transformers
  4. For each WAV: resample to 16 kHz mono, forward pass, mean-pool last hidden
  5. Write an NPZ with the same schema as emotion2vec embeddings
     (clip_id, embeddings, embedding_dim, model_id)

Output joins unchanged into train_valence_head_cv.py / evaluate_valence_head_cv.py
via --embeddings.

Usage (local / Kaggle):
    python extract_hf_ssl_embeddings.py \\
        --model facebook/wav2vec2-large-robust \\
        --manifest path/to/split.csv \\
        --output /kaggle/working/wav2vec2_large_robust.npz

    python extract_hf_ssl_embeddings.py \\
        --model microsoft/wavlm-large \\
        --manifest path/to/split.csv \\
        --output /kaggle/working/wavlm_large.npz

References:
    https://huggingface.co/facebook/wav2vec2-large-robust
    https://huggingface.co/microsoft/wavlm-large
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
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import bootstrap_imports, extend_our_speech_corpus_paths

bootstrap_imports(__file__)
extend_our_speech_corpus_paths()
from config import SPLIT_COLUMN, SPLIT_MANIFEST_CSV

# Convenience aliases for CLI --model shortcuts
MODEL_ALIASES = {
    "wav2vec2-large-robust": "facebook/wav2vec2-large-robust",
    "wavlm-large": "microsoft/wavlm-large",
    "wavlm-base-plus": "microsoft/wavlm-base-plus",
}

TARGET_SR = 16_000


def resolve_model_id(name: str) -> str:
    """Map a short alias to a full HuggingFace model id.

    Args:
        name: Alias or full HF repo id.

    Returns:
        str: HuggingFace model id.
    """
    key = name.strip()
    return MODEL_ALIASES.get(key, key)


def pick_device(explicit: str | None = None) -> str:
    """Choose CUDA if available, otherwise CPU.

    Args:
        explicit: Optional device string such as ``cuda:0`` or ``cpu``.

    Returns:
        str: Torch device string.
    """
    if explicit:
        return explicit
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def load_wav_mono_16k(path: Path) -> np.ndarray:
    """Load a WAV as float32 mono at 16 kHz.

    Args:
        path: Path to a WAV file.

    Returns:
        np.ndarray: 1-D waveform at TARGET_SR.
    """
    import soundfile as sf
    from scipy import signal

    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != TARGET_SR:
        n_out = int(round(len(wav) * TARGET_SR / sr))
        wav = signal.resample(wav, n_out).astype(np.float32)
    return wav


def load_encoder(model_id: str, device: str):
    """Load HF Wav2Vec2 or WavLM backbone + processor.

    Args:
        model_id: Full HuggingFace model id.
        device: Torch device string.

    Returns:
        tuple: (processor, model) in eval mode on ``device``.
    """
    from transformers import AutoFeatureExtractor, AutoModel

    print(f"[ssl] loading {model_id} on {device}")
    processor = AutoFeatureExtractor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model.to(device)
    model.eval()
    return processor, model


@torch.inference_mode()
def embed_one(
    processor,
    model,
    wav: np.ndarray,
    device: str,
) -> np.ndarray:
    """Mean-pool the last hidden state into a single utterance vector.

    For single-clip inference there is no pad masking to apply: HF
    ``attention_mask`` is sample-rate length, while ``last_hidden_state`` is
    frame-rate length (~320× shorter for wav2vec2/WavLM). Mean over time is
    therefore the correct and safe pool.

    Args:
        processor: HF feature extractor.
        model: HF SSL encoder.
        wav: Mono float32 waveform at 16 kHz.
        device: Torch device string.

    Returns:
        np.ndarray: 1-D float32 embedding (typically 1024-d for *-large).
    """
    inputs = processor(
        wav,
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding=False,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # Drop sample-rate attention_mask — it does not match hidden T.
    inputs.pop("attention_mask", None)
    out = model(**inputs)
    hidden = out.last_hidden_state  # (1, T_frames, D)
    pooled = hidden.mean(dim=1)
    return pooled.squeeze(0).detach().cpu().numpy().astype(np.float32)


def extract_embeddings(
    model_id: str,
    manifest_path: Path = SPLIT_MANIFEST_CSV,
    output_path: Path | None = None,
    device: str | None = None,
    limit: int | None = None,
    splits: tuple[str, ...] | None = ("train", "test"),
) -> dict:
    """Extract SSL embeddings for selected splits and write a compressed NPZ.

    Args:
        model_id: HF model id or alias (wav2vec2-large-robust / wavlm-large).
        manifest_path: Split manifest with clip_id and audio_path.
        output_path: Destination NPZ. Defaults beside the manifest.
        device: Optional torch device override.
        limit: Optional max rows (smoke tests).
        splits: Split values to include. None keeps all rows.

    Returns:
        dict: Extraction summary (counts, dim, model_id, output path).

    Raises:
        RuntimeError: If no embeddings are successfully extracted.
    """
    model_id = resolve_model_id(model_id)
    if output_path is None:
        safe = model_id.replace("/", "_")
        output_path = manifest_path.parent / f"our_speech_corpus_{safe}.npz"

    df = pd.read_csv(manifest_path)
    if splits:
        df = df[df[SPLIT_COLUMN].isin(splits)].copy()
    if limit is not None:
        df = df.head(limit)

    device_str = pick_device(device)
    processor, model = load_encoder(model_id, device_str)

    clip_ids: list[str] = []
    embeddings: list[np.ndarray] = []
    missing_audio: list[str] = []
    failed: list[str] = []

    for _, row in df.iterrows():
        clip_id = str(row["clip_id"])
        wav_path = Path(str(row["audio_path"]))
        if not wav_path.is_file():
            missing_audio.append(clip_id)
            continue
        try:
            wav = load_wav_mono_16k(wav_path)
            emb = embed_one(processor, model, wav, device_str)
        except Exception as exc:
            print(f"[ssl] fail {clip_id}: {exc}")
            failed.append(clip_id)
            continue
        if emb.size == 0 or not np.isfinite(emb).all():
            failed.append(clip_id)
            continue
        clip_ids.append(clip_id)
        embeddings.append(emb)
        if (len(clip_ids) % 50) == 0:
            print(f"[ssl] processed {len(clip_ids)}/{len(df)}")

    if not embeddings:
        raise RuntimeError("No embeddings extracted")

    emb_matrix = np.vstack(embeddings)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        clip_id=np.array(clip_ids, dtype=object),
        embeddings=emb_matrix,
        embedding_dim=np.array([emb_matrix.shape[1]], dtype=np.int32),
        model_id=np.array([model_id], dtype=object),
    )

    summary = {
        "embeddings_file": str(output_path),
        "model_id": model_id,
        "requested_rows": int(len(df)),
        "extracted": int(len(clip_ids)),
        "missing_audio": int(len(missing_audio)),
        "failed": int(len(failed)),
        "embedding_dim": int(emb_matrix.shape[1]),
        "missing_audio_clip_ids_sample": missing_audio[:20],
        "failed_clip_ids_sample": failed[:20],
    }
    print(
        f"[ssl] wrote {output_path} "
        f"({summary['extracted']} vectors, dim={summary['embedding_dim']})"
    )
    return summary


def main() -> None:
    """CLI entrypoint for HF SSL embedding extraction.

    Args:
        None. Command-line flags: ``--model``, ``--manifest``, ``--output``,
        ``--device``, ``--limit``, and ``--splits``.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Extract wav2vec2 / WavLM embeddings")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HF id or alias: wav2vec2-large-robust | wavlm-large",
    )
    parser.add_argument("--manifest", type=Path, default=SPLIT_MANIFEST_CSV)
    parser.add_argument("--output", type=Path, default=None)
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
        model_id=args.model,
        manifest_path=args.manifest,
        output_path=args.output,
        device=args.device,
        limit=args.limit,
        splits=tuple(args.splits) if args.splits else None,
    )


if __name__ == "__main__":
    main()
