"""Manifest loading, grouping, and stratified microbatch construction."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from fine_tuning.paths import LABEL_TO_ID, VALENCE_CLASSES

TARGET_SR = 16_000


def manifest_fingerprint(manifest_path: Path) -> str:
    """SHA256 of the manifest bytes for cache invalidation."""
    return hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def load_labelled_frame(manifest_path: Path) -> pd.DataFrame:
    """Load labelled rows with canonical valence labels and question groups."""
    frame = pd.read_csv(manifest_path)
    labelled = frame[frame["is_labelled"]].copy().reset_index(drop=True)
    labelled["ground_truth_label"] = (
        labelled["ground_truth_label"].astype(str).str.strip().str.lower()
    )
    unknown = sorted(set(labelled["ground_truth_label"]) - set(VALENCE_CLASSES))
    if unknown:
        raise ValueError(f"Unexpected ground_truth_label values: {unknown}")
    labelled["y"] = labelled["ground_truth_label"].map(LABEL_TO_ID).astype(np.int64)
    labelled["question"] = labelled["question"].astype(str)
    labelled["clip_id"] = labelled["clip_id"].astype(str)
    return labelled


def class_weights_from_y(y: np.ndarray, n_classes: int = 3) -> np.ndarray:
    """Inverse-frequency weights matching sklearn balanced mode."""
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    return (counts.sum() / (n_classes * counts)).astype(np.float32)


def make_grouped_folds(
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Deterministic StratifiedGroupKFold splits."""
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(cv.split(np.zeros(len(y)), y, groups))


def make_random_folds(
    y: np.ndarray,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Deterministic StratifiedKFold splits (leakage diagnostic only)."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(cv.split(np.zeros(len(y)), y))


def remap_audio_paths(
    df: pd.DataFrame,
    audio_dir: Path,
    *,
    strict: bool = True,
) -> pd.DataFrame:
    """Set ``audio_path`` to ``{audio_dir}/{clip_id}.wav``."""
    audio_dir = Path(audio_dir)
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    out = df.copy()
    missing_ids: list[str] = []
    remapped: list[str] = []
    for _, row in out.iterrows():
        clip_id = str(row["clip_id"]).strip()
        wav_path = audio_dir / f"{clip_id}.wav"
        remapped.append(str(wav_path))
        if not wav_path.is_file():
            missing_ids.append(clip_id)

    out["audio_path"] = remapped
    if missing_ids and strict:
        raise FileNotFoundError(
            f"{len(missing_ids)}/{len(out)} WAV files missing under {audio_dir}; "
            f"first missing clip_id: {missing_ids[0]}"
        )
    return out


def load_wav_mono_16k(path: Path) -> np.ndarray:
    """Load a WAV as float32 mono at 16 kHz."""
    import soundfile as sf
    from scipy import signal

    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != TARGET_SR:
        n_out = int(round(len(wav) * TARGET_SR / sr))
        wav = signal.resample(wav, n_out).astype(np.float32)
    return wav
