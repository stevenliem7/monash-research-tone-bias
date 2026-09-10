"""Paths and constants for the fine_tuning package."""

from __future__ import annotations

import sys
from pathlib import Path

FINE_TUNING_ROOT = Path(__file__).resolve().parent
REPO_ROOT = FINE_TUNING_ROOT.parent
SRC_ROOT = REPO_ROOT / "src"

if SRC_ROOT.is_dir() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    from config import (  # noqa: E402
        CORPUS_FINAL_AUDIO_DIR,
        SPLIT_MANIFEST_CSV,
        SPLIT_SEED,
        VALENCE_CLASSES,
        WAVLM_EMBEDDINGS_NPZ,
        WAVLM_FINETUNED_CACHE_DIR,
        WAVLM_FINETUNED_CHECKPOINT_DIR,
        WAVLM_FINETUNED_FINAL_PT,
        WAVLM_FINETUNED_LEAKAGE_JSON,
        WAVLM_FINETUNED_NESTED_CV_JSON,
        WAVLM_FINETUNED_OOF_CSV,
        WAVLM_FINETUNED_PAIRED_JSON,
        WAVLM_FINETUNED_REPORTS_DIR,
        WAVLM_FINETUNED_ROOT,
        WAVLM_FINETUNED_RUN_STATE_JSON,
        WAVLM_MODEL_ID,
    )
except ImportError:
    DATA_ROOT = REPO_ROOT / "data"
    DATA_WAVLM = DATA_ROOT / "our_speech_corpus_wavlm"
    DATA_WAVLM_FINETUNED = DATA_ROOT / "our_speech_corpus_wavlm_finetuned"
    VALENCE_CLASSES = ("positive", "neutral", "negative")
    SPLIT_SEED = 42
    SPLIT_MANIFEST_CSV = REPO_ROOT / "our_speech_corpus_manifest_split_80_20.csv"
    CORPUS_FINAL_AUDIO_DIR = (
        REPO_ROOT / "our_speech_corpus_final" / "our_speech_corpus_final"
    )
    WAVLM_EMBEDDINGS_NPZ = DATA_WAVLM / "our_speech_corpus_microsoft_wavlm-large.npz"
    WAVLM_MODEL_ID = "microsoft/wavlm-large"
    WAVLM_FINETUNED_ROOT = DATA_WAVLM_FINETUNED
    WAVLM_FINETUNED_CACHE_DIR = DATA_WAVLM_FINETUNED / "cache"
    WAVLM_FINETUNED_CHECKPOINT_DIR = DATA_WAVLM_FINETUNED / "checkpoints"
    WAVLM_FINETUNED_REPORTS_DIR = DATA_WAVLM_FINETUNED / "reports"
    WAVLM_FINETUNED_NESTED_CV_JSON = WAVLM_FINETUNED_REPORTS_DIR / "nested_cv.json"
    WAVLM_FINETUNED_PAIRED_JSON = WAVLM_FINETUNED_REPORTS_DIR / "paired_inference.json"
    WAVLM_FINETUNED_LEAKAGE_JSON = WAVLM_FINETUNED_REPORTS_DIR / "leakage_check.json"
    WAVLM_FINETUNED_OOF_CSV = WAVLM_FINETUNED_REPORTS_DIR / "oof_predictions.csv"
    WAVLM_FINETUNED_FINAL_PT = WAVLM_FINETUNED_CHECKPOINT_DIR / "final.pt"
    WAVLM_FINETUNED_RUN_STATE_JSON = DATA_WAVLM_FINETUNED / "run_state.json"

LABEL_TO_ID = {label: i for i, label in enumerate(VALENCE_CLASSES)}
ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}
NEG_ID = LABEL_TO_ID["negative"]

CACHE_FORMAT_VERSION = 1
PROTOCOL_VERSION = "honest_v1"
PROBE_LAYERS = (4, 8, 12, 16, 20, 24)
SCALAR_LRS = (1e-4, 1e-3, 3e-3, 1e-2)
ENCODER_LRS = (1e-5, 3e-5)
ACTIVE_DEPTHS = (2, 4)
OUTER_SEEDS = (42, 43, 44)
PRIMARY_SEED = 42
LAST_LAYER_IDX = 24  # post final encoder.layer_norm in WavLM-large stack

__all__ = [
    "ACTIVE_DEPTHS",
    "CACHE_FORMAT_VERSION",
    "CORPUS_FINAL_AUDIO_DIR",
    "ENCODER_LRS",
    "FINE_TUNING_ROOT",
    "ID_TO_LABEL",
    "LABEL_TO_ID",
    "LAST_LAYER_IDX",
    "NEG_ID",
    "OUTER_SEEDS",
    "PRIMARY_SEED",
    "PROBE_LAYERS",
    "PROTOCOL_VERSION",
    "REPO_ROOT",
    "SCALAR_LRS",
    "SPLIT_MANIFEST_CSV",
    "SPLIT_SEED",
    "SRC_ROOT",
    "VALENCE_CLASSES",
    "WAVLM_EMBEDDINGS_NPZ",
    "WAVLM_FINETUNED_CACHE_DIR",
    "WAVLM_FINETUNED_CHECKPOINT_DIR",
    "WAVLM_FINETUNED_FINAL_PT",
    "WAVLM_FINETUNED_LEAKAGE_JSON",
    "WAVLM_FINETUNED_NESTED_CV_JSON",
    "WAVLM_FINETUNED_OOF_CSV",
    "WAVLM_FINETUNED_PAIRED_JSON",
    "WAVLM_FINETUNED_REPORTS_DIR",
    "WAVLM_FINETUNED_ROOT",
    "WAVLM_FINETUNED_RUN_STATE_JSON",
    "WAVLM_MODEL_ID",
]
