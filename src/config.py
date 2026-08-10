"""Universal paths and settings for monash-research-tone-bias.

Import from any script under ``src/``:

    from config import REPO_ROOT, SPLIT_MANIFEST_CSV, bootstrap_imports

When running scripts with ``uv run``, ``PYTHONPATH=src`` in ``.env`` makes
``config`` importable from any depth. Without that, use :mod:`load_config` or
call :func:`bootstrap_imports` after inserting ``src/`` on ``sys.path`` (see
:mod:`load_config` for a copy-paste snippet).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Repo layout
SRC_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SRC_ROOT.parent
WORKSPACE_ROOT = REPO_ROOT.parent

# Aliases used across older scripts
PROJECT_ROOT = REPO_ROOT
WORKSPACE = WORKSPACE_ROOT

#  HEET source CSVs 
HEET_RAW_CSV = REPO_ROOT / "heet_dataset.csv"
HEET_CLEAN_CSV = REPO_ROOT / "heet_dataset_clean.csv"
HEET_NEW_NEGATIVE_CSV = REPO_ROOT / "heet_dataset_new_negative_clean.csv"

#  Shared data (backbone-agnostic) 
DATA_ROOT = REPO_ROOT / "data"
UNIFIED_MANIFEST_CSV = DATA_ROOT / "our_speech_corpus_unified.csv"
SPLIT_MANIFEST_CSV = DATA_ROOT / "our_speech_corpus_manifest_split_80_20.csv"
LABEL_PASS_DIFF_CSV = DATA_ROOT / "label_pass_diff.csv"
LABEL_PASS_AGREEMENT_JSON = DATA_ROOT / "label_pass_agreement.json"

# Per-backbone artifact directories
DATA_EMOTION2VEC = DATA_ROOT / "our_speech_corpus_emotion2vec"
DATA_WAVLM = DATA_ROOT / "our_speech_corpus_wavlm"
DATA_WAV2VEC2 = DATA_ROOT / "our_speech_corpus_wav2vec2"
DATA_TEXT = DATA_ROOT / "our_speech_corpus_text"

# Legacy alias (was data/our_speech_corpus/)
DATA_DIR = DATA_ROOT

# emotion2vec artifacts
EMOTION2VEC_EMBEDDINGS_NPZ = DATA_EMOTION2VEC / "our_speech_corpus_embeddings_emotion2vec.npz"
EMOTION2VEC_QC_REPORT_JSON = DATA_EMOTION2VEC / "reports" / "our_speech_corpus_qc_report.json"
EMOTION2VEC_HEAD_RESULTS_JSON = (
    DATA_EMOTION2VEC / "reports" / "our_speech_corpus_valence_head_results_new_labels.json"
)
EMOTION2VEC_CV_RESULTS_JSON = (
    DATA_EMOTION2VEC / "reports" / "our_speech_corpus_valence_head_cv_new_labels.json"
)

# WavLM / wav2vec2 artifacts
WAVLM_EMBEDDINGS_NPZ = DATA_WAVLM / "our_speech_corpus_microsoft_wavlm-large.npz"
WAVLM_POOL_EMBEDDINGS_NPZ = DATA_WAVLM / "our_speech_corpus_microsoft_wavlm-large_unlabelled_pool.npz"
WAVLM_CV_RESULTS_JSON = (
    DATA_WAVLM / "reports" / "our_speech_corpus_valence_head_cv_wavlm_new_label.json"
)
WAVLM_CV_GROUP_QUESTION_JSON = (
    DATA_WAVLM / "reports" / "valence_head_cv_wavlm_group_question.json"
)
WAV2VEC2_EMBEDDINGS_NPZ = DATA_WAV2VEC2 / "our_speech_corpus_facebook_wav2vec2-large-robust.npz"
WAV2VEC2_CV_RESULTS_JSON = (
    DATA_WAV2VEC2 / "reports" / "our_speech_corpus_valence_head_cv_new_labels.json"
)

# Text-only (transcript) valence baselines
TEXT_CV_RESULTS_JSON = (
    DATA_TEXT / "reports" / "valence_head_cv_text_group_question.json"
)
TEXT_REPORTS_DIR = DATA_TEXT / "reports"
MINILM_EMBEDDINGS_NPZ = DATA_TEXT / "minilm_embeddings.npz"
MINILM_CV_GROUP_QUESTION_JSON = (
    TEXT_REPORTS_DIR / "valence_head_cv_minilm_group_question.json"
)
DISTILBERT_EMBEDDINGS_NPZ = DATA_TEXT / "distilbert_embeddings.npz"
DISTILBERT_CV_GROUP_QUESTION_JSON = (
    TEXT_REPORTS_DIR / "valence_head_cv_distilbert_group_question.json"
)
ROBERTA_EMBEDDINGS_NPZ = DATA_TEXT / "roberta_embeddings.npz"
ROBERTA_CV_GROUP_QUESTION_JSON = (
    TEXT_REPORTS_DIR / "valence_head_cv_roberta_group_question.json"
)
DISTILBERT_WEAK_LABEL_AGREEMENT_JSON = (
    TEXT_REPORTS_DIR / "distilbert_weak_label_agreement.json"
)
ROBERTA_WEAK_LABEL_AGREEMENT_JSON = (
    TEXT_REPORTS_DIR / "roberta_weak_label_agreement.json"
)
DISTILBERT_SST2_MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"
# Native 3-class (neg/neu/pos) sentiment classifier for weak-label agreement
TWITTER_ROBERTA_SENTIMENT_MODEL_ID = (
    "cardiffnlp/twitter-roberta-base-sentiment-latest"
)
MINILM_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

# Backward-compatible names from pipeline_config
EMBEDDINGS_NPZ = EMOTION2VEC_EMBEDDINGS_NPZ
QC_REPORT_JSON = EMOTION2VEC_QC_REPORT_JSON
HEAD_RESULTS_JSON = EMOTION2VEC_HEAD_RESULTS_JSON

#  Audio corpora 
CORPORA_CLEANED_ROOT = WORKSPACE_ROOT / "corpora_cleaned"
CORPUS_FINAL_AUDIO_DIR = CORPORA_CLEANED_ROOT / "our_speech_corpus_final"
CORPUS_FINAL_SUBDIR = "our_speech_corpus_final"
CORPUS_A_AUDIO_DIR = CORPORA_CLEANED_ROOT / "our_speech_corpus_cleaned"
CORPUS_B_AUDIO_DIR = CORPORA_CLEANED_ROOT / "our_speech_corpus_new_negative_cleaned"

# Raw corpora (pre-cleaning) and manifests
CORPORA_RAW_ROOT = WORKSPACE_ROOT / "corpora"
MANIFESTS_ROOT = REPO_ROOT / "manifests"

# RQ1 transfer corpora live under corpora_cleaned/
RQ1_CORPORA_ROOT = CORPORA_CLEANED_ROOT

#  Results 
RESULTS_ROOT = WORKSPACE_ROOT / "results"
RESULTS_EMOTION2VEC = RESULTS_ROOT / "emotion2vec"
RESULTS_EIV = RESULTS_ROOT / "empathic_insight_voice_all36"

#  our_speech_corpus_final package (scripts) 
OUR_SPEECH_CORPUS_PKG = SRC_ROOT / "our_speech_corpus_final"

#  Manifest schema 
MANIFEST_COLUMNS = [
    "clip_id",
    "source_batch",
    "question",
    "emotion_label",
    "valence_label",
    "audio_path",
    "response_text",
    "ground_truth_label",
    "is_labelled",
    "duration_s",
]

SPLIT_COLUMN = "split"
SPLIT_VALUES = ("train", "test", "unlabeled_pool")

SOURCE_BATCH_A = "corpusA_1200"
SOURCE_BATCH_B = "corpusB_2000"

VALENCE_CLASSES = ("positive", "neutral", "negative")

# Split policy: 80/20 on labelled rows, stratified by class + batch
TRAIN_FRACTION = 0.8
SPLIT_SEED = 42

#  Model IDs 
EMOTION2VEC_MODEL_ID = "iic/emotion2vec_plus_large"
MODEL_ID = EMOTION2VEC_MODEL_ID
WAVLM_MODEL_ID = "microsoft/wavlm-large"
WAV2VEC2_MODEL_ID = "facebook/wav2vec2-large-robust"

#  Misc 
DOTENV_PATH = REPO_ROOT / ".env"
GEMINI_DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "our_speech_corpus_test"
HYBRID_NEGATIVE_JSONL = WORKSPACE_ROOT / "hybrid_negative_voice_assistant_2000.jsonl"
EMPATHIC_INSIGHT_VOICE_MLP_DIR = WORKSPACE_ROOT / "empathic_insight_voice_small_models"

#  Corpus extractors (raw -> cleaned) 
EMOVOICE_RAW_DIR = CORPORA_RAW_ROOT / "EmoVoice-DB"
EMOVOICE_CLEANED_DIR = CORPORA_CLEANED_ROOT / "emovoice_cleaned"
STYLETALK_RAW_DIR = CORPORA_RAW_ROOT / "StyleTalk"
STYLETALK_CLEANED_DIR = CORPORA_CLEANED_ROOT / "styletalk_cleaned"
IEMOCAP_SYN_RAW_DIR = CORPORA_RAW_ROOT / "IEMOCAP_SYN" / "cosyvoice2"
IEMOCAP_SYN_CLEANED_DIR = CORPORA_CLEANED_ROOT / "iemocap_synth_cleaned"
TESS_HUMAN_RAW_DIR = CORPORA_RAW_ROOT / "TESS" / "data" / "tess" / "tess"
TESS_HUMAN_CLEANED_DIR = CORPORA_CLEANED_ROOT / "tess_human_cleaned"
TESS_INDEXTTS_RAW_DIR = CORPORA_RAW_ROOT / "TESS_SYN" / "indextts2"
TESS_INDEXTTS_CLEANED_DIR = CORPORA_CLEANED_ROOT / "tess_indextts_cleaned"
DEEPDIALOGUE_XTTS_CLEANED_DIR = CORPORA_CLEANED_ROOT / "deepdialogue_xtts_cleaned"
IEMOCAP_HUMAN_WAVS_DIR = CORPORA_RAW_ROOT / "IEMOCAP" / "wavs"
IEMOCAP_HUMAN_CLEANED_DIR = CORPORA_CLEANED_ROOT / "iemocap_human_cleaned"
MANIFEST_DEEPDIALOGUE_HF_CSV = MANIFESTS_ROOT / "manifest_deepdialogue_hf.csv"
MANIFEST_IEMOCAP_HF_CSV = MANIFESTS_ROOT / "manifest_iemocap_hf.csv"

#  Inference entrypoints 
EIV_ALL36_SCRIPT = SRC_ROOT / "inference" / "empathic_insight_voice_all36_kaggle.py"
EMOTION2VEC_SCRIPT = SRC_ROOT / "inference" / "emotion2vec.py"


def bootstrap_imports(caller_file: str | Path) -> Path:
    """Insert ``src/`` on ``sys.path`` so ``import config`` works from any depth.

    Args:
        caller_file: Typically ``__file__`` of the calling script.

    Returns:
        Path: Absolute path to ``src/`` (directory containing this module).

    Raises:
        RuntimeError: If ``src/config.py`` cannot be located above the caller.
    """
    for parent in Path(caller_file).resolve().parents:
        if (parent / "config.py").is_file():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return parent
    raise RuntimeError(f"Could not locate src/config.py above {caller_file}")


def extend_our_speech_corpus_paths() -> None:
    """Add ``our_speech_corpus_final`` subdirs for legacy flat imports.

    Inserts ``preprocessing/``, ``embeddings/``, and ``heads/`` onto ``sys.path``
    when present so scripts can import siblings without package prefixes.

    Returns:
        None
    """
    for sub in ("preprocessing", "embeddings", "heads", "inference", "cross_corpus_evaluators"):
        folder = OUR_SPEECH_CORPUS_PKG / sub
        if folder.is_dir() and str(folder) not in sys.path:
            sys.path.insert(0, str(folder))
