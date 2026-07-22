#!/usr/bin/env python3
"""
Authors:
    Steven Liem (steven.liem@sydney.edu.au)

Empathic-Insight-Voice inference on EMONET-VOICE emotion heads (+ Valence).

Uses 36 heads for valence aggregation after dropping 4 unmapped/ambiguous heads (Astonishment/Surprise, Awe, Sexual Lust, Intoxication). Interest,
Doubt, Confusion, Concentration, Contemplation are polarity=neutral.

Mass method: softmax → count-normalized polarity mass (sum/n_heads per
pos/neu/neg) → argmax polarity (no mass_margin).

Designed for Kaggle (GPU). Defaults detect /kaggle/input and /kaggle/working.

Kaggle:
  1. Upload corpora_cleaned as a Dataset (or zip under /kaggle/input/...)
  2. Run:

     !pip -q install transformers huggingface_hub scikit-learn librosa soundfile
     !python empathic_insight_voice_all36_kaggle.py \\
         --corpora-root /kaggle/input/corpora-cleaned \\
         --batch-size 16 

* You can configure the batch size depending on the number of GPUs you have available. 

This script only runs inference and writes per-corpus predictions.csv + metrics.json. Summary tables / diagrams:
      src/evaluators/eiv_summary_table.py
      src/evaluators/eiv_all_emotions.py

Local / Colab:
     uv run python empathic_insight_voice_all36_kaggle.py --dry-run
     uv run python empathic_insight_voice_all36_kaggle.py --corpus tess_human_cleaned --limit 50

Refs:
    https://huggingface.co/laion/Empathic-Insight-Voice-Small
    https://huggingface.co/mkrausio/EmoWhisper-AnS-Small-v0.1
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# ---------------------------------------------------------------------------
# Paths — Kaggle / Colab / local
# ---------------------------------------------------------------------------
_ON_KAGGLE = Path("/kaggle/working").is_dir()
_ON_COLAB = (not _ON_KAGGLE) and Path("/content").is_dir()
_LOCAL_ROOT = Path(__file__).resolve().parents[1]

if _ON_KAGGLE:
    CORPORA_ROOT = Path("/kaggle/input/corpora-cleaned")
    RESULTS_ROOT = Path("/kaggle/working/eiv_all36_results")
    MLP_DIR = Path("/kaggle/working/empathic_insight_voice_small_models")
elif _ON_COLAB:
    CORPORA_ROOT = Path("/content/corpora_cleaned")
    RESULTS_ROOT = Path("/content/eiv_all36_results")
    MLP_DIR = Path("/content/empathic_insight_voice_small_models")
else:
    CORPORA_ROOT = _LOCAL_ROOT / "corpora_cleaned"
    RESULTS_ROOT = _LOCAL_ROOT / "results" / "empathic_insight_voice_all36"
    MLP_DIR = _LOCAL_ROOT / "empathic_insight_voice_small_models"

# Human valence labels (HEET); only our_speech_corpus_cleaned uses this.
DEFAULT_HEET = Path(__file__).resolve().parent / "heet_dataset_clean.csv"

SAMPLING_RATE = 16000
MAX_AUDIO_SECONDS = 30.0
WHISPER_MODEL_ID = "mkrausio/EmoWhisper-AnS-Small-v0.1"
HF_MLP_REPO_ID = "laion/Empathic-Insight-Voice-Small"
WHISPER_SEQ_LEN = 1500
WHISPER_EMBED_DIM = 768
PROJECTION_DIM = 64
MLP_HIDDEN = [64, 32, 16]
MLP_DROPOUTS = [0.0, 0.1, 0.1, 0.1]

VALENCE_CLASSES = ("positive", "neutral", "negative")
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

# HF filename stem (model_<STEM>_best.pth) → (display name, polarity)
# polarity drives top1 + mass. Mass uses count-normalized sums over each polarity.
# Dropped (unmapped / ambiguous on EMONET→EmoDB): Astonishment/Surprise, Awe,
# Sexual Lust, Intoxication/Altered States.
EMOTION_HEADS: dict[str, tuple[str, str]] = {
    # positive
    "Affection": ("Affection", "positive"),
    "Amusement": ("Amusement", "positive"),
    "Elation": ("Elation", "positive"),
    "Pleasure_Ecstasy": ("Pleasure/Ecstasy", "positive"),
    "Contentment": ("Contentment", "positive"),
    "Thankfulness_Gratitude": ("Thankfulness/Gratitude", "positive"),
    "Infatuation": ("Infatuation", "positive"),
    "Hope_Enthusiasm_Optimism": ("Hope/Enthusiasm/Optimism", "positive"),
    "Triumph": ("Triumph", "positive"),
    "Pride": ("Pride", "positive"),
    "Relief": ("Relief", "positive"),
    "Teasing": ("Teasing", "positive"),
    # negative
    "Impatience_and_Irritability": ("Impatience and Irritability", "negative"),
    "Fear": ("Fear", "negative"),
    "Distress": ("Distress", "negative"),
    "Embarrassment": ("Embarrassment", "negative"),
    "Shame": ("Shame", "negative"),
    "Disappointment": ("Disappointment", "negative"),
    "Sadness": ("Sadness", "negative"),
    "Bitterness": ("Bitterness", "negative"),
    "Contempt": ("Contempt", "negative"),
    "Disgust": ("Disgust", "negative"),
    "Anger": ("Anger", "negative"),
    "Malevolence_Malice": ("Malevolence/Malice", "negative"),
    "Sourness": ("Sourness", "negative"),
    "Pain": ("Pain", "negative"),
    "Helplessness": ("Helplessness", "negative"),
    "Fatigue_Exhaustion": ("Fatigue/Exhaustion", "negative"),
    "Emotional_Numbness": ("Emotional Numbness", "negative"),
    "Jealousy_&_Envy": ("Jealousy / Envy", "negative"),
    "Longing": ("Longing", "negative"),
    # neutral (EmoDB-neutral; excluded from pos/neg mass)
    "Interest": ("Interest", "neutral"),
    "Doubt": ("Doubt", "neutral"),
    "Confusion": ("Confusion", "neutral"),
    "Concentration": ("Concentration", "neutral"),
    "Contemplation": ("Contemplation", "neutral"),
}
assert len(EMOTION_HEADS) == 36, len(EMOTION_HEADS)

NEG_HEADS = [k for k, (_, v) in EMOTION_HEADS.items() if v == "negative"]
POS_HEADS = [k for k, (_, v) in EMOTION_HEADS.items() if v == "positive"]
NEU_HEADS = [k for k, (_, v) in EMOTION_HEADS.items() if v == "neutral"]
NONPOLAR_LABELS = frozenset({"neutral", "ambiguous"})


class FullEmbeddingMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.proj = nn.Linear(WHISPER_SEQ_LEN * WHISPER_EMBED_DIM, PROJECTION_DIM)
        layers: list[nn.Module] = [nn.ReLU(), nn.Dropout(MLP_DROPOUTS[0])]
        dim = PROJECTION_DIM
        for i, h in enumerate(MLP_HIDDEN):
            layers += [nn.Linear(dim, h), nn.ReLU(), nn.Dropout(MLP_DROPOUTS[i + 1])]
            dim = h
        layers.append(nn.Linear(dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        if x.ndim == 4 and x.shape[1] == 1:
            x = x.squeeze(1)
        return self.mlp(self.proj(self.flatten(x)))


def pick_device(explicit: str | None = None) -> torch.device:
    if explicit:
        return torch.device(explicit)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_gt_valence(path: Path) -> str | None:
    valence = path.stem.split("_")[-1].strip().lower()
    return valence if valence in VALENCE_CLASSES else None


def parse_gt_emotion(path: Path) -> str | None:
    parts = path.stem.split("_")
    return parts[-2].strip().lower() if len(parts) >= 2 else None


def load_human_gt_labels(heet_path: Path | None) -> dict[str, str]:
    """Map wav filename → human ground_truth_label from HEET CSV.

    Only rows with positive/neutral/negative labels are kept. Missing path
    or empty labels yield an empty mapping (filename GT is unchanged).
    """
    if heet_path is None or not heet_path.is_file():
        return {}
    heet = pd.read_csv(heet_path)
    if "ground_truth_label" not in heet.columns or "audio_path" not in heet.columns:
        return {}
    labels = heet["ground_truth_label"].fillna("").astype(str).str.strip().str.lower()
    names = heet["audio_path"].fillna("").map(lambda p: Path(str(p)).name)
    keep = labels.isin(VALENCE_CLASSES)
    return dict(zip(names[keep], labels[keep]))


def bin_valence(score: float, band: float) -> str:
    if score < -band:
        return "negative"
    if score > band:
        return "positive"
    return "neutral"


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if n <= 0:
        return None, None
    p = successes / n
    z2 = z * z
    den = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / den
    half = z * math.sqrt((p * (1 - p) + z2 / (4 * n)) / n) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def load_wav_mono_16k(path: Path) -> np.ndarray:
    """Load mono audio at 16 kHz via librosa (EIV HF README pipeline).

    Args:
        path: Path to an audio file.

    Returns:
        np.ndarray: Float32 mono waveform at SAMPLING_RATE.
    """
    waveform, _sr = librosa.load(str(path), sr=SAMPLING_RATE, mono=True)
    return waveform.astype(np.float32, copy=False)


def load_mlp(pth: Path, device: torch.device) -> FullEmbeddingMLP:
    mlp = FullEmbeddingMLP()
    try:
        state = torch.load(pth, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(pth, map_location="cpu")
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    mlp.load_state_dict(state)
    return mlp.to(device).eval()


def download_heads(mlp_dir: Path) -> dict[str, Path]:
    """Download Valence + all 36 emotion heads; return stem → .pth path."""
    stems = ["Valence", *EMOTION_HEADS.keys()]
    patterns = [f"model_{s}_best.pth" for s in stems]
    mlp_dir.mkdir(parents=True, exist_ok=True)
    print(f"[eiv36] downloading {len(patterns)} MLP heads → {mlp_dir}")
    snapshot_download(
        repo_id=HF_MLP_REPO_ID,
        local_dir=mlp_dir,
        allow_patterns=patterns,
        repo_type="model",
    )
    found: dict[str, Path] = {}
    for stem in stems:
        matches = list(mlp_dir.rglob(f"model_{stem}_best.pth"))
        if not matches:
            raise FileNotFoundError(f"missing model_{stem}_best.pth under {mlp_dir}")
        found[stem] = matches[0]
    print(f"[eiv36] ready: {len(found)} heads (Valence + {len(EMOTION_HEADS)} emotions)")
    return found


@torch.no_grad()
def embed_batch(processor, whisper, wavs: list[Path], device: torch.device) -> torch.Tensor:
    max_samples = int(MAX_AUDIO_SECONDS * SAMPLING_RATE)
    waveforms = []
    for wav in wavs:
        w = load_wav_mono_16k(wav)
        if len(w) > max_samples:
            w = w[:max_samples]
        waveforms.append(w)
    feats = processor(waveforms, sampling_rate=SAMPLING_RATE, return_tensors="pt").input_features
    feats = feats.to(device).to(whisper.dtype)
    emb = whisper.get_encoder()(input_features=feats).last_hidden_state
    seq = emb.shape[1]
    if seq < WHISPER_SEQ_LEN:
        pad = torch.zeros(
            (emb.shape[0], WHISPER_SEQ_LEN - seq, WHISPER_EMBED_DIM),
            device=device,
            dtype=emb.dtype,
        )
        emb = torch.cat((emb, pad), dim=1)
    elif seq > WHISPER_SEQ_LEN:
        emb = emb[:, :WHISPER_SEQ_LEN, :]
    out = emb.detach().cpu().float()
    del emb, feats
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


@torch.no_grad()
def score_embs(mlp: FullEmbeddingMLP, embs: torch.Tensor, device: torch.device) -> list[float]:
    x = embs.to(device).to(next(mlp.parameters()).dtype)
    return [float(v) for v in mlp(x).reshape(-1).detach().cpu()]


def mass_pred_from_norm_masses(pos_mass: float, neu_mass: float, neg_mass: float) -> str:
    """Pick valence by argmax of count-normalized polarity masses.

    Args:
        pos_mass: Mean softmax mass over positive heads.
        neu_mass: Mean softmax mass over neutral heads.
        neg_mass: Mean softmax mass over negative heads.

    Returns:
        str: positive, neutral, or negative. Ties prefer positive > neutral > negative.
    """
    scores = {"positive": pos_mass, "neutral": neu_mass, "negative": neg_mass}
    return max(scores, key=scores.get)


def aggregate_emotions(score_row: dict[str, float]) -> dict:
    """Softmax over emotion heads → top-1 valence + count-normalized mass argmax."""
    stems = list(EMOTION_HEADS.keys())
    raw = torch.tensor([score_row[s] for s in stems], dtype=torch.float32)
    probs = torch.softmax(raw, dim=0)
    top_i = int(torch.argmax(probs).item())
    top_stem = stems[top_i]
    top_name, top_polarity = EMOTION_HEADS[top_stem]
    # Neutral/ambiguous top-1 → neutral valence.
    if top_polarity in NONPOLAR_LABELS:
        top_valence = "neutral"
    else:
        top_valence = top_polarity

    # Count-normalized mass: sum(p) / n_heads so 12 pos ≈ 19 neg ≈ 5 neu.
    pos_mass = float(
        sum(probs[i].item() for i, s in enumerate(stems) if s in POS_HEADS) / len(POS_HEADS)
    )
    neu_mass = float(
        sum(probs[i].item() for i, s in enumerate(stems) if s in NEU_HEADS) / len(NEU_HEADS)
    )
    neg_mass = float(
        sum(probs[i].item() for i, s in enumerate(stems) if s in NEG_HEADS) / len(NEG_HEADS)
    )

    # Top-3 for inspection
    top3_idx = torch.topk(probs, k=min(3, len(stems))).indices.tolist()
    top3 = [
        {
            "stem": stems[j],
            "name": EMOTION_HEADS[stems[j]][0],
            "prob": float(probs[j].item()),
            "raw": float(score_row[stems[j]]),
        }
        for j in top3_idx
    ]

    return {
        "top_emotion": top_name,
        "top_emotion_stem": top_stem,
        "top_emotion_polarity": top_polarity,
        "top_emotion_prob": float(probs[top_i].item()),
        "pred_top1": top_valence,
        "pos_mass": pos_mass,
        "neu_mass": neu_mass,
        "neg_mass": neg_mass,
        "pred_mass": mass_pred_from_norm_masses(pos_mass, neu_mass, neg_mass),
        "top3_json": json.dumps(top3),
        **{f"raw_{s}": float(score_row[s]) for s in stems},
        **{f"prob_{s}": float(probs[i].item()) for i, s in enumerate(stems)},
    }


def metrics_for_method(y_true: list[str], y_pred: list[str]) -> dict:
    labels = list(VALENCE_CLASSES)
    if not y_true:
        return {"n": 0}
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {
        label: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, label in enumerate(labels)
    }
    neg_idx = labels.index("negative")
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == "negative" and p == "negative")
    pred_neg = sum(1 for p in y_pred if p == "negative")
    true_neg = sum(1 for t in y_true if t == "negative")
    neg_prec = tp / pred_neg if pred_neg else 0.0
    neg_rec = tp / true_neg if true_neg else 0.0
    prec_lo, prec_hi = wilson_interval(tp, pred_neg)
    rec_lo, rec_hi = wilson_interval(tp, true_neg)
    counts = pd.Series(y_true).value_counts()
    majority = str(counts.idxmax()) if len(counts) else None
    majority_acc = float((counts.max() / len(y_true)) if len(y_true) else 0.0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        "majority_class": majority,
        "majority_baseline_acc": majority_acc,
        "per_class": per_class,
        "neg_precision": float(neg_prec),
        "neg_recall": float(neg_rec),
        "neg_precision_wilson95": [prec_lo, prec_hi],
        "neg_recall_wilson95": [rec_lo, rec_hi],
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, digits=4, zero_division=0
        ),
    }


def evaluate_dataframe(df: pd.DataFrame, pred_col: str) -> dict:
    truth_col = (
        "prompted_valence"
        if "prompted_valence" in df.columns
        else "ground_truth_valence"
    )
    g = df[df[truth_col].isin(VALENCE_CLASSES) & df[pred_col].isin(VALENCE_CLASSES)]
    return metrics_for_method(g[truth_col].tolist(), g[pred_col].tolist())


def select_wavs(corpus: str, limit: int | None, per_class_limit: int | None) -> list[Path]:
    wavs = sorted((CORPORA_ROOT / corpus).glob("*.wav"))
    if per_class_limit is not None:
        buckets: dict[str, list[Path]] = {v: [] for v in VALENCE_CLASSES}
        for w in wavs:
            v = parse_gt_valence(w)
            if v in buckets and len(buckets[v]) < per_class_limit:
                buckets[v].append(w)
        chosen: list[Path] = []
        for v in VALENCE_CLASSES:
            chosen.extend(buckets[v])
        return chosen
    if limit is not None:
        return wavs[:limit]
    return wavs


def run_corpus(
    corpus: str,
    processor,
    whisper,
    mlps: dict[str, FullEmbeddingMLP],
    whisper_device: torch.device,
    mlp_device: torch.device,
    limit: int | None,
    per_class_limit: int | None,
    batch_size: int,
    neutral_band: float,
    human_gt: dict[str, str] | None = None,
) -> pd.DataFrame:
    corpus_dir = CORPORA_ROOT / corpus
    if not corpus_dir.is_dir():
        print(f"[eiv36] skip missing {corpus_dir}")
        return pd.DataFrame()

    wavs = select_wavs(corpus, limit, per_class_limit)
    print(f"[eiv36] {corpus}: {len(wavs)} wavs × {len(EMOTION_HEADS)} emotion heads")
    human_gt = human_gt or {}

    rows: list[dict] = []
    for start in range(0, len(wavs), batch_size):
        batch = wavs[start : start + batch_size]
        embs = embed_batch(processor, whisper, batch, whisper_device)
        batch_scores = {stem: score_embs(mlp, embs, mlp_device) for stem, mlp in mlps.items()}
        del embs
        for i, wav in enumerate(batch):
            emotion_raw = {s: batch_scores[s][i] for s in EMOTION_HEADS}
            agg = aggregate_emotions(emotion_raw)
            raw_v = batch_scores["Valence"][i]
            rows.append(
                {
                    "corpus": corpus,
                    "filename": wav.name,
                    # Filename / synthesis target valence — NOT human HEET labels.
                    "prompted_valence": parse_gt_valence(wav),
                    "ground_truth_emotion": parse_gt_emotion(wav),
                    "human_ground_truth_label": human_gt.get(wav.name),
                    "raw_valence": raw_v,
                    "pred_valence_head": bin_valence(raw_v, neutral_band),
                    **agg,
                }
            )
        done = min(start + batch_size, len(wavs))
        print(f"[eiv36] {corpus}: {done}/{len(wavs)}")
    return pd.DataFrame(rows)


def summarize_corpus(df: pd.DataFrame, neutral_band: float) -> dict:
    corpus = str(df["corpus"].iloc[0]) if len(df) else "unknown"
    methods = {
        "valence_head": "pred_valence_head",
        "top1": "pred_top1",
        "mass": "pred_mass",
    }
    out: dict = {
        "corpus": corpus,
        "n_total": int(len(df)),
        "n_emotion_heads": len(EMOTION_HEADS),
        "n_pos_heads": len(POS_HEADS),
        "n_neu_heads": len(NEU_HEADS),
        "n_neg_heads": len(NEG_HEADS),
        "neutral_band": neutral_band,
        "mass_rule": "count_normalized_argmax",
        "methods": {},
        "top_emotion_counts": df["top_emotion"].value_counts().to_dict() if len(df) else {},
    }

    print("\n" + "=" * 72)
    print(f"CORPUS {corpus}  n={len(df)}  ({len(EMOTION_HEADS)} emotion heads)")
    print("=" * 72)

    for name, col in methods.items():
        met = evaluate_dataframe(df, col)
        out["methods"][name] = met
        print(f"\n--- {name} ---")
        if met.get("n", 0) == 0:
            print("  (no evaluated rows)")
            continue
        print(
            f"  acc={met['accuracy']:.4f}  bal_acc={met['balanced_accuracy']:.4f}  "
            f"macro_f1={met['macro_f1']:.4f}"
        )
        print(
            f"  neg precision={met['neg_precision']:.4f}  "
            f"neg recall={met['neg_recall']:.4f}"
        )
        print(met["classification_report"])

    print("\nTop predicted emotions (counts):")
    for emo, n in list(out["top_emotion_counts"].items())[:10]:
        print(f"  {emo}: {n}")
    return out


def resolve_corpora_root(explicit: Path) -> Path:
    """On Kaggle, accept either the dataset root or a nested corpora_cleaned/."""
    if explicit.is_dir() and any(explicit.glob("*.wav")):
        return explicit
    if explicit.is_dir() and any((explicit / c).is_dir() for c in CORPUS_DIRS):
        return explicit
    # Common Kaggle nesting: /kaggle/input/<dataset>/corpora_cleaned
    nested = explicit / "corpora_cleaned"
    if nested.is_dir():
        return nested
    # Search under /kaggle/input for corpus dirs
    if _ON_KAGGLE and Path("/kaggle/input").is_dir():
        for child in Path("/kaggle/input").iterdir():
            if not child.is_dir():
                continue
            if any((child / c).is_dir() for c in CORPUS_DIRS):
                print(f"[eiv36] auto-detected corpora-root: {child}")
                return child
            nested2 = child / "corpora_cleaned"
            if nested2.is_dir() and any((nested2 / c).is_dir() for c in CORPUS_DIRS):
                print(f"[eiv36] auto-detected corpora-root: {nested2}")
                return nested2
    return explicit


def main(argv: list[str] | None = None) -> None:
    global CORPORA_ROOT, RESULTS_ROOT, MLP_DIR

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--corpora-root", type=Path, default=CORPORA_ROOT)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--mlp-dir", type=Path, default=MLP_DIR)
    parser.add_argument(
        "--heet",
        type=Path,
        default=DEFAULT_HEET,
        help="HEET CSV with human ground_truth_label (default: beside this script)",
    )
    parser.add_argument("--corpus", action="append", default=None)
    parser.add_argument("--device", type=str, default=None, help="Whisper device (default: cuda if available)")
    parser.add_argument(
        "--mlp-device",
        type=str,
        default="cpu",
        help="Device for the 41 MLP heads. Default cpu — each head's "
        "proj layer is ~295MB; 41 on a 16GB T4 OOMs beside Whisper.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--per-class-limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--neutral-band", type=float, default=0.5)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="First available corpus, 5 files only",
    )
    args = parser.parse_args(argv)

    CORPORA_ROOT = resolve_corpora_root(args.corpora_root)
    RESULTS_ROOT = args.results_root
    MLP_DIR = args.mlp_dir
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"[eiv36] corpora-root = {CORPORA_ROOT}")
    print(f"[eiv36] results-root = {RESULTS_ROOT}")
    print(f"[eiv36] mlp-dir      = {MLP_DIR}")
    print(f"[eiv36] emotion heads = {len(EMOTION_HEADS)} "
          f"(pos={len(POS_HEADS)} neu={len(NEU_HEADS)} neg={len(NEG_HEADS)})")

    human_gt = load_human_gt_labels(args.heet)
    print(f"[eiv36] human GT labels = {len(human_gt)} from {args.heet}")

    corpora = args.corpus or list(CORPUS_DIRS)
    limit = args.limit
    if args.dry_run:
        available = [c for c in corpora if (CORPORA_ROOT / c).is_dir()]
        if not available:
            raise FileNotFoundError(f"No corpora under {CORPORA_ROOT}")
        corpora = [available[0]]
        limit = 5 if limit is None else limit
        print(f"[eiv36] dry-run: {corpora[0]} limit={limit}")

    whisper_device = pick_device(args.device)
    mlp_device = pick_device(args.mlp_device)
    print(f"[eiv36] whisper={whisper_device}  mlps={mlp_device}")

    head_paths = download_heads(MLP_DIR)
    print(f"[eiv36] loading Whisper {WHISPER_MODEL_ID}")
    processor = WhisperProcessor.from_pretrained(WHISPER_MODEL_ID)
    whisper = (
        WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL_ID)
        .to(whisper_device)
        .eval()
    )
    # Load MLPs after Whisper so GPU isn't packed with 41×~295MB proj layers.
    print(f"[eiv36] loading {len(head_paths)} MLPs onto {mlp_device}")
    mlps = {stem: load_mlp(pth, mlp_device) for stem, pth in head_paths.items()}
    print(f"[eiv36] loaded {len(mlps)} MLPs")
    if whisper_device.type == "cuda":
        torch.cuda.empty_cache()

    for corpus in corpora:
        df = run_corpus(
            corpus,
            processor,
            whisper,
            mlps,
            whisper_device,
            mlp_device,
            limit,
            args.per_class_limit,
            max(1, args.batch_size),
            args.neutral_band,
            human_gt=human_gt,
        )
        if df.empty:
            continue
        out_dir = RESULTS_ROOT / corpus
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / "predictions.csv", index=False)
        metrics = summarize_corpus(df, args.neutral_band)
        with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"[eiv36] wrote {out_dir / 'predictions.csv'}")
        print(f"[eiv36] wrote {out_dir / 'metrics.json'}")

    print(f"\n[eiv36] inference done → {RESULTS_ROOT}")
    print("[eiv36] next: uv run python src/evaluators/eiv_summary_table.py")
    print("[eiv36] next: uv run python src/evaluators/eiv_all_emotions.py")


if __name__ == "__main__":
    main()
