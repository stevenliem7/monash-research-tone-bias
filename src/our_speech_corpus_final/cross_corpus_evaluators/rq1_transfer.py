#!/usr/bin/env python3
"""
RQ1: does our valence head, trained on synthetic Gemini Live speech, transfer to
human speech?

Fit one head on the Gemini WavLM embeddings, then score four corpora:
two human (IEMOCAP, TESS) and their synthetic counterparts. Ground truth is the
last token of the filename:

    iemocap_human_1_happy_positive.wav  ->  positive

Kaggle (GPU):

    !pip -q install transformers soundfile scipy scikit-learn
    !python rq1_transfer.py \
        --train-npz    /kaggle/input/<ds>/our_speech_corpus_microsoft_wavlm-large.npz \
        --manifest     /kaggle/input/<ds>/our_speech_corpus_manifest_split_80_20.csv \
        --corpora-root /kaggle/input/<ds>/corpora_cleaned \
        --out-dir      /kaggle/working/rq1

Smoke test first (20 clips/corpus):  add --limit 20
Embeddings are cached per corpus in --out-dir, so a re-run only re-fits the head.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

CLASSES = ("positive", "neutral", "negative")
CORPORA = (
    "iemocap_human_cleaned",
    "iemocap_synth_cleaned",
    "tess_human_cleaned",
    "tess_indextts_cleaned",
)
MODEL_ID = "microsoft/wavlm-large"
SR = 16_000


def label_from_name(wav: Path) -> str | None:
    """Parse valence from the last underscore token of a WAV stem.

    Args:
        wav: Path to a corpus WAV (e.g. ``iemocap_human_1_happy_positive.wav``).

    Returns:
        str | None: ``positive``, ``neutral``, or ``negative`` if the tail token
        is a valid class; otherwise None.
    """
    tail = wav.stem.rsplit("_", 1)[-1].strip().lower()
    return tail if tail in CLASSES else None


def build_head(kind: str, seed: int) -> Pipeline:
    """Build a sklearn pipeline for one valence-head candidate.

    Args:
        kind: ``l2_logistic`` or ``rbf_svc``.
        seed: Random state for the classifier.

    Returns:
        Pipeline: StandardScaler followed by the chosen classifier with
        balanced class weights.
    """
    clf = (
        SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", random_state=seed)
        if kind == "rbf_svc"
        else LogisticRegression(
            C=0.1, max_iter=2000, class_weight="balanced", random_state=seed
        )
    )
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def align_to_train(X: np.ndarray, mu_train: np.ndarray, sd_train: np.ndarray) -> np.ndarray:
    """Diagonal CORAL: re-centre/rescale target features onto the training cloud.

    Label-free unsupervised domain adaptation. Removes per-dimension mean/scale
    shift (speaker, mic, codec); cannot fix a genuine difference in what the
    classes sound like.

    Args:
        X: Target-corpus embedding matrix.
        mu_train: Per-dimension mean of the training embeddings.
        sd_train: Per-dimension std of the training embeddings (with epsilon).

    Returns:
        np.ndarray: Aligned target features with training mean and scale.
    """
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-8
    return ((X - mu) / sd) * sd_train + mu_train


def in_domain_cv(
    X: np.ndarray, y: list[str], kind: str, seed: int, folds: int = 5
) -> tuple[float, float]:
    """Ceiling: the same head trained and tested on this corpus via stratified CV.

    Separates "our head does not transfer" from "mean-pooled WavLM cannot do
    valence on this corpus at all". Runs on cached embeddings only (no GPU).

    Args:
        X: Corpus embedding matrix.
        y: String valence labels aligned with ``X``.
        kind: Head type passed to :func:`build_head`.
        seed: Random state for CV and the classifier.
        folds: Number of stratified folds.

    Returns:
        tuple[float, float]: (accuracy, macro_f1) from out-of-fold predictions.
    """
    pred = cross_val_predict(
        build_head(kind, seed),
        X,
        y,
        cv=StratifiedKFold(folds, shuffle=True, random_state=seed),
    )
    return (
        accuracy_score(y, pred),
        f1_score(y, pred, labels=list(CLASSES), average="macro", zero_division=0),
    )


def load_train(npz_path: Path, manifest_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Join cached Gemini embeddings to the labelled manifest rows.

    Args:
        npz_path: NPZ with ``clip_id`` and ``embeddings`` arrays.
        manifest_path: Split manifest CSV with ``is_labelled`` and labels.

    Returns:
        tuple[np.ndarray, np.ndarray]: (X_train, y_train) feature matrix and
        lowercase string labels.

    Raises:
        ValueError: If no labelled manifest rows match embedding clip_ids.
    """
    bundle = np.load(npz_path, allow_pickle=True)
    vecs = dict(
        zip(
            (str(c) for c in bundle["clip_id"]),
            np.asarray(bundle["embeddings"], dtype=np.float32),
        )
    )
    manifest = pd.read_csv(manifest_path)
    rows = manifest[manifest["is_labelled"] & manifest["clip_id"].astype(str).isin(vecs)]
    if rows.empty:
        raise ValueError("No labelled manifest rows matched the embedding clip_ids")
    X = np.vstack([vecs[str(c)] for c in rows["clip_id"]])
    y = rows["ground_truth_label"].astype(str).str.strip().str.lower().to_numpy()
    return X, y


def read_wav(path: Path) -> np.ndarray:
    """Load a WAV as mono float32 resampled to 16 kHz.

    Args:
        path: Path to the audio file.

    Returns:
        np.ndarray: 1-D waveform at :data:`SR`.
    """
    import soundfile as sf
    from scipy import signal

    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        wav = signal.resample(wav, int(round(len(wav) * SR / sr))).astype(np.float32)
    return wav


@lru_cache(maxsize=1)
def get_encoder(device: str):
    """Load WavLM feature extractor and model once per process.

    Args:
        device: Torch device string (e.g. ``cuda:0`` or ``cpu``).

    Returns:
        tuple: (processor, model) with the model in eval mode on ``device``.
    """
    from transformers import AutoFeatureExtractor, AutoModel

    print(f"[ssl] loading {MODEL_ID} on {device}")
    processor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(device).eval()
    return processor, model


def embed(wavs: list[Path], cache: Path, device: str) -> tuple[np.ndarray, list[str]]:
    """Mean-pool WavLM vectors for ``wavs``, reusing or writing ``cache``.

    Args:
        wavs: Ordered list of corpus WAV paths to embed.
        cache: NPZ path for ``clip_id`` / ``embeddings`` cache.
        device: Torch device for the SSL encoder.

    Returns:
        tuple[np.ndarray, list[str]]: Embedding matrix and kept clip stems
        (failed WAVs are skipped).
    """
    ids = [w.stem for w in wavs]
    if cache.is_file():
        bundle = np.load(cache, allow_pickle=True)
        cached_ids = [str(c) for c in bundle["clip_id"]]
        if cached_ids == ids:
            print(f"[emb] reuse {cache.name} ({len(ids)} clips)")
            return np.asarray(bundle["embeddings"], dtype=np.float32), cached_ids

    import torch

    processor, model = get_encoder(device)
    kept_ids: list[str] = []
    out: list[np.ndarray] = []
    with torch.inference_mode():
        for i, wav_path in enumerate(wavs, 1):
            try:
                inputs = processor(
                    read_wav(wav_path),
                    sampling_rate=SR,
                    return_tensors="pt",
                    padding=False,
                )
                # Mask is at sample rate; hidden states are ~320x shorter. Drop it.
                inputs.pop("attention_mask", None)
                hidden = model(
                    **{k: v.to(device) for k, v in inputs.items()}
                ).last_hidden_state
                out.append(hidden.mean(dim=1).squeeze(0).cpu().numpy().astype(np.float32))
                kept_ids.append(wav_path.stem)
            except Exception as exc:  # one bad WAV must not cost the whole corpus
                print(f"[emb] skip {wav_path.name}: {exc}")
            if i % 100 == 0:
                print(f"[emb] {cache.stem}: {i}/{len(wavs)}")

    X = np.vstack(out)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache, clip_id=np.array(kept_ids, dtype=object), embeddings=X
    )
    print(f"[emb] wrote {cache.name} ({len(kept_ids)} clips)")
    return X, kept_ids


def main() -> None:
    """CLI entrypoint for RQ1 transfer evaluation.

    Args:
        None. Command-line flags: ``--train-npz``, ``--manifest``, ``--corpora-root``,
        ``--out-dir``, ``--corpora``, ``--head``, ``--device``, ``--limit``, ``--seed``,
        ``--target-znorm``, and ``--in-domain-cv``.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--train-npz", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--corpora-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("rq1_transfer"))
    parser.add_argument("--corpora", nargs="+", default=list(CORPORA))
    parser.add_argument("--head", choices=("l2_logistic", "rbf_svc"), default="l2_logistic")
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None, help="clips per corpus (smoke)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--target-znorm",
        action="store_true",
        help="Mean/var-align each corpus onto the training cloud before predicting",
    )
    parser.add_argument(
        "--in-domain-cv",
        action="store_true",
        help="Also report the same head trained on the target corpus itself (ceiling)",
    )
    args = parser.parse_args()

    if args.device is None:
        import torch

        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train = load_train(args.train_npz, args.manifest)
    print(
        f"[fit] {args.head} on n={len(y_train)} Gemini clips  "
        f"{dict(zip(*np.unique(y_train, return_counts=True)))}"
    )
    head = build_head(args.head, args.seed).fit(X_train, y_train)
    mu_train, sd_train = X_train.mean(axis=0), X_train.std(axis=0) + 1e-8

    summary = []
    for name in args.corpora:
        corpus_dir = args.corpora_root / name
        if not corpus_dir.is_dir():
            raise FileNotFoundError(corpus_dir)
        wavs = [w for w in sorted(corpus_dir.glob("*.wav")) if label_from_name(w)]
        if args.limit:
            wavs = wavs[: args.limit]
        if not wavs:
            raise FileNotFoundError(f"No valence-tagged WAVs in {corpus_dir}")

        X, ids = embed(wavs, args.out_dir / f"{name}_wavlm-large.npz", args.device)
        truth = {w.stem: label_from_name(w) for w in wavs}
        y_true = [truth[i] for i in ids]
        # In-domain CV uses the raw target features: alignment is a no-op there.
        X_scored = align_to_train(X, mu_train, sd_train) if args.target_znorm else X
        y_pred = head.predict(X_scored)

        print(f"\n=== {name}  (n={len(ids)}) ===")
        print(classification_report(
            y_true, y_pred, labels=list(CLASSES), digits=3, zero_division=0
        ))
        print("confusion (rows=true, cols=pred), order " + ", ".join(CLASSES))
        print(confusion_matrix(y_true, y_pred, labels=list(CLASSES)))
        pd.DataFrame({"clip_id": ids, "true": y_true, "pred": y_pred}).to_csv(
            args.out_dir / f"{name}_predictions.csv", index=False
        )
        row = {
            "corpus": name,
            "n": len(ids),
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(
                y_true, y_pred, labels=list(CLASSES), average="macro", zero_division=0
            ),
        }
        if args.in_domain_cv:
            acc, macro = in_domain_cv(X, y_true, args.head, args.seed)
            row["in_domain_acc"] = acc
            row["in_domain_macro_f1"] = macro
            row["transfer_gap"] = macro - row["macro_f1"]
            print(f"[ceiling] in-domain 5-fold: acc={acc:.3f} macroF1={macro:.3f}")
        summary.append(row)

    table = pd.DataFrame(summary)
    table.to_csv(args.out_dir / "rq1_summary.csv", index=False)
    print(
        f"\n[summary] head={args.head}  target_znorm={args.target_znorm}  "
        f"in_domain_cv={args.in_domain_cv}"
    )
    print(table.to_string(index=False, float_format="%.3f"))
    print(f"\n[wrote] {args.out_dir}")


def _self_test() -> None:
    """Run lightweight assertions on label parsing and CORAL alignment.

    Args:
        None.

    Returns:
        None
    """
    assert label_from_name(Path("iemocap_human_1_happy_positive.wav")) == "positive"
    assert label_from_name(Path("some_untagged_clip.wav")) is None
    rng = np.random.default_rng(0)
    train = rng.normal(3.0, 2.0, size=(200, 8))
    target = rng.normal(-5.0, 0.5, size=(150, 8))
    aligned = align_to_train(target, train.mean(0), train.std(0) + 1e-8)
    assert np.allclose(aligned.mean(0), train.mean(0), atol=1e-6)
    assert np.allclose(aligned.std(0), train.std(0), atol=1e-3)
    print("self-test ok")

if __name__ == "__main__":
    import sys

    if "--self-test" in sys.argv:
        _self_test()
    else:
        main()
