#!/usr/bin/env python3
"""
Authors:
    Steven Liem (steven.liem@sydney.edu.au)

Repeated stratified cross-validation harness for the 3-class valence head.

Motivation
----------
A single 80/20 split on 95 negatives produced negative-recall estimates ranging
from 0.167 to 0.667 across train-fraction sweeps. That range was NOT driven by
the split fraction: it tracked which candidate won CV selection, and the winners
differed in *class weighting* rather than capacity:

  - LogisticRegression(class_weight="balanced")     -> neg recall 0.46-0.67
  - LinearDiscriminantAnalysis(priors=empirical)    -> neg recall 0.17-0.22

CV margins deciding between them were ~0.004-0.009 against fold sds of
0.013-0.046, i.e. pure noise. This harness fixes three things:

  1. MATCHED OPERATING POINTS. Every candidate is evaluated under BOTH weighting
     regimes ("balanced" and "empirical"). The operating point becomes a
     documented research decision, not a noisy CV tiebreak.
  2. REPEATED CV. RepeatedStratifiedKFold replaces the single split, so every
     negative is evaluated `--repeats` times and metrics are reported as
     mean +/- sd across repeats rather than one high-variance number.
  3. PER-EMOTION POOLING. Out-of-fold predictions are pooled so negative recall
     can be broken down by emotion_label (disgust / sad / angry / fearful),
     which a 14-28 clip test split cannot support.

Also adds the two missing baselines: a nonlinear rung (PCA -> MLP) and a
trivial-acoustics baseline (duration / RMS / ZCR), plus majority class.

Extended candidates (reported alongside the original four; not auto-selected):
  - rbf_svc      RBF-SVM (n<d workhorse; no calibrated probs)
  - pls_da       PLS-DA (supervised reduction vs PCA variance axes)
  - linear_svc   LinearSVC hinge (different inductive bias vs logistic)
  - hist_gbm     Hist gradient boosting (negative control on trees)
  - focal_mlp    Tuned MLP + class-balanced focal loss (PyTorch; skipped if
                 torch is unavailable)

Default protocol is 5 folds x 10 *repeats* (not 10-fold CV). With 95
negatives, 5-fold keeps ~19 negatives per test fold; 10 repeats mean every
clip is OOF-tested 10 times so metrics are mean+/-sd rather than one noisy
partition.

Group-aware CV (blocking check for lexical leakage)
----------------------------------------------------
WavLM encodes spoken lexical content; prompts are heavily templated. Pass
``--group-by response_text`` (spoken AI text) or ``--group-by question``
(prompt template) to use StratifiedGroupKFold so near-duplicates cannot sit
in both train and test. Compare macro-F1 to the ungrouped run: if it holds,
numbers are clean; if it drops, the random-split grid was inflated.

IMPORTANT — this script does NOT select a winner.
It reports the full grid. Selection is a decision you make deliberately and
document. Picking the max cell here would repeat the original error.

IMPORTANT — scope of these numbers.
This runs on ALL labelled rows (n=854). Because the previous held-out split was
consulted six times during the train-fraction sweep, its one-shot guarantee is
already spent; these are DEVELOPMENT estimates. A final confirmatory number
requires either newly labelled clips or a pre-registered re-lock.

Usage:
    uv run python src/our_speech_corpus_final/evaluate_valence_head_cv.py
    uv run python src/our_speech_corpus_final/evaluate_valence_head_cv.py \\
        --repeats 10 --folds 5 --acoustic-features
    # Extended grid on WavLM embeddings:
    uv run python src/our_speech_corpus_final/train_valence_head_cv.py \\
        --embeddings /path/to/our_speech_corpus_microsoft_wavlm-large.npz \\
        --manifest /path/to/manifest_remapped.csv \\
        --output /path/to/valence_head_cv_extended.json
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
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from config import bootstrap_imports, extend_our_speech_corpus_paths

bootstrap_imports(__file__)
extend_our_speech_corpus_paths()
from config import (  # noqa: E402
    EMOTION2VEC_CV_RESULTS_JSON,
    EMBEDDINGS_NPZ,
    SPLIT_MANIFEST_CSV,
    SPLIT_SEED,
    VALENCE_CLASSES,
)

LABEL_TO_ID = {label: i for i, label in enumerate(VALENCE_CLASSES)}
ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}
NEG_ID = LABEL_TO_ID["negative"]

CV_RESULTS_JSON = EMOTION2VEC_CV_RESULTS_JSON


# --------------------------------------------------------------------------
# statistics helpers
# --------------------------------------------------------------------------
def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Args:
        k: Number of successes.
        n: Number of trials.
        z: Normal quantile (1.96 = 95%).

    Returns:
        tuple[float, float]: (lower, upper) bounds clipped to [0, 1].
    """
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z / denom * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


# --------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------
def load_labelled(
    embeddings_path: Path,
    manifest_path: Path,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Join cached embeddings with all labelled manifest rows.

    Args:
        embeddings_path: NPZ containing clip_id and embeddings arrays.
        manifest_path: Unified manifest CSV with ground_truth_label.

    Returns:
        tuple: (X, y, frame) where X is (n, d) float32, y is integer class ids,
        and frame holds the aligned manifest rows.

    Raises:
        ValueError: If no rows align or a label is outside VALENCE_CLASSES.
    """
    bundle = np.load(embeddings_path, allow_pickle=True)
    emb_ids = [str(x) for x in bundle["clip_id"].tolist()]
    emb_matrix = np.asarray(bundle["embeddings"], dtype=np.float32)
    lookup = {cid: emb_matrix[i] for i, cid in enumerate(emb_ids)}

    manifest = pd.read_csv(manifest_path)
    labelled = manifest[manifest["is_labelled"]].copy()

    vectors: list[np.ndarray] = []
    labels: list[int] = []
    keep: list[int] = []
    for idx, row in labelled.iterrows():
        clip_id = str(row["clip_id"])
        gt = str(row["ground_truth_label"]).strip().lower()
        if clip_id not in lookup:
            continue
        if gt not in LABEL_TO_ID:
            raise ValueError(f"Unexpected ground_truth_label={gt!r} for {clip_id}")
        vectors.append(lookup[clip_id])
        labels.append(LABEL_TO_ID[gt])
        keep.append(idx)

    if not vectors:
        raise ValueError("No labelled rows matched embeddings by clip_id")

    frame = labelled.loc[keep].reset_index(drop=True)
    return np.vstack(vectors), np.asarray(labels, dtype=np.int64), frame


def build_acoustic_features(
    frame: pd.DataFrame,
    audio_col: str = "audio_path",
) -> np.ndarray | None:
    """Compute trivial acoustic features (duration, RMS, ZCR) per clip.

    This is the blocking sanity baseline: if a head on frozen embeddings cannot
    beat duration + loudness, the embedding is contributing nothing.

    Args:
        frame: Manifest rows aligned with the feature matrix.
        audio_col: Column holding the audio file path.

    Returns:
        np.ndarray | None: (n, 3) feature matrix, or None if audio is unreadable.
    """
    try:
        import soundfile as sf
    except ImportError:
        print("[acoustic] soundfile not installed - skipping trivial baseline")
        return None

    if audio_col not in frame.columns:
        print(f"[acoustic] no {audio_col} column - skipping trivial baseline")
        return None

    feats: list[list[float]] = []
    failures = 0
    for path in frame[audio_col].astype(str):
        try:
            wav, sr = sf.read(path, dtype="float32", always_2d=False)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if wav.size == 0:
                raise ValueError("empty")
            duration = wav.size / float(sr)
            rms = float(np.sqrt(np.mean(wav**2)))
            zcr = float(np.mean(np.abs(np.diff(np.sign(wav))) > 0))
            feats.append([duration, rms, zcr])
        except Exception:
            failures += 1
            feats.append([np.nan, np.nan, np.nan])

    arr = np.asarray(feats, dtype=np.float64)
    if failures:
        print(f"[acoustic] {failures} clips unreadable - imputing column medians")
        medians = np.nanmedian(arr, axis=0)
        idx = np.where(np.isnan(arr))
        arr[idx] = np.take(medians, idx[1])
    if np.isnan(arr).any():
        print("[acoustic] features still contain NaN - skipping trivial baseline")
        return None
    return arr


# --------------------------------------------------------------------------
# candidates
# --------------------------------------------------------------------------
class PLSDA(BaseEstimator, ClassifierMixin):
    """Multiclass PLS-DA: PLSRegression on one-hot labels, predict by argmax.

    Finds directions of max covariance with the label encoding (unlike PCA,
    which maximises variance alone). ``n_components`` is capped by
    ``min(n_samples-1, n_features, n_classes)`` at fit time when needed.
    """

    def __init__(self, n_components: int = 16):
        """Configure PLS-DA with a target number of latent components.

        Args:
            n_components: Requested PLS components (capped at fit time).

        Returns:
            None
        """
        self.n_components = n_components

    def fit(self, X, y):
        """Fit PLS regression on one-hot encoded class labels.

        Args:
            X: Training feature matrix.
            y: Integer class labels.

        Returns:
            self: Fitted estimator with ``pls_`` and ``classes_`` attributes.
        """
        X, y = check_X_y(X, y, dtype=np.float64)
        self.classes_ = np.unique(y)
        self._lb = LabelBinarizer()
        Y = self._lb.fit_transform(y)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        n_comp = min(self.n_components, X.shape[0] - 1, X.shape[1], Y.shape[1])
        n_comp = max(1, n_comp)
        self.pls_ = PLSRegression(n_components=n_comp, scale=False)
        self.pls_.fit(X, Y)
        return self

    def predict(self, X):
        """Predict class labels by argmax over PLS regression scores.

        Args:
            X: Feature matrix.

        Returns:
            np.ndarray: Integer predicted class ids.
        """
        check_is_fitted(self, "pls_")
        X = check_array(X, dtype=np.float64)
        scores = self.pls_.predict(X)
        if scores.ndim == 1:
            scores = scores.reshape(-1, 1)
        # LabelBinarizer for 3 classes yields (n, 3); argmax recovers class id.
        idx = np.argmax(scores, axis=1)
        return self._lb.classes_[idx]


class FocalMLP(BaseEstimator, ClassifierMixin):
    """Small PyTorch MLP trained with class-balanced focal loss.

    Kept deliberately modest (dropout + early stop) given only ~95 negatives.
    Requires torch; raise ImportError from fit if missing.
    """

    def __init__(
        self,
        hidden=(128, 64),
        dropout: float = 0.3,
        gamma: float = 2.0,
        balanced_alpha: bool = True,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 60,
        batch_size: int = 64,
        patience: int = 10,
        seed: int = 42,
    ):
        """Configure a small MLP trained with class-balanced focal loss.

        Args:
            hidden: Hidden layer widths.
            dropout: Dropout probability between hidden layers.
            gamma: Focal-loss focusing parameter.
            balanced_alpha: If True, use inverse-frequency class alphas.
            lr: AdamW learning rate.
            weight_decay: AdamW weight decay.
            max_epochs: Maximum training epochs.
            batch_size: Mini-batch size.
            patience: Early-stopping patience on validation focal loss.
            seed: RNG seed for weight init and train/val split.

        Returns:
            None
        """
        self.hidden = hidden
        self.dropout = dropout
        self.gamma = gamma
        self.balanced_alpha = balanced_alpha
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.seed = seed

    def fit(self, X, y):
        """Train the MLP with focal loss and internal validation early stopping.

        Args:
            X: Training feature matrix.
            y: Integer class labels.

        Returns:
            self: Fitted estimator with ``net_`` on the chosen device.

        Raises:
            ImportError: If PyTorch is not installed.
        """
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError as exc:
            raise ImportError(
                "focal_mlp requires torch; pip/uv install torch to enable it"
            ) from exc

        X, y = check_X_y(X, y, dtype=np.float32)
        self.classes_ = np.unique(y)
        n_classes = int(self.classes_.max()) + 1
        n_features = X.shape[1]

        torch.manual_seed(self.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        layers: list[nn.Module] = []
        prev = n_features
        for width in self.hidden:
            layers.extend(
                [nn.Linear(prev, width), nn.ReLU(), nn.Dropout(self.dropout)]
            )
            prev = width
        layers.append(nn.Linear(prev, n_classes))
        net = nn.Sequential(*layers).to(device)

        counts = np.bincount(y, minlength=n_classes).astype(np.float64)
        counts = np.maximum(counts, 1.0)
        if self.balanced_alpha:
            alpha = (counts.sum() / (n_classes * counts)).astype(np.float32)
        else:
            alpha = np.ones(n_classes, dtype=np.float32)
        alpha_t = torch.tensor(alpha, device=device)

        gamma = self.gamma

        def focal_loss(logits, target):
            """Compute class-balanced focal loss for one mini-batch.

            Args:
                logits: Raw classifier outputs, shape (batch, n_classes).
                target: Integer class indices, shape (batch,).

            Returns:
                torch.Tensor: Scalar mean focal loss.
            """
            log_p = nn.functional.log_softmax(logits, dim=1)
            p = log_p.exp()
            gather_log = log_p.gather(1, target.unsqueeze(1)).squeeze(1)
            gather_p = p.gather(1, target.unsqueeze(1)).squeeze(1)
            at = alpha_t.gather(0, target)
            return (-(at * (1.0 - gather_p) ** gamma * gather_log)).mean()

        # Hold out 15% of the *training fold* for early stopping only.
        rng = np.random.default_rng(self.seed)
        n = len(y)
        perm = rng.permutation(n)
        n_val = max(1, int(0.15 * n))
        val_idx, tr_idx = perm[:n_val], perm[n_val:]

        def _loader(idx, shuffle):
            """Build a DataLoader for a row-index subset.

            Args:
                idx: Integer row indices into ``X`` and ``y``.
                shuffle: Whether to shuffle batches.

            Returns:
                DataLoader: Batched (features, labels) tensors.
            """
            ds = TensorDataset(
                torch.from_numpy(X[idx]),
                torch.from_numpy(y[idx].astype(np.int64)),
            )
            return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

        train_loader = _loader(tr_idx, True)
        val_x = torch.from_numpy(X[val_idx]).to(device)
        val_y = torch.from_numpy(y[val_idx].astype(np.int64)).to(device)

        opt = torch.optim.AdamW(
            net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        best_state = None
        best_val = float("inf")
        stale = 0
        net.train()
        for _ in range(self.max_epochs):
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = focal_loss(net(xb), yb)
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                val_loss = float(focal_loss(net(val_x), val_y).item())
            net.train()
            if val_loss < best_val - 1e-4:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break

        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        self.net_ = net
        self.device_ = device
        self.n_features_in_ = n_features
        return self

    def predict(self, X):
        """Predict class labels from the trained MLP.

        Args:
            X: Feature matrix.

        Returns:
            np.ndarray: Integer predicted class ids.
        """
        check_is_fitted(self, "net_")
        import torch

        X = check_array(X, dtype=np.float32)
        self.net_.eval()
        with torch.no_grad():
            logits = self.net_(torch.from_numpy(X).to(self.device_))
            pred = logits.argmax(dim=1).cpu().numpy()
        return pred


def build_candidates(balanced: bool, seed: int) -> dict[str, dict]:
    """Build candidate heads at a single, explicit operating point.

    All candidates share the same nominal class-weighting regime so that CV
    compares CAPACITY, not operating point. Mechanisms differ by estimator
    (class_weight / priors / oversampling / focal alpha) but the nominal
    regime is matched.

    Args:
        balanced: If True, weight classes to uniform; else use empirical priors.
        seed: Random state for solvers and PCA.

    Returns:
        dict[str, dict]: name -> {"model": estimator, "oversample": bool}.
    """
    class_weight = "balanced" if balanced else None
    priors = np.array([1 / 3, 1 / 3, 1 / 3]) if balanced else None

    candidates: dict[str, dict] = {
        "pca32_logistic": {
            "model": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("pca", PCA(n_components=32, random_state=seed)),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=2000,
                            class_weight=class_weight,
                            random_state=seed,
                        ),
                    ),
                ]
            ),
            "oversample": False,
        },
        "l2_logistic": {
            "model": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            C=0.1,
                            max_iter=2000,
                            class_weight=class_weight,
                            random_state=seed,
                        ),
                    ),
                ]
            ),
            "oversample": False,
        },
        "shrinkage_lda": {
            "model": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LinearDiscriminantAnalysis(
                            solver="lsqr", shrinkage="auto", priors=priors
                        ),
                    ),
                ]
            ),
            "oversample": False,
        },
        "pca32_mlp": {
            "model": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("pca", PCA(n_components=32, random_state=seed)),
                    (
                        "clf",
                        MLPClassifier(
                            hidden_layer_sizes=(32,),
                            alpha=1.0,
                            max_iter=1500,
                            random_state=seed,
                        ),
                    ),
                ]
            ),
            "oversample": balanced,
        },
        # --- extended grid ---
        "rbf_svc": {
            "model": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        SVC(
                            kernel="rbf",
                            C=1.0,
                            gamma="scale",
                            class_weight=class_weight,
                            random_state=seed,
                        ),
                    ),
                ]
            ),
            "oversample": False,
        },
        "pls_da": {
            "model": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", PLSDA(n_components=16)),
                ]
            ),
            # PLS has no class_weight; match regime via oversampling.
            "oversample": balanced,
        },
        "linear_svc": {
            "model": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LinearSVC(
                            C=0.1,
                            class_weight=class_weight,
                            dual="auto",
                            max_iter=5000,
                            random_state=seed,
                        ),
                    ),
                ]
            ),
            "oversample": False,
        },
        "hist_gbm": {
            "model": HistGradientBoostingClassifier(
                max_depth=3,
                learning_rate=0.05,
                max_iter=200,
                class_weight=class_weight,
                random_state=seed,
            ),
            "oversample": False,
        },
    }

    try:
        import torch  # noqa: F401

        candidates["focal_mlp"] = {
            "model": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        FocalMLP(
                            balanced_alpha=balanced,
                            seed=seed,
                        ),
                    ),
                ]
            ),
            "oversample": False,
        }
    except ImportError:
        print("[note] torch not installed - skipping focal_mlp")

    return candidates


def oversample_to_balance(
    X: np.ndarray, y: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Duplicate minority-class rows until all classes match the largest.

    Used for estimators that expose no class_weight parameter (MLPClassifier).

    Args:
        X: Training features.
        y: Training labels.
        rng: Random generator for sampling with replacement.

    Returns:
        tuple[np.ndarray, np.ndarray]: Balanced (X, y).
    """
    counts = np.bincount(y, minlength=len(VALENCE_CLASSES))
    target = counts.max()
    parts_x, parts_y = [X], [y]
    for cls, count in enumerate(counts):
        if count == 0 or count >= target:
            continue
        idx = np.flatnonzero(y == cls)
        extra = rng.choice(idx, size=target - count, replace=True)
        parts_x.append(X[extra])
        parts_y.append(y[extra])
    return np.vstack(parts_x), np.concatenate(parts_y)


# --------------------------------------------------------------------------
# cross-validation
# --------------------------------------------------------------------------
def normalise_text(value: object) -> str:
    """Lowercase + collapse whitespace for group keys.

    Args:
        value: Raw cell (question or response_text).

    Returns:
        str: Normalised text, or empty string if missing.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return " ".join(str(value).lower().split())


def group_labels_from_frame(frame: pd.DataFrame, group_by: str) -> np.ndarray:
    """Build integer group ids from a manifest column.

    Args:
        frame: Labelled rows aligned with X/y.
        group_by: ``response_text`` or ``question``.

    Returns:
        np.ndarray: Integer group id per row.

    Raises:
        ValueError: If the column is missing.
    """
    if group_by not in frame.columns:
        raise ValueError(f"--group-by {group_by!r} but column missing from manifest")
    texts = frame[group_by].map(normalise_text)
    # Empty / missing text: unique singleton group per row (no forced merge).
    keys: list[str] = []
    for i, t in enumerate(texts):
        keys.append(t if t else f"__empty_{i}__")
    codes, uniques = pd.factorize(np.asarray(keys, dtype=object), sort=False)
    print(
        f"[groups] by={group_by}  unique={len(uniques)} / n={len(frame)}  "
        f"max_group={int(np.bincount(codes).max())}"
    )
    if group_by == "response_text" and len(uniques) > 0.95 * len(frame):
        print(
            "[groups] note: response_text is nearly unique; lexical leakage is "
            "small. Also run --group-by question (templated prompts) — that is "
            "usually the harder leakage test."
        )
    return codes.astype(np.int64)


def run_repeated_cv(
    X: np.ndarray,
    y: np.ndarray,
    candidate: dict,
    folds: int,
    repeats: int,
    seed: int,
    groups: np.ndarray | None = None,
) -> tuple[list[dict], np.ndarray]:
    """Run repeated CV, returning per-repeat metrics and OOF predictions.

    Ungrouped: RepeatedStratifiedKFold.
    Grouped: StratifiedGroupKFold repeated with distinct shuffle seeds so
    near-duplicate text (same group) never appears in both train and test.

    Args:
        X: Feature matrix.
        y: Integer labels.
        candidate: Entry from build_candidates.
        folds: Number of stratified folds per repeat.
        repeats: Number of CV repetitions.
        seed: Random state for the fold generator and oversampling.
        groups: Optional integer group ids (same length as y).

    Returns:
        tuple: (per_repeat_metrics, oof_predictions) where oof_predictions has
        shape (repeats, n_samples).
    """
    oof = np.full((repeats, len(y)), -1, dtype=np.int64)
    rng = np.random.default_rng(seed)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        if groups is None:
            cv = RepeatedStratifiedKFold(
                n_splits=folds, n_repeats=repeats, random_state=seed
            )
            splits = list(cv.split(X, y))
        else:
            splits = []
            for r in range(repeats):
                cv = StratifiedGroupKFold(
                    n_splits=folds, shuffle=True, random_state=seed + r
                )
                splits.extend(cv.split(X, y, groups))

        for fold_i, (tr, te) in enumerate(splits):
            repeat_i = fold_i // folds
            X_tr, y_tr = X[tr], y[tr]
            if candidate["oversample"]:
                X_tr, y_tr = oversample_to_balance(X_tr, y_tr, rng)
            model = clone(candidate["model"])
            model.fit(X_tr, y_tr)
            oof[repeat_i, te] = model.predict(X[te])

    # Guard: every sample must be predicted once per repeat.
    if (oof < 0).any():
        missing = int((oof < 0).sum())
        raise RuntimeError(
            f"OOF incomplete ({missing} unfilled cells) — group sizes may be "
            "incompatible with n_splits; try fewer folds or inspect groups"
        )

    per_repeat: list[dict] = []
    for r in range(repeats):
        pred = oof[r]
        prec, rec, f1, _ = precision_recall_fscore_support(
            y, pred, labels=list(range(len(VALENCE_CLASSES))), zero_division=0
        )
        per_repeat.append(
            {
                "accuracy": float(accuracy_score(y, pred)),
                "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
                "weighted_f1": float(
                    f1_score(y, pred, average="weighted", zero_division=0)
                ),
                "per_class": {
                    ID_TO_LABEL[i]: {
                        "precision": float(prec[i]),
                        "recall": float(rec[i]),
                        "f1": float(f1[i]),
                    }
                    for i in range(len(VALENCE_CLASSES))
                },
            }
        )
    return per_repeat, oof


def summarise(per_repeat: list[dict]) -> dict:
    """Reduce per-repeat metrics to mean and sd.

    Args:
        per_repeat: Output of run_repeated_cv.

    Returns:
        dict: Mean/sd for headline metrics and per-class recall/precision/F1.
    """

    def ms(values: list[float]) -> dict:
        """Compute mean and sample standard deviation.

        Args:
            values: Scalar metric values across CV repeats.

        Returns:
            dict: ``mean`` and ``sd`` keys.
        """
        arr = np.asarray(values, dtype=float)
        return {"mean": float(arr.mean()), "sd": float(arr.std(ddof=1))}

    out = {
        "n_repeats": len(per_repeat),
        "accuracy": ms([r["accuracy"] for r in per_repeat]),
        "macro_f1": ms([r["macro_f1"] for r in per_repeat]),
        "weighted_f1": ms([r["weighted_f1"] for r in per_repeat]),
        "per_class": {},
    }
    for label in VALENCE_CLASSES:
        out["per_class"][label] = {
            metric: ms([r["per_class"][label][metric] for r in per_repeat])
            for metric in ("precision", "recall", "f1")
        }
    return out


def per_emotion_negative_recall(
    y: np.ndarray,
    oof: np.ndarray,
    frame: pd.DataFrame,
    emotion_col: str = "emotion_label",
) -> dict:
    """Break negative-class recall down by input emotion, pooled across repeats.

    Args:
        y: True integer labels.
        oof: (repeats, n_samples) out-of-fold predictions.
        frame: Manifest rows aligned with y.
        emotion_col: Column holding the injected user emotion.

    Returns:
        dict: emotion -> {n_clips, mean_recall, sd_recall, wilson_ci}.
    """
    if emotion_col not in frame.columns:
        return {"error": f"no {emotion_col} column in manifest"}

    neg_mask = y == NEG_ID
    emotions = frame.loc[neg_mask, emotion_col].fillna("unknown").astype(str).to_numpy()
    repeats = oof.shape[0]

    buckets: dict[str, list[float]] = defaultdict(list)
    counts: dict[str, int] = {}
    for emotion in sorted(set(emotions)):
        sel = emotions == emotion
        counts[emotion] = int(sel.sum())
        for r in range(repeats):
            pred_neg = oof[r][neg_mask][sel] == NEG_ID
            buckets[emotion].append(float(pred_neg.mean()))

    out: dict = {}
    for emotion, values in buckets.items():
        arr = np.asarray(values)
        n_clips = counts[emotion]
        # CI computed on DISTINCT clips, not clips x repeats: repeats are not
        # independent trials and would give a spuriously narrow interval.
        lo, hi = wilson_ci(int(round(arr.mean() * n_clips)), n_clips)
        out[emotion] = {
            "n_clips": n_clips,
            "mean_recall": float(arr.mean()),
            "sd_recall": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "wilson_ci_on_clips": [lo, hi],
            "directional_only": n_clips < 5,
        }
    return out


def majority_baseline(y: np.ndarray) -> dict:
    """Always-predict-majority baseline on the full labelled set.

    Args:
        y: Integer labels.

    Returns:
        dict: Accuracy, macro-F1 and the predicted class name.
    """
    majority = int(np.bincount(y).argmax())
    pred = np.full_like(y, majority)
    return {
        "predicted_class": ID_TO_LABEL[majority],
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "negative_recall": 0.0,
    }


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def evaluate_all(
    embeddings_path: Path = EMBEDDINGS_NPZ,
    manifest_path: Path = SPLIT_MANIFEST_CSV,
    output_path: Path = CV_RESULTS_JSON,
    folds: int = 5,
    repeats: int = 10,
    seed: int = SPLIT_SEED,
    acoustic: bool = False,
    only: tuple[str, ...] | None = None,
    group_by: str | None = None,
) -> dict:
    """Run the full grid: candidates x operating points, plus baselines.

    Default ``folds=5``, ``repeats=10`` (RepeatedStratifiedKFold) because:
      - With ~100 negatives, 5-fold leaves ~20 negatives in each test fold —
        enough for a stable per-fold neg-recall signal. 10-fold would leave
        ~9–10 and inflate fold noise; 2–3 fold starves the train side.
      - A single 5-fold pass still has high partition variance (fold sds of
        macro-F1 were 0.01–0.05 in earlier sweeps). ``repeats=10`` evaluates
        every clip out-of-fold 10 times and reports mean±sd across repeats.
        Diminishing returns past ~10–20 repeats for this n; 10 keeps wall
        time practical once SVM / focal MLP join the grid.
      - Total fits/config = folds × repeats = 50 — not 10-fold CV.

    Pass ``group_by='response_text'`` or ``'question'`` to switch to
    StratifiedGroupKFold (lexical / template leakage check).

    Args:
        embeddings_path: Cached embeddings NPZ.
        manifest_path: Unified manifest with labels.
        output_path: Destination JSON.
        folds: Stratified folds per repeat.
        repeats: CV repetitions (not the same as n_splits).
        seed: Random state.
        acoustic: Whether to add the trivial-acoustics baseline.
        only: If set, restrict to these candidate names (still both regimes).
        group_by: Optional column for group-aware CV.

    Returns:
        dict: Full results payload.
    """
    X, y, frame = load_labelled(embeddings_path, manifest_path)
    counts = {ID_TO_LABEL[i]: int(c) for i, c in enumerate(np.bincount(y))}
    groups = group_labels_from_frame(frame, group_by) if group_by else None
    cv_name = (
        f"StratifiedGroupKFold(group_by={group_by})"
        if group_by
        else "RepeatedStratifiedKFold"
    )
    print(f"[data] n={len(y)} dim={X.shape[1]} classes={counts}")
    print(
        f"[cv]   {cv_name}  {repeats} repeats x {folds} folds = "
        f"{repeats * folds} fits/config"
    )
    print("[note] development estimates: prior held-out split already consulted\n")

    results: dict = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "embeddings_path": str(embeddings_path),
        "manifest_path": str(manifest_path),
        "embedding_dim": int(X.shape[1]),
        "n_labelled": int(len(y)),
        "class_counts": counts,
        "protocol": {
            "folds": folds,
            "repeats": repeats,
            "seed": seed,
            "cv": cv_name,
            "group_by": group_by,
            "selection": "none - full grid reported, selection is a manual decision",
            "only": list(only) if only else None,
        },
        "majority_baseline": majority_baseline(y),
        "grid": {},
    }

    for balanced in (True, False):
        regime = "balanced" if balanced else "empirical"
        results["grid"][regime] = {}
        candidates = build_candidates(balanced, seed)
        if only:
            missing = sorted(set(only) - set(candidates))
            if missing:
                raise ValueError(
                    f"Unknown candidate(s) {missing}; known={sorted(candidates)}"
                )
            candidates = {k: candidates[k] for k in only if k in candidates}
        for name, candidate in candidates.items():
            per_repeat, oof = run_repeated_cv(
                X, y, candidate, folds, repeats, seed, groups=groups
            )
            summary = summarise(per_repeat)
            summary["per_emotion_negative_recall"] = per_emotion_negative_recall(
                y, oof, frame
            )
            results["grid"][regime][name] = summary
            neg = summary["per_class"]["negative"]["recall"]
            print(
                f"[{regime:>9}] {name:<16} "
                f"acc={summary['accuracy']['mean']:.3f}±{summary['accuracy']['sd']:.3f}  "
                f"macF1={summary['macro_f1']['mean']:.3f}±{summary['macro_f1']['sd']:.3f}  "
                f"negR={neg['mean']:.3f}±{neg['sd']:.3f}"
            )
        print()

    if acoustic:
        feats = build_acoustic_features(frame)
        if feats is not None:
            trivial = {
                "model": Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "clf",
                            LogisticRegression(
                                max_iter=2000,
                                class_weight="balanced",
                                random_state=seed,
                            ),
                        ),
                    ]
                ),
                "oversample": False,
            }
            per_repeat, _ = run_repeated_cv(
                feats, y, trivial, folds, repeats, seed, groups=groups
            )
            results["trivial_acoustic_baseline"] = summarise(per_repeat)
            summary = results["trivial_acoustic_baseline"]
            print(
                f"[ trivial ] duration+rms+zcr   "
                f"acc={summary['accuracy']['mean']:.3f}±{summary['accuracy']['sd']:.3f}  "
                f"macF1={summary['macro_f1']['mean']:.3f}±{summary['macro_f1']['sd']:.3f}  "
                f"negR={summary['per_class']['negative']['recall']['mean']:.3f}\n"
            )

    print("[per-emotion negative recall, balanced regime]")
    for name, summary in results["grid"]["balanced"].items():
        breakdown = summary["per_emotion_negative_recall"]
        if "error" in breakdown:
            print(f"  {name}: {breakdown['error']}")
            continue
        parts = []
        for emotion, stats in sorted(breakdown.items()):
            flag = "*" if stats["directional_only"] else ""
            parts.append(
                f"{emotion}={stats['mean_recall']:.2f}(n={stats['n_clips']}){flag}"
            )
        print(f"  {name:<16} " + "  ".join(parts))
    print("  * n<5 clips: directional only, do not report as a rate\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[wrote] {output_path}")
    print(
        "\n[reminder] This script reports; it does not select. Choose the "
        "operating point deliberately and document the choice."
    )
    return results


def main() -> None:
    """CLI entrypoint for repeated stratified CV valence-head evaluation.

    Args:
        None. Command-line flags: ``--embeddings``, ``--manifest``, ``--output``,
        ``--folds``, ``--repeats``, ``--seed``, ``--acoustic-features``, ``--only``,
        and ``--group-by``.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description="Repeated stratified CV for the valence head"
    )
    parser.add_argument("--embeddings", type=Path, default=EMBEDDINGS_NPZ)
    parser.add_argument("--manifest", type=Path, default=SPLIT_MANIFEST_CSV)
    parser.add_argument("--output", type=Path, default=CV_RESULTS_JSON)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="CV repetitions (default 10). Not 10-fold CV.",
    )
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    parser.add_argument(
        "--acoustic-features",
        action="store_true",
        help="Add the trivial duration/RMS/ZCR baseline (requires soundfile)",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help=(
            "Restrict to these candidate names, e.g. "
            "--only rbf_svc pls_da linear_svc hist_gbm focal_mlp"
        ),
    )
    parser.add_argument(
        "--group-by",
        choices=("response_text", "question"),
        default=None,
        help=(
            "StratifiedGroupKFold on normalised text. "
            "response_text = spoken AI lexical leakage; "
            "question = templated prompt leakage (usually stronger)."
        ),
    )
    args = parser.parse_args()

    if not args.embeddings.is_file():
        raise FileNotFoundError(f"Embeddings not found: {args.embeddings}")
    if not args.manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    evaluate_all(
        embeddings_path=args.embeddings,
        manifest_path=args.manifest,
        output_path=args.output,
        folds=args.folds,
        repeats=args.repeats,
        seed=args.seed,
        acoustic=args.acoustic_features,
        only=tuple(args.only) if args.only else None,
        group_by=args.group_by,
    )


if __name__ == "__main__":
    main()