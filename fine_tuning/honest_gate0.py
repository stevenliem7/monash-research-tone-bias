"""Honest Gate 0: frozen representation comparison with no test-fold selection.

Arms compared on identical question-grouped outer folds:

    final layer   + MLP
    uniform mix   + MLP
    learned mix   + MLP
    final layer   + fixed RBF-SVM (C=1, gamma=scale)   — historical incumbent
    final layer   + inner-tuned RBF-SVM
    uniform mean  + fixed RBF-SVM
    uniform mean  + inner-tuned RBF-SVM

Selection (epoch, scalar LR, SVM grid) uses only the outer training partition.
Outer score labels are predicted once after a step-matched refit.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from fine_tuning.metrics import classification_report_dict, scalar_weight_diagnostics
from fine_tuning.model import FrozenPooledClassifier, seed_everything
from fine_tuning.paths import LAST_LAYER_IDX, PROTOCOL_VERSION, SCALAR_LRS
from fine_tuning.splits import (
    FoldSpec,
    LabelAccessGuard,
    epochs_for_matched_steps,
    inner_folds,
    leaky_outer_folds,
    outer_folds,
)


@dataclass
class MlpConfig:
    head_lr: float = 1e-4
    scalar_lr: float = 1e-3
    weight_decay: float = 0.01
    batch_size: int = 16
    max_epochs: int = 100
    warmup_steps: int = 50


def _class_weights(y: np.ndarray, n_classes: int = 3) -> np.ndarray:
    counts = np.maximum(np.bincount(y, minlength=n_classes).astype(np.float64), 1.0)
    return (counts.sum() / (n_classes * counts)).astype(np.float32)


def _build_optimizer(model: FrozenPooledClassifier, cfg: MlpConfig) -> torch.optim.AdamW:
    groups = [
        {
            "params": list(model.head.parameters()),
            "lr": cfg.head_lr,
            "weight_decay": cfg.weight_decay,
        }
    ]
    if model.learn_mix:
        groups.append(
            {
                "params": list(model.mix.parameters()),
                "lr": cfg.scalar_lr,
                "weight_decay": 0.0,
            }
        )
    return torch.optim.AdamW(groups)


def run_mlp_epochs(
    model: FrozenPooledClassifier,
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    *,
    epochs: int,
    cfg: MlpConfig,
    seed: int,
    X_eval: np.ndarray | None = None,
    y_eval: np.ndarray | None = None,
) -> list[float]:
    """Fixed-length training. Returns per-epoch eval macro-F1 when eval is set."""
    seed_everything(seed)
    model.train()
    opt = _build_optimizer(model, cfg)
    weights = torch.tensor(_class_weights(y_fit), dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights)
    xt = torch.from_numpy(np.ascontiguousarray(X_fit)).float()
    yt = torch.from_numpy(y_fit.astype(np.int64))
    rng = np.random.default_rng(seed)
    n = len(y_fit)
    base_lrs = [g["lr"] for g in opt.param_groups]
    step = 0
    curve: list[float] = []

    for _ in range(epochs):
        model.train()
        order = rng.permutation(n)
        for start in range(0, n, cfg.batch_size):
            idx = order[start : start + cfg.batch_size]
            scale = min(1.0, (step + 1) / max(1, cfg.warmup_steps))
            for g, base in zip(opt.param_groups, base_lrs):
                g["lr"] = base * scale
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(xt[idx]), yt[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
        if X_eval is not None and y_eval is not None:
            curve.append(macro_f1(model, X_eval, y_eval))
    return curve


@torch.no_grad()
def predict_mlp(model: FrozenPooledClassifier, X: np.ndarray) -> np.ndarray:
    model.eval()
    logits = model(torch.from_numpy(np.ascontiguousarray(X)).float())
    return logits.argmax(dim=1).numpy()


def macro_f1(model: FrozenPooledClassifier, X: np.ndarray, y: np.ndarray) -> float:
    return float(f1_score(y, predict_mlp(model, X), average="macro", zero_division=0))


FEATURES = {
    "final": lambda X: X[:, LAST_LAYER_IDX : LAST_LAYER_IDX + 1, :],
    "uniform": lambda X: X,  # MLP mean-pools over layers
    # Mean vector for SVM. Do not reuse "uniform": reshape would flatten 25*H.
    "uniform_vec": lambda X: X.mean(axis=1, keepdims=True),
    "stack": lambda X: X,
}

# Matches train_valence_head_cv.py candidate "rbf_svc" (StandardScaler + SVC).
# Source macro-F1 0.589: data/.../valence_head_cv_wavlm_group_question.json,
# regime=balanced, group=question, mean over 10 repeats = 0.5886.
FIXED_SVC = {"C": 1.0, "gamma": "scale"}
FIXED_SVC_SOURCE = {
    "code": (
        "monash-research-tone-bias/src/our_speech_corpus_final/heads/"
        "train_valence_head_cv.py :: candidates['rbf_svc']"
    ),
    "params": {"kernel": "rbf", "C": 1.0, "gamma": "scale", "class_weight": "balanced"},
    "preprocessing": "StandardScaler",
    "report": (
        "data/our_speech_corpus_wavlm/reports/"
        "valence_head_cv_wavlm_group_question.json"
    ),
    "regime": "balanced",
    "group_by": "question",
    "macro_f1_mean": 0.5885680206176349,
    "macro_f1_sd": 0.013151856253506409,
    "n_repeats": 10,
    "headline_rounded": 0.589,
}
SVC_GRID = [
    {"C": c, "gamma": g}
    for c in (1.0, 3.0, 10.0, 30.0)
    for g in ("scale", 1e-3, 1e-4)
]


def _svc(params: dict) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "svc",
                SVC(
                    kernel="rbf",
                    C=params["C"],
                    gamma=params["gamma"],
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def run_svc_fixed(
    X: np.ndarray,
    y: np.ndarray,
    outer: FoldSpec,
    guard: LabelAccessGuard,
) -> dict:
    """Historical incumbent hyperparameters: C=1, gamma=scale. No search."""
    if X.ndim != 2:
        raise ValueError(f"SVM expects 2-D features, got shape {X.shape}")
    if X.shape[1] > 2048:
        raise ValueError(
            f"SVM feature dim {X.shape[1]} looks like a flattened layer stack; "
            "use FEATURES['uniform_vec'] (mean) or FEATURES['final'], not reshape of "
            "the full (N, L, H) tensor."
        )
    model = _svc(FIXED_SVC).fit(X[outer.fit], guard.labels(y, outer.fit, "fit"))
    return {
        "pred": model.predict(X[outer.score]),
        "selected": dict(FIXED_SVC),
        "inner_mean_macro_f1": None,
    }


def run_svc_tuned(
    X: np.ndarray,
    y: np.ndarray,
    outer: FoldSpec,
    inners: list[FoldSpec],
    guard: LabelAccessGuard,
) -> dict:
    """Grid C/gamma on inner folds only, then refit on outer fit."""
    if X.ndim != 2:
        raise ValueError(f"SVM expects 2-D features, got shape {X.shape}")
    if X.shape[1] > 2048:
        raise ValueError(
            f"SVM feature dim {X.shape[1]} looks like a flattened layer stack; "
            "use FEATURES['uniform_vec'] (mean) or FEATURES['final']."
        )
    scores: dict[int, list[float]] = {i: [] for i in range(len(SVC_GRID))}
    for inner in inners:
        y_fit = guard.labels(y, inner.fit, "fit")
        y_val = guard.labels(y, inner.score, "select")
        for i, params in enumerate(SVC_GRID):
            model = _svc(params).fit(X[inner.fit], y_fit)
            pred = model.predict(X[inner.score])
            scores[i].append(float(f1_score(y_val, pred, average="macro", zero_division=0)))
    means = {i: float(np.mean(v)) for i, v in scores.items()}
    best_i = max(means, key=means.get)
    best = SVC_GRID[best_i]
    model = _svc(best).fit(X[outer.fit], guard.labels(y, outer.fit, "fit"))
    return {
        "pred": model.predict(X[outer.score]),
        "selected": {"C": best["C"], "gamma": best["gamma"]},
        "inner_mean_macro_f1": means[best_i],
    }


def run_mlp_arm(
    X: np.ndarray,
    y: np.ndarray,
    outer: FoldSpec,
    inners: list[FoldSpec],
    guard: LabelAccessGuard,
    *,
    learn_mix: bool,
    cfg: MlpConfig,
    seed: int,
) -> dict:
    """Pick epoch (and mix LR) on inner folds, then step-matched refit once."""
    n_layers = X.shape[1] if learn_mix else X.shape[1]
    hidden = int(X.shape[-1])
    lr_grid = tuple(SCALAR_LRS) if learn_mix else (cfg.scalar_lr,)

    per_lr: dict[float, dict] = {}
    for scalar_lr in lr_grid:
        run_cfg = MlpConfig(**{**cfg.__dict__, "scalar_lr": scalar_lr})
        best_epochs: list[int] = []
        best_scores: list[float] = []
        for inner in inners:
            y_fit = guard.labels(y, inner.fit, "fit")
            y_val = guard.labels(y, inner.score, "select")
            seed_everything(seed)
            model = FrozenPooledClassifier(
                n_layers=n_layers, hidden_size=hidden, learn_mix=learn_mix
            )
            if not learn_mix:
                # Uniform / final: mix averages layers; for final, X already 1 layer.
                model.mix.logits.requires_grad_(False)
            curve = run_mlp_epochs(
                model,
                X[inner.fit],
                y_fit,
                epochs=run_cfg.max_epochs,
                cfg=run_cfg,
                seed=seed,
                X_eval=X[inner.score],
                y_eval=y_val,
            )
            best_epochs.append(int(np.argmax(curve)) + 1)
            best_scores.append(float(np.max(curve)))
        per_lr[scalar_lr] = {
            "mean_macro_f1": float(np.mean(best_scores)),
            "median_epoch": float(np.median(best_epochs)),
            "epochs": best_epochs,
        }

    best_lr = max(per_lr, key=lambda k: per_lr[k]["mean_macro_f1"])
    chosen = per_lr[best_lr]
    n_inner_fit = int(np.mean([len(i.fit) for i in inners]))
    refit_epochs = epochs_for_matched_steps(
        inner_epochs=chosen["median_epoch"],
        n_inner_fit=n_inner_fit,
        n_outer_fit=len(outer.fit),
        batch_size=cfg.batch_size,
    )
    run_cfg = MlpConfig(**{**cfg.__dict__, "scalar_lr": best_lr})
    seed_everything(seed)
    model = FrozenPooledClassifier(
        n_layers=n_layers, hidden_size=hidden, learn_mix=learn_mix
    )
    if not learn_mix:
        model.mix.logits.requires_grad_(False)
    run_mlp_epochs(
        model,
        X[outer.fit],
        guard.labels(y, outer.fit, "fit"),
        epochs=refit_epochs,
        cfg=run_cfg,
        seed=seed,
    )
    out: dict = {
        "pred": predict_mlp(model, X[outer.score]),
        "selected": {
            "scalar_lr": best_lr if learn_mix else None,
            "inner_median_epoch": chosen["median_epoch"],
            "refit_epochs": refit_epochs,
        },
        "inner_mean_macro_f1": chosen["mean_macro_f1"],
    }
    if learn_mix:
        out["scalar_diagnostics"] = scalar_weight_diagnostics(
            model.mix.logits.detach().numpy()
        )
    return out


# name, feature kind, head, learn_mix
ARMS = (
    ("final_mlp", "final", "mlp", False),
    ("uniform_mlp", "uniform", "mlp", False),
    ("mix_mlp", "stack", "mlp", True),
    ("final_svc_fixed", "final", "svc_fixed", False),
    ("final_svc_tuned", "final", "svc_tuned", False),
    ("uniform_svc_fixed", "uniform_vec", "svc_fixed", False),
    ("uniform_svc_tuned", "uniform_vec", "svc_tuned", False),
)


def run_protocol(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    fold_iter,
    n_inner: int,
    cfg: MlpConfig,
    seed: int,
    tag: str,
    verbose: bool = True,
) -> dict:
    """Run every arm over one family of folds."""
    n = len(y)
    per_repeat: dict[int, dict[str, np.ndarray]] = {}
    fold_rows: list[dict] = []
    audits: list[dict] = []
    feats = {kind: fn(X) for kind, fn in FEATURES.items()}

    for outer in fold_iter:
        rep = per_repeat.setdefault(
            outer.repeat, {name: -np.ones(n, dtype=np.int64) for name, *_ in ARMS}
        )
        if outer.can_select:
            inners = [outer]
        else:
            inners = inner_folds(outer, y, groups, n_splits=n_inner, seed=seed)

        row: dict = {**outer.describe(), "arms": {}}
        for name, kind, head, learn_mix in ARMS:
            guard = LabelAccessGuard(outer)
            xa = feats[kind]
            # SVM arms need 2-D features.
            if head.startswith("svc"):
                xa_flat = xa.reshape(xa.shape[0], -1)
            t0 = time.time()
            if head == "svc_fixed":
                res = run_svc_fixed(xa_flat, y, outer, guard)
            elif head == "svc_tuned":
                res = run_svc_tuned(xa_flat, y, outer, inners, guard)
            else:
                res = run_mlp_arm(
                    xa,
                    y,
                    outer,
                    inners,
                    guard,
                    learn_mix=learn_mix,
                    cfg=cfg,
                    seed=seed,
                )
            if not outer.can_select:
                guard.assert_clean()
            rep[name][outer.score] = res["pred"]
            fold_macro = float(
                f1_score(y[outer.score], res["pred"], average="macro", zero_division=0)
            )
            row["arms"][name] = {
                "macro_f1": fold_macro,
                "selected": res["selected"],
                "inner_mean_macro_f1": res.get("inner_mean_macro_f1"),
                "seconds": round(time.time() - t0, 1),
            }
            if "scalar_diagnostics" in res:
                row["arms"][name]["scalar_diagnostics"] = res["scalar_diagnostics"]
            audits.append({"arm": name, **guard.summary()})
        fold_rows.append(row)
        if verbose:
            print(
                f"[{tag}] {outer.label} "
                + " ".join(f"{k}={v['macro_f1']:.4f}" for k, v in row["arms"].items())
            )

    pooled: dict[str, dict] = {}
    for name, *_ in ARMS:
        by_repeat = [
            classification_report_dict(y, rep[name]) for rep in per_repeat.values()
        ]
        macros = [r["macro_f1"] for r in by_repeat]
        pooled[name] = {
            "macro_f1_mean_over_repeats": float(np.mean(macros)),
            "macro_f1_sd_over_repeats": (
                float(np.std(macros, ddof=1)) if len(macros) > 1 else 0.0
            ),
            "n_repeats": len(macros),
            "repeat_0": by_repeat[0],
        }

    return {
        "tag": tag,
        "folds": fold_rows,
        "pooled": pooled,
        "audits": audits[: len(ARMS)],
        "oof_by_repeat": {
            name: np.stack([rep[name] for rep in per_repeat.values()]).tolist()
            for name, *_ in ARMS
        },
    }


def four_cell_table(pooled: dict) -> str:
    def cell(name: str) -> str:
        p = pooled.get(name)
        if p is None:
            return "n/a"
        return f"{p['macro_f1_mean_over_repeats']:.4f}"

    return "\n".join(
        [
            "| representation | MLP | RBF-SVM fixed | RBF-SVM tuned |",
            "|---|---|---|---|",
            f"| final layer | {cell('final_mlp')} | {cell('final_svc_fixed')} | {cell('final_svc_tuned')} |",
            f"| uniform mix | {cell('uniform_mlp')} | {cell('uniform_svc_fixed')} | {cell('uniform_svc_tuned')} |",
            f"| learned mix | {cell('mix_mlp')} | n/a | n/a |",
        ]
    )


def build_honest_gate0_report(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    n_outer: int = 5,
    n_inner: int = 3,
    repeats: int = 1,
    seed: int = 42,
    max_epochs: int = 100,
    include_leaky: bool = False,
    verbose: bool = True,
) -> dict:
    from datetime import datetime, timezone

    cfg = MlpConfig(max_epochs=max_epochs)
    honest = run_protocol(
        X,
        y,
        groups,
        fold_iter=outer_folds(
            y, groups, n_splits=n_outer, n_repeats=repeats, seed=seed
        ),
        n_inner=n_inner,
        cfg=cfg,
        seed=seed,
        tag="honest",
        verbose=verbose,
    )

    prefer_mix = (
        honest["pooled"]["mix_mlp"]["macro_f1_mean_over_repeats"]
        > honest["pooled"]["final_mlp"]["macro_f1_mean_over_repeats"]
    )
    report: dict = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_version": PROTOCOL_VERSION,
        "protocol": {
            "outer": f"StratifiedGroupKFold({n_outer}) x {repeats} repeats",
            "inner": f"StratifiedGroupKFold({n_inner}) inside training partition",
            "selection": (
                "epoch count, scalar LR, and tuned-SVM grid chosen on inner folds only; "
                "outer refit uses matched optimizer steps"
            ),
            "decision_rule": "argmax for every arm",
            "arms": [a[0] for a in ARMS],
        },
        "honest": honest,
        "four_cell_table": four_cell_table(honest["pooled"]),
        "oof": {
            "final": honest["pooled"]["final_mlp"]["repeat_0"],
            "uniform": honest["pooled"]["uniform_mlp"]["repeat_0"],
            "learned": honest["pooled"]["mix_mlp"]["repeat_0"],
            "final_svc_fixed": honest["pooled"]["final_svc_fixed"]["repeat_0"],
            "final_svc_tuned": honest["pooled"]["final_svc_tuned"]["repeat_0"],
            "uniform_svc_fixed": honest["pooled"]["uniform_svc_fixed"]["repeat_0"],
            "uniform_svc_tuned": honest["pooled"]["uniform_svc_tuned"]["repeat_0"],
        },
        "prefer_frozen_mix": prefer_mix,
        "selected_scalar_lrs": [
            fold["arms"]["mix_mlp"]["selected"].get("scalar_lr")
            for fold in honest["folds"]
            if fold["arms"].get("mix_mlp", {}).get("selected")
        ],
        "fixed_svc_source": FIXED_SVC_SOURCE,
        "contextual": {
            "historical_rbf_svc": FIXED_SVC_SOURCE["headline_rounded"],
            "historical_rbf_svc_exact": FIXED_SVC_SOURCE["macro_f1_mean"],
            "historical_rbf_svc_source": FIXED_SVC_SOURCE["report"],
            "note": (
                "0.589 is question-grouped balanced rbf_svc on final-layer WavLM "
                "embeddings (C=1, gamma=scale). It is contextual SoTA, not a matched "
                "control for the neural Gate 0 protocol. Compare Gate 0 SVM arms to "
                "each other and to MLPs under the same folds."
            ),
            "leaked_gate0_final_mlp": 0.6417,
            "leaked_gate0_learned_mix": 0.6371,
        },
    }

    if include_leaky:
        leaky = run_protocol(
            X,
            y,
            groups,
            fold_iter=leaky_outer_folds(
                y, groups, n_splits=n_outer, n_repeats=repeats, seed=seed
            ),
            n_inner=n_inner,
            cfg=cfg,
            seed=seed,
            tag="LEAKY",
            verbose=verbose,
        )
        report["leaky_calibration"] = leaky
        report["leaked_minus_honest"] = {
            name: round(
                leaky["pooled"][name]["macro_f1_mean_over_repeats"]
                - honest["pooled"][name]["macro_f1_mean_over_repeats"],
                4,
            )
            for name, *_ in ARMS
        }
    return report
