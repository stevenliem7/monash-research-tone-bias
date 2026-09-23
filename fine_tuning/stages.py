"""Experiment stages: Gate 0, nested grouped CV, leakage check, final fit."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

from fine_tuning.cache import (
    build_pooled_cache,
    ensure_boundary_cache,
    extract_pooled_all_layers,
    load_pooled_cache,
)
from fine_tuning.data import (
    class_weights_from_y,
    load_labelled_frame,
    make_random_folds,
    manifest_fingerprint,
)
from fine_tuning.honest_gate0 import build_honest_gate0_report
from fine_tuning.metrics import (
    classification_report_dict,
    earliest_within_one_se,
    evidence_level,
    mcnemar_secondary,
    paired_group_bootstrap,
    paired_group_randomization,
)
from fine_tuning.model import (
    FreezePlan,
    FrozenPooledClassifier,
    SuffixMixClassifier,
    configure_wavlm_for_partial,
    load_wavlm,
    seed_everything,
)
from fine_tuning.optimise import (
    TrainConfig,
    eval_suffix_parity,
    predict_frozen_pooled,
    predict_partial_wavlm,
    train_eval_partial_honest,
    train_eval_pooled_honest,
    train_frozen_pooled_fixed,
    train_partial_fixed,
    BoundaryBatcher,
)
from fine_tuning.paths import (
    ACTIVE_DEPTHS,
    CACHE_FORMAT_VERSION,
    CORPUS_FINAL_AUDIO_DIR,
    ENCODER_LRS,
    OUTER_SEEDS,
    PRIMARY_SEED,
    PROBE_LAYERS,
    PROTOCOL_VERSION,
    SCALAR_LRS,
    SPLIT_MANIFEST_CSV,
    WAVLM_FINETUNED_ROOT,
    WAVLM_MODEL_ID,
)
from fine_tuning.splits import FoldSpec, inner_folds, outer_folds


@dataclass
class ExperimentConfig:
    """Top-level run configuration."""

    manifest_path: Path = SPLIT_MANIFEST_CSV
    corpora_root: Path = CORPUS_FINAL_AUDIO_DIR
    output_dir: Path = WAVLM_FINETUNED_ROOT
    model_id: str = WAVLM_MODEL_ID
    device: str | None = None
    limit: int | None = None
    stage: str = "all"
    outer_folds: int = 5
    inner_folds: int = 3
    seed: int = PRIMARY_SEED
    repeats: int = 1
    include_leaky_calibration: bool = False
    ignore_gate0: bool = False
    use_partial_ft: bool = True
    gate0_max_epochs: int = 100
    nested_inner_max_epochs: int = 15
    pooled_provider: Callable[..., Any] | None = None
    wav_loader: Callable[[Path], np.ndarray] | None = None


def _device(cfg: ExperimentConfig) -> str:
    if cfg.device:
        return cfg.device
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _dirs(cfg: ExperimentConfig) -> dict[str, Path]:
    root = cfg.output_dir
    paths = {
        "root": root,
        "cache": root / "cache",
        "checkpoints": root / "checkpoints",
        "reports": root / "reports",
        "run_state": root / "run_state.json",
    }
    for key in ("cache", "checkpoints", "reports"):
        paths[key].mkdir(parents=True, exist_ok=True)
    return paths


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _update_run_state(path: Path, **kwargs: Any) -> None:
    state = {}
    if path.is_file():
        state = json.loads(path.read_text(encoding="utf-8"))
    state.update(kwargs)
    state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(path, state)


def _default_wav_loader(path: Path) -> np.ndarray:
    from fine_tuning.data import load_wav_mono_16k

    return load_wav_mono_16k(path)


def load_frame(cfg: ExperimentConfig) -> pd.DataFrame:
    frame = load_labelled_frame(cfg.manifest_path)
    if cfg.limit is not None:
        frame = frame.head(cfg.limit).copy().reset_index(drop=True)
    return frame


def _cache_expected(cfg: ExperimentConfig, fingerprint: str) -> dict:
    from fine_tuning.cache import load_meta

    meta = load_meta(cfg.output_dir / "cache")
    hidden_size = int(meta["hidden_size"]) if meta and "hidden_size" in meta else 1024
    n_layers = (
        int(meta["n_transformer_layers"])
        if meta and "n_transformer_layers" in meta
        else 24
    )
    return {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "model_id": cfg.model_id,
        "manifest_fingerprint": fingerprint,
        "hidden_size": hidden_size,
        "n_transformer_layers": n_layers,
    }


def _align_pooled(frame: pd.DataFrame, bundle: dict) -> np.ndarray:
    lookup = {str(c): i for i, c in enumerate(bundle["clip_id"])}
    idxs = [lookup[c] for c in frame["clip_id"].tolist()]
    return bundle["pooled"].numpy()[idxs]


def _freeze_plan(endpoint: int, depth: int) -> FreezePlan:
    start = max(0, endpoint - depth)
    return FreezePlan(
        active_start=start,
        active_end=endpoint,
        retained_endpoint=endpoint,
    )


def _probe_window(X: np.ndarray, endpoint: int, depth: int) -> np.ndarray:
    """Stack matching the active mix window ending at ``endpoint`` (1-indexed).

    Uses cached layers ``[endpoint-depth+1 .. endpoint]`` so the probe sees the
    same truncated representation the candidate arm would mix (live after FT).
    """
    start = max(0, endpoint - depth)
    return X[:, start + 1 : endpoint + 1, :]


def stage_layer_cache(cfg: ExperimentConfig) -> dict:
    """Build pooled_all_layers cache for all labelled clips."""
    paths = _dirs(cfg)
    frame = load_frame(cfg)
    device = _device(cfg)
    fp = manifest_fingerprint(cfg.manifest_path)
    expected = {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "model_id": cfg.model_id,
        "manifest_fingerprint": fp,
        "hidden_size": 1024,
        "n_transformer_layers": 24,
    }
    try:
        bundle = load_pooled_cache(paths["cache"], expected)
        return {"status": "cache_hit", "n": len(bundle["clip_id"])}
    except (FileNotFoundError, ValueError):
        pass

    if cfg.pooled_provider is not None:
        pooled, lengths, clip_ids = cfg.pooled_provider(frame)
    else:
        processor, model = load_wavlm(cfg.model_id, device)
        model.eval()
        wav_loader = cfg.wav_loader or _default_wav_loader
        pooled_list = []
        lengths = []
        clip_ids = []
        t0 = time.time()
        for _, row in frame.iterrows():
            wav_path = Path(str(row["audio_path"]))
            wav = wav_loader(wav_path)
            pooled, fl = extract_pooled_all_layers(processor, model, [wav], device)
            pooled_list.append(pooled)
            lengths.append(fl[0])
            clip_ids.append(str(row["clip_id"]))
            if (len(clip_ids) % 25) == 0:
                print(f"[layer-cache] {len(clip_ids)}/{len(frame)}")
        pooled = np.stack(pooled_list, axis=0)
        lengths = np.asarray(lengths, dtype=np.int32)
        print(f"[layer-cache] done in {time.time() - t0:.1f}s")

    build_pooled_cache(
        clip_ids=clip_ids,
        pooled=pooled,
        frame_lengths=np.asarray(lengths),
        cache_dir=paths["cache"],
        manifest_fingerprint=fp,
        model_id=cfg.model_id,
    )
    _update_run_state(paths["run_state"], stage="layer-cache", status="done")
    return {"status": "built", "n": len(clip_ids)}


def _partial_training_context(cfg: ExperimentConfig) -> dict:
    """Load WavLM once and keep a resettable pretrained state dict."""
    device = _device(cfg)
    processor, wavlm = load_wavlm(cfg.model_id, device)
    base_state = {k: v.detach().cpu().clone() for k, v in wavlm.state_dict().items()}
    frame = load_frame(cfg)
    clip_ids = frame["clip_id"].astype(str).tolist()
    clip_id_to_row = {c: i for i, c in enumerate(clip_ids)}
    wav_loader = cfg.wav_loader or _default_wav_loader
    paths = _dirs(cfg)
    boundary_cache: dict[int, dict] = {}
    fp = manifest_fingerprint(cfg.manifest_path)
    pooled_bundle = load_pooled_cache(paths["cache"], _cache_expected(cfg, fp))
    raw_lengths = pooled_bundle.get("frame_lengths")
    lookup = {str(c): i for i, c in enumerate(pooled_bundle["clip_id"])}
    if raw_lengths is None:
        frame_lengths = np.ones(len(clip_ids), dtype=np.int64)
    else:
        if hasattr(raw_lengths, "numpy"):
            raw_lengths = raw_lengths.numpy()
        frame_lengths = np.asarray(
            [int(raw_lengths[lookup[c]]) for c in clip_ids], dtype=np.int64
        )

    def boundary_bundle(active_start: int) -> dict:
        if active_start not in boundary_cache:
            boundary_cache[active_start] = ensure_boundary_cache(
                cache_dir=paths["cache"],
                active_start=active_start,
                frame=frame,
                processor=processor,
                model=wavlm,
                device=device,
                wav_loader=wav_loader,
            )
        return boundary_cache[active_start]

    def reset_wavlm() -> None:
        wavlm.load_state_dict(base_state)
        configure_wavlm_for_partial(wavlm)

    return {
        "device": device,
        "wavlm": wavlm,
        "processor": processor,
        "reset_wavlm": reset_wavlm,
        "clip_ids": clip_ids,
        "clip_id_to_row": clip_id_to_row,
        "boundary_bundle": boundary_bundle,
        "frame_lengths": frame_lengths,
    }


def stage_frozen_gate(cfg: ExperimentConfig) -> dict:
    """Honest Gate 0: matched MLP + fixed/tuned SVM arms."""
    paths = _dirs(cfg)
    frame = load_frame(cfg)
    fp = manifest_fingerprint(cfg.manifest_path)
    bundle = load_pooled_cache(paths["cache"], _cache_expected(cfg, fp))
    X_all = _align_pooled(frame, bundle)
    y = frame["y"].to_numpy()
    groups = frame["question"].to_numpy()

    report = build_honest_gate0_report(
        X_all,
        y,
        groups,
        n_outer=cfg.outer_folds,
        n_inner=cfg.inner_folds,
        repeats=cfg.repeats,
        seed=cfg.seed,
        max_epochs=cfg.gate0_max_epochs,
        include_leaky=cfg.include_leaky_calibration,
        verbose=True,
    )
    out = paths["reports"] / "gate0_frozen.json"
    _write_json(out, report)
    _update_run_state(
        paths["run_state"],
        stage="frozen-gate",
        status="done",
        prefer_frozen_mix=report["prefer_frozen_mix"],
        protocol_version=PROTOCOL_VERSION,
    )
    return report


def _probe_endpoint_honest(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    outer: FoldSpec,
    *,
    seed: int,
    device: str,
    inner_n: int,
    probe_depth: int,
    max_epochs: int = 40,
) -> int:
    """Choose earliest endpoint within 1 SE using the same mix window as arms."""
    inners = inner_folds(outer, y, groups, n_splits=inner_n, seed=seed)
    scores: dict[int, list[float]] = {L: [] for L in PROBE_LAYERS}
    for L in PROBE_LAYERS:
        depth = min(probe_depth, L)
        Xs = _probe_window(X, L, depth)
        for inner in inners:
            seed_everything(seed)
            model = FrozenPooledClassifier(
                n_layers=Xs.shape[1],
                hidden_size=int(Xs.shape[2]),
                learn_mix=True,
            )
            cfg_t = TrainConfig(
                scalar_lr=1e-3,
                max_epochs=max_epochs,
                seed=seed,
                effective_batch=16,
            )
            out = train_frozen_pooled_fixed(
                model,
                Xs[inner.fit],
                y[inner.fit],
                class_weights_from_y(y[inner.fit]),
                cfg=cfg_t,
                device=device,
                X_eval=Xs[inner.score],
                y_eval=y[inner.score],
            )
            curve = out["curve"]
            scores[L].append(float(np.max(curve)) if curve else 0.0)
    for L in PROBE_LAYERS:
        if not scores[L]:
            scores[L] = [0.0]
    return earliest_within_one_se(scores)


def stage_nested_cv(cfg: ExperimentConfig) -> dict:
    """Nested grouped CV with leak-proof selection and step-matched refits."""
    paths = _dirs(cfg)
    gate_path = paths["reports"] / "gate0_frozen.json"
    if gate_path.is_file():
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if gate.get("prefer_frozen_mix") and not cfg.ignore_gate0:
            report = {
                "skipped": True,
                "reason": "prefer_frozen_mix",
                "prefer_frozen_mix": True,
                "protocol_version": PROTOCOL_VERSION,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            _write_json(paths["reports"] / "nested_cv.json", report)
            _update_run_state(
                paths["run_state"], stage="nested-cv", status="skipped_gate0"
            )
            return report

    frame = load_frame(cfg)
    device = _device(cfg)
    fp = manifest_fingerprint(cfg.manifest_path)
    bundle = load_pooled_cache(paths["cache"], _cache_expected(cfg, fp))
    X_all = _align_pooled(frame, bundle)
    y = frame["y"].to_numpy()
    groups = frame["question"].to_numpy()
    partial_ctx = _partial_training_context(cfg) if cfg.use_partial_ft else None
    probe_depth = max(ACTIVE_DEPTHS)

    oof_base = np.full(len(y), -1, dtype=np.int64)
    oof_ft = np.full(len(y), -1, dtype=np.int64)
    seed_oof: dict[int, np.ndarray] = {
        s: np.full(len(y), -1, dtype=np.int64) for s in OUTER_SEEDS
    }
    outer_rows = []

    for outer in outer_folds(
        y, groups, n_splits=cfg.outer_folds, n_repeats=1, seed=cfg.seed
    ):
        fold_i = outer.fold
        inners = inner_folds(
            outer, y, groups, n_splits=cfg.inner_folds, seed=cfg.seed
        )
        endpoint = _probe_endpoint_honest(
            X_all,
            y,
            groups,
            outer,
            seed=cfg.seed,
            device=device,
            inner_n=cfg.inner_folds,
            probe_depth=probe_depth,
            max_epochs=min(40, cfg.nested_inner_max_epochs * 2),
        )

        # Scalar LR on truncated probe window (frozen mix), inner only.
        Xs_endpoint = _probe_window(X_all, endpoint, probe_depth)
        scalar_lr = _select_scalar_lr(
            Xs_endpoint, y, outer, inners, seed=cfg.seed, device=device
        )

        combo_scores: dict[tuple[int, float], list[float]] = {}
        combo_epochs: dict[tuple[int, float], list[int]] = {}
        for depth in ACTIVE_DEPTHS:
            for enc_lr in ENCODER_LRS:
                key = (depth, enc_lr)
                combo_scores[key] = []
                combo_epochs[key] = []
                start = max(0, endpoint - depth)
                if cfg.use_partial_ft:
                    assert partial_ctx is not None
                    partial_ctx["boundary_bundle"](start)
                    # Selection only: evaluate each inner fold as score set.
                    for inner in inners:
                        if cfg.use_partial_ft:
                            # Train on inner.fit, score on inner.score with fixed
                            # max epochs curve (epoch picked later via median).
                            from fine_tuning.optimise import train_partial_fixed

                            partial_ctx["reset_wavlm"]()
                            seed_everything(cfg.seed)
                            model = SuffixMixClassifier(
                                partial_ctx["wavlm"],
                                active_start=start,
                                active_end=endpoint,
                                n_frozen_lower=0,
                            )
                            batcher = BoundaryBatcher(
                                boundary_bundle=partial_ctx["boundary_bundle"](start),
                                clip_id_to_row=partial_ctx["clip_id_to_row"],
                                clip_ids=partial_ctx["clip_ids"],
                                frame_lengths=partial_ctx["frame_lengths"],
                            )
                            y_fit = y[inner.fit]
                            cfg_t = TrainConfig(
                                scalar_lr=scalar_lr,
                                encoder_lr=enc_lr,
                                max_epochs=cfg.nested_inner_max_epochs,
                                seed=cfg.seed,
                                effective_batch=8,
                            )
                            out = train_partial_fixed(
                                model,
                                row_indices=inner.fit,
                                y=y_fit,
                                class_weights=class_weights_from_y(y_fit),
                                batcher=batcher,
                                cfg=cfg_t,
                                device=partial_ctx["device"],
                                eval_rows=inner.score,
                                y_eval=y[inner.score],
                                progress_label=(
                                    f"nested-cv fold{fold_i} d{depth} lr{enc_lr:g}"
                                ),
                            )
                            curve = out["curve"]
                            combo_scores[key].append(
                                float(np.max(curve)) if curve else -1.0
                            )
                            combo_epochs[key].append(
                                int(np.argmax(curve)) + 1
                                if curve
                                else cfg.nested_inner_max_epochs
                            )
                else:
                    Xs = _probe_window(X_all, endpoint, depth)
                    for inner in inners:
                        seed_everything(cfg.seed)
                        model = FrozenPooledClassifier(
                            n_layers=Xs.shape[1],
                            hidden_size=int(Xs.shape[2]),
                            learn_mix=True,
                        )
                        cfg_t = TrainConfig(
                            scalar_lr=scalar_lr,
                            max_epochs=cfg.nested_inner_max_epochs,
                            seed=cfg.seed,
                            effective_batch=16,
                        )
                        out = train_frozen_pooled_fixed(
                            model,
                            Xs[inner.fit],
                            y[inner.fit],
                            class_weights_from_y(y[inner.fit]),
                            cfg=cfg_t,
                            device=device,
                            X_eval=Xs[inner.score],
                            y_eval=y[inner.score],
                        )
                        curve = out["curve"]
                        combo_scores[key].append(
                            float(np.max(curve)) if curve else -1.0
                        )
                        combo_epochs[key].append(
                            int(np.argmax(curve)) + 1
                            if curve
                            else cfg.nested_inner_max_epochs
                        )

        def _rank(key: tuple[int, float]) -> tuple[float, float, float]:
            return (
                float(np.mean(combo_scores[key])),
                float(np.median(combo_epochs[key])),
                -key[0],
            )

        best_key = max(combo_scores, key=_rank)
        depth, enc_lr = best_key
        median_epoch = float(np.median(combo_epochs[best_key]))

        # Matched frozen control: honest inner selection + refit on outer.fit.
        Xs_arm = _probe_window(X_all, endpoint, depth)
        base = train_eval_pooled_honest(
            X=Xs_arm,
            y=y,
            outer=outer,
            inners=inners,
            learn_mix=True,
            scalar_lr=scalar_lr,
            seed=cfg.seed,
            device=device,
            max_epochs=100,
            lr_grid=(scalar_lr,),
        )
        oof_base[outer.score] = base["pred"]

        for seed in OUTER_SEEDS:
            fold_seed = seed + fold_i * 1000
            if cfg.use_partial_ft:
                assert partial_ctx is not None
                ft = train_eval_partial_honest(
                    ctx=partial_ctx,
                    y=y,
                    outer=outer,
                    inners=inners,
                    endpoint=endpoint,
                    depth=depth,
                    encoder_lr=enc_lr,
                    scalar_lr=scalar_lr,
                    seed=fold_seed,
                    max_epochs=cfg.nested_inner_max_epochs,
                    progress_label=f"nested-cv fold{fold_i} ft seed{seed}",
                )
            else:
                ft = train_eval_pooled_honest(
                    X=Xs_arm,
                    y=y,
                    outer=outer,
                    inners=inners,
                    learn_mix=True,
                    scalar_lr=scalar_lr,
                    seed=fold_seed,
                    device=device,
                    max_epochs=cfg.nested_inner_max_epochs,
                    lr_grid=(scalar_lr,),
                )
            seed_oof[seed][outer.score] = ft["pred"]
            if seed == PRIMARY_SEED:
                oof_ft[outer.score] = ft["pred"]
                ft_primary = ft

        outer_rows.append(
            {
                "fold": fold_i,
                "endpoint": endpoint,
                "active_depth": depth,
                "encoder_lr": enc_lr,
                "scalar_lr": scalar_lr,
                "median_best_epoch": median_epoch,
                "refit_epochs": ft_primary.get("refit_epochs"),
                "prediction_shift": ft_primary.get("prediction_shift"),
                "fold_seed": ft_primary.get("fold_seed"),
                "combo_means": {
                    f"d{d}_lr{lr:g}": float(np.mean(combo_scores[(d, lr)]))
                    for d in ACTIVE_DEPTHS
                    for lr in ENCODER_LRS
                },
                "base_macro_f1": base["val_report"]["macro_f1"],
                "ft_macro_f1": classification_report_dict(
                    y[outer.score], oof_ft[outer.score]
                )["macro_f1"],
                "base_selected": base.get("selected"),
                "ft_selected": ft_primary.get("selected"),
            }
        )

    boot = paired_group_bootstrap(y, oof_base, oof_ft, groups, seed=cfg.seed)
    rand = paired_group_randomization(y, oof_base, oof_ft, groups, seed=cfg.seed)
    mcn = mcnemar_secondary(y, oof_base, oof_ft)
    seed_deltas = []
    for seed in OUTER_SEEDS:
        d = (
            classification_report_dict(y, seed_oof[seed])["macro_f1"]
            - classification_report_dict(y, oof_base)["macro_f1"]
        )
        seed_deltas.append(d)
    level = evidence_level(boot, rand, seed_deltas)

    endpoints = [r["endpoint"] for r in outer_rows]
    depths = [r["active_depth"] for r in outer_rows]
    enc_lrs = [r["encoder_lr"] for r in outer_rows]
    epochs = [r["median_best_epoch"] for r in outer_rows]
    locked = {
        "endpoint": int(max(set(endpoints), key=endpoints.count)),
        "active_depth": int(max(set(depths), key=depths.count)),
        "encoder_lr": float(max(set(enc_lrs), key=enc_lrs.count)),
        "median_epoch": float(np.median(epochs)),
        "scalar_lr": float(np.median([r["scalar_lr"] for r in outer_rows])),
    }

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_version": PROTOCOL_VERSION,
        "protocol": {
            "outer": f"StratifiedGroupKFold({cfg.outer_folds})",
            "inner": f"StratifiedGroupKFold({cfg.inner_folds})",
            "group_by": "question",
            "outer_seeds": list(OUTER_SEEDS),
            "primary_seed": PRIMARY_SEED,
            "selection": "inner folds only; outer refit uses matched optimizer steps",
            "estimate_type": (
                "developmental_nested_cv_partial_wavlm"
                if cfg.use_partial_ft
                else "developmental_nested_cv"
            ),
        },
        "outer_folds": outer_rows,
        "locked_config": locked,
        "oof_baseline": classification_report_dict(y, oof_base),
        "oof_finetuned": classification_report_dict(y, oof_ft),
        "paired": {**boot, "randomization": rand, "mcnemar_secondary": mcn},
        "seed_deltas_macro_f1": seed_deltas,
        "evidence_level": level,
        "contextual_frozen_rbf_svc_macro_f1": 0.589,
    }
    _write_json(paths["reports"] / "nested_cv.json", report)
    _write_json(paths["reports"] / "paired_inference.json", report["paired"])

    oof_df = pd.DataFrame(
        {
            "clip_id": frame["clip_id"],
            "question": frame["question"],
            "y_true": y,
            "pred_baseline": oof_base,
            "pred_finetuned": oof_ft,
            **{f"pred_seed_{s}": seed_oof[s] for s in OUTER_SEEDS},
        }
    )
    oof_df.to_csv(paths["reports"] / "oof_predictions.csv", index=False)
    _update_run_state(
        paths["run_state"],
        stage="nested-cv",
        status="done",
        locked=locked,
        protocol_version=PROTOCOL_VERSION,
    )
    return report


def _select_scalar_lr(
    X: np.ndarray,
    y: np.ndarray,
    outer: FoldSpec,
    inners: list[FoldSpec],
    *,
    seed: int,
    device: str,
) -> float:
    """Pick scalar LR from inner-fold means only (no outer score)."""
    scores: dict[float, list[float]] = {lr: [] for lr in SCALAR_LRS}
    for lr in SCALAR_LRS:
        for inner in inners:
            seed_everything(seed)
            model = FrozenPooledClassifier(
                n_layers=X.shape[1], hidden_size=int(X.shape[2]), learn_mix=True
            )
            cfg_t = TrainConfig(
                scalar_lr=lr, max_epochs=40, seed=seed, effective_batch=16
            )
            out = train_frozen_pooled_fixed(
                model,
                X[inner.fit],
                y[inner.fit],
                class_weights_from_y(y[inner.fit]),
                cfg=cfg_t,
                device=device,
                X_eval=X[inner.score],
                y_eval=y[inner.score],
            )
            curve = out["curve"]
            scores[lr].append(float(np.max(curve)) if curve else -1.0)
    return float(max(scores, key=lambda k: float(np.mean(scores[k]))))


def stage_leakage_check(cfg: ExperimentConfig) -> dict:
    """Compare fixed locked config under grouped vs random 5-fold CV."""
    paths = _dirs(cfg)
    nested_path = paths["reports"] / "nested_cv.json"
    if not nested_path.is_file():
        report = {
            "skipped": True,
            "reason": "no nested_cv.json; partial FT not run",
            "protocol_version": PROTOCOL_VERSION,
        }
        _write_json(paths["reports"] / "leakage_check.json", report)
        _update_run_state(
            paths["run_state"], stage="leakage-check", status="skipped_no_nested"
        )
        return report
    nested = json.loads(nested_path.read_text(encoding="utf-8"))
    if nested.get("skipped"):
        report = {
            "skipped": True,
            "reason": nested.get("reason", "nested_cv skipped"),
            "protocol_version": PROTOCOL_VERSION,
        }
        _write_json(paths["reports"] / "leakage_check.json", report)
        _update_run_state(
            paths["run_state"], stage="leakage-check", status="skipped_nested"
        )
        return report
    locked = nested["locked_config"]
    frame = load_frame(cfg)
    device = _device(cfg)
    fp = manifest_fingerprint(cfg.manifest_path)
    bundle = load_pooled_cache(paths["cache"], _cache_expected(cfg, fp))
    X_all = _align_pooled(frame, bundle)
    y = frame["y"].to_numpy()
    groups = frame["question"].to_numpy()
    endpoint = locked["endpoint"]
    depth = locked["active_depth"]
    Xs = _probe_window(X_all, endpoint, depth)
    scalar_lr = locked["scalar_lr"]
    refit_epochs = max(1, int(round(locked["median_epoch"])))

    def _run(fold_specs: list[FoldSpec]) -> float:
        oof = np.full(len(y), -1, dtype=np.int64)
        for outer in fold_specs:
            inners = inner_folds(
                outer, y, groups, n_splits=cfg.inner_folds, seed=cfg.seed
            )
            out = train_eval_pooled_honest(
                X=Xs,
                y=y,
                outer=outer,
                inners=inners,
                learn_mix=True,
                scalar_lr=scalar_lr,
                seed=cfg.seed,
                device=device,
                max_epochs=max(refit_epochs, 15),
                lr_grid=(scalar_lr,),
            )
            oof[outer.score] = out["pred"]
        return classification_report_dict(y, oof)["macro_f1"]

    grouped_specs = list(
        outer_folds(y, groups, n_splits=cfg.outer_folds, seed=cfg.seed)
    )
    # Random folds as FoldSpec list (no groups).
    random_specs = []
    for fold_i, (tr, te) in enumerate(
        make_random_folds(y, cfg.outer_folds, cfg.seed)
    ):
        random_specs.append(FoldSpec(fit=tr, score=te, fold=fold_i, label=f"rand[{fold_i}]"))

    grouped = _run(grouped_specs)
    random = _run(random_specs)
    report = {
        "protocol_version": PROTOCOL_VERSION,
        "locked_config": locked,
        "grouped_macro_f1": grouped,
        "random_macro_f1": random,
        "delta_random_minus_grouped": float(random - grouped),
        "note": (
            "A large positive gap suggests prompt memorization under random folds; "
            "never treat that gap as model improvement."
        ),
    }
    _write_json(paths["reports"] / "leakage_check.json", report)
    _update_run_state(paths["run_state"], stage="leakage-check", status="done")
    return report


def stage_train_full(cfg: ExperimentConfig) -> dict:
    """Fit locked procedure on all labelled clips and save final.pt."""
    paths = _dirs(cfg)
    nested_path = paths["reports"] / "nested_cv.json"
    gate_path = paths["reports"] / "gate0_frozen.json"
    frame = load_frame(cfg)
    device = _device(cfg)
    fp = manifest_fingerprint(cfg.manifest_path)
    bundle = load_pooled_cache(paths["cache"], _cache_expected(cfg, fp))
    X_all = _align_pooled(frame, bundle)
    y = frame["y"].to_numpy()

    nested: dict | None = None
    if nested_path.is_file():
        nested = json.loads(nested_path.read_text(encoding="utf-8"))
    use_frozen = nested is None or nested.get("skipped") or nested.get("prefer_frozen_mix")
    if use_frozen:
        if not gate_path.is_file():
            raise FileNotFoundError(
                "Need gate0_frozen.json for frozen train-full "
                "(nested_cv missing or skipped)"
            )
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        lrs = [x for x in gate.get("selected_scalar_lrs", []) if x is not None]
        scalar_lr = float(np.median(lrs)) if lrs else 1e-3
        # Prefer mix_mlp selected median epoch when present.
        refit_epochs = 50
        for fold in gate.get("honest", {}).get("folds", []):
            sel = fold.get("arms", {}).get("mix_mlp", {}).get("selected", {})
            if sel.get("refit_epochs"):
                refit_epochs = int(sel["refit_epochs"])
                break
        locked = {
            "mode": "frozen_scalar_mix",
            "scalar_lr": scalar_lr,
            "refit_epochs": refit_epochs,
        }
        seed_everything(PRIMARY_SEED)
        model = FrozenPooledClassifier(
            n_layers=X_all.shape[1],
            hidden_size=int(X_all.shape[2]),
            learn_mix=True,
        )
        cfg_t = TrainConfig(
            scalar_lr=scalar_lr,
            max_epochs=refit_epochs,
            seed=PRIMARY_SEED,
            effective_batch=16,
        )
        result = train_frozen_pooled_fixed(
            model, X_all, y, class_weights_from_y(y), cfg=cfg_t, device=device
        )
    else:
        locked = nested["locked_config"]
        endpoint = locked["endpoint"]
        depth = locked["active_depth"]
        start = max(0, endpoint - depth)
        # Convert median inner epoch to full-data steps.
        from fine_tuning.splits import epochs_for_matched_steps

        n_outer_fit_typical = int(0.8 * len(y))
        refit_epochs = epochs_for_matched_steps(
            inner_epochs=float(locked["median_epoch"]),
            n_inner_fit=int(0.8 * n_outer_fit_typical),
            n_outer_fit=len(y),
            batch_size=8,
        )
        partial_ctx = _partial_training_context(cfg)
        partial_ctx["boundary_bundle"](start)
        partial_ctx["reset_wavlm"]()
        seed_everything(PRIMARY_SEED)
        model = SuffixMixClassifier(
            partial_ctx["wavlm"],
            active_start=start,
            active_end=endpoint,
            n_frozen_lower=0,
        )
        batcher = BoundaryBatcher(
            boundary_bundle=partial_ctx["boundary_bundle"](start),
            clip_id_to_row=partial_ctx["clip_id_to_row"],
            clip_ids=partial_ctx["clip_ids"],
            frame_lengths=partial_ctx["frame_lengths"],
        )
        cfg_t = TrainConfig(
            scalar_lr=locked["scalar_lr"],
            encoder_lr=locked["encoder_lr"],
            max_epochs=refit_epochs,
            seed=PRIMARY_SEED,
            effective_batch=8,
        )
        print(
            f"[train-full] partial WavLM endpoint={endpoint} depth={depth} "
            f"epochs={cfg_t.max_epochs} n={len(y)}"
        )
        result = train_partial_fixed(
            model,
            row_indices=np.arange(len(y)),
            y=y,
            class_weights=class_weights_from_y(y),
            batcher=batcher,
            cfg=cfg_t,
            device=partial_ctx["device"],
            progress_label="train-full",
        )
        print(f"[train-full] done steps={result['optimizer_steps']}")
        locked = {
            "mode": "partial_wavlm",
            **locked,
            "active_start": start,
            "active_end": endpoint,
            "refit_epochs": refit_epochs,
        }

    payload = {
        "model_id": cfg.model_id,
        "protocol_version": PROTOCOL_VERSION,
        "locked": locked,
        "label_to_id": {
            k: int(v)
            for k, v in __import__(
                "fine_tuning.paths", fromlist=["LABEL_TO_ID"]
            ).LABEL_TO_ID.items()
        },
        "state_dict": result["state_dict"],
        "scalar_diagnostics": result.get("scalar_diagnostics"),
        "best_epoch": result.get("best_epoch"),
        "optimizer_steps": result.get("optimizer_steps"),
        "manifest_fingerprint": fp,
        "class_counts": {
            name: int((y == i).sum())
            for i, name in enumerate(
                __import__(
                    "fine_tuning.paths", fromlist=["VALENCE_CLASSES"]
                ).VALENCE_CLASSES
            )
        },
        "library_versions": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "transformers": __import__("transformers").__version__,
        },
    }
    out = paths["checkpoints"] / "final.pt"
    torch.save(payload, out)
    _update_run_state(
        paths["run_state"],
        stage="train-full",
        status="done",
        final=str(out),
        protocol_version=PROTOCOL_VERSION,
    )
    return {"final_checkpoint": str(out), "locked": locked}


def stage_preflight(cfg: ExperimentConfig) -> dict:
    """Memory / GradScaler / LayerDrop / SpecAugment / parity checks."""
    from fine_tuning.optimise import _autocast, _grad_scaler

    device = _device(cfg)
    report: dict[str, Any] = {
        "device": device,
        "protocol_version": PROTOCOL_VERSION,
    }
    try:
        processor, model = load_wavlm(cfg.model_id, device)
        configure_wavlm_for_partial(model)
        report["layerdrop"] = float(model.config.layerdrop)
        report["apply_spec_augment"] = bool(model.config.apply_spec_augment)
        report["transformers"] = __import__("transformers").__version__
        wav = np.zeros(16_000, dtype=np.float32)
        waves = [
            np.zeros(16_000, dtype=np.float32),
            np.zeros(24_000, dtype=np.float32),
            np.zeros(12_000, dtype=np.float32),
        ]
        parity = eval_suffix_parity(
            model, processor, waves, active_start=20, device=device
        )
        report["suffix_parity"] = parity
        inputs = processor(wav, sampling_rate=16_000, return_tensors="pt", padding=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        inputs.pop("attention_mask", None)
        plan = FreezePlan(active_start=20, active_end=24, retained_endpoint=24)
        seed_everything(0)
        clf = SuffixMixClassifier(
            model,
            active_start=plan.active_start,
            active_end=plan.active_end,
            n_frozen_lower=0,
        ).to(device)
        clf.train()
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
            boundary = out.hidden_states[plan.active_start].detach()
        frame_mask = torch.ones(boundary.shape[:2], device=device, dtype=torch.long)
        opt = torch.optim.AdamW(
            [p for p in clf.parameters() if p.requires_grad], lr=1e-5
        )
        scaler = _grad_scaler(device)
        opt.zero_grad(set_to_none=True)
        with _autocast(device):
            logits = clf(boundary, frame_mask)
            loss = logits.float().pow(2).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("preflight non-finite loss")
        scaler.scale(loss).backward()
        nonzero = []
        for n, p in clf.named_parameters():
            if (
                p.requires_grad
                and p.grad is not None
                and torch.isfinite(p.grad).all()
                and p.grad.abs().sum() > 0
            ):
                nonzero.append(n)
        scaler.step(opt)
        scaler.update()
        report["trainable_with_grad"] = nonzero[:20]
        report["n_trainable_with_grad"] = len(nonzero)
        report["status"] = "ok" if parity["passed"] else "failed"
        if not parity["passed"]:
            report["error"] = f"suffix parity failed: {parity}"
    except Exception as exc:  # noqa: BLE001
        report["status"] = "failed"
        report["error"] = str(exc)
    paths = _dirs(cfg)
    _write_json(paths["reports"] / "preflight.json", report)
    _update_run_state(paths["run_state"], stage="preflight", status=report["status"])
    if report["status"] != "ok" and cfg.stage in ("all", "preflight"):
        raise RuntimeError(f"preflight failed: {report.get('error')}")
    return report


def run_experiment(cfg: ExperimentConfig) -> dict:
    """Dispatch ``cfg.stage`` or run the full pipeline.

    ``--stage all`` always runs nested partial FT after Gate 0. Gate 0's
    ``prefer_frozen_mix`` flag is recorded for comparison but does not skip
    nested-cv / leakage-check / train-full on the full pipeline.
    """
    paths = _dirs(cfg)
    stage = cfg.stage
    # Full pipeline = measure partial FT end-to-end; do not early-exit on Gate 0.
    if stage == "all" and not cfg.ignore_gate0:
        cfg.ignore_gate0 = True
        print(
            "[run] stage=all: forcing nested partial FT "
            "(Gate 0 prefer_frozen_mix will not skip nested-cv)"
        )
    _update_run_state(
        paths["run_state"],
        stage=stage,
        status="running",
        protocol_version=PROTOCOL_VERSION,
        ignore_gate0=cfg.ignore_gate0,
    )
    results: dict[str, Any] = {}
    order = (
        [
            "preflight",
            "layer-cache",
            "frozen-gate",
            "nested-cv",
            "leakage-check",
            "train-full",
        ]
        if stage == "all"
        else [stage]
    )
    dispatch = {
        "preflight": stage_preflight,
        "layer-cache": stage_layer_cache,
        "frozen-gate": stage_frozen_gate,
        "nested-cv": stage_nested_cv,
        "leakage-check": stage_leakage_check,
        "train-full": stage_train_full,
    }
    for name in order:
        print(f"[run] starting stage={name}")
        results[name] = dispatch[name](cfg)
        print(f"[run] finished stage={name}")
    _update_run_state(paths["run_state"], stage=stage, status="done")
    return results
