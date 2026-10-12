"""Dense full WavLM fine-tuning with honest nested grouped CV."""

from __future__ import annotations

import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from fine_tuning.data import class_weights_from_y, load_wav_mono_16k, remap_audio_paths
from fine_tuning.full_ft_utils import (
    HEADLINE_RULE,
    decide_headline,
    duration_summary,
    evidence_criteria,
    evidence_label,
    label_aware_duration_batches,
    matched_encoder_epochs,
    minority_coverage,
)
from fine_tuning.metrics import (
    classification_report_dict,
    mcnemar_secondary,
    paired_group_bootstrap,
    paired_group_randomization,
)
from fine_tuning.model import (
    FullWavLMClassifier,
    configure_wavlm_for_partial,
    is_full_ft_trainable,
    load_wavlm,
    seed_everything,
)
from fine_tuning.optimise import TrainConfig, build_optimizer, linear_warmup_factor
from fine_tuning.paths import (
    FULL_FT_BATCH_SIZE,
    FULL_FT_ENCODER_LRS,
    FULL_FT_EPOCH_CHECKPOINTS,
    FULL_FT_HEAD_LRS,
    FULL_FT_HEAD_PRETRAIN_EPOCHS,
    FULL_FT_HEAD_WARMUP_EPOCHS,
    FULL_FT_MARGIN_DELTA,
    FULL_FT_NEG_F1_MARGIN,
    FULL_FT_N_RESAMPLES,
    FULL_FT_PRIMARY_POOL,
    FULL_FT_WARMUP_STEPS,
    LAST_LAYER_IDX,
    NEG_ID,
    OUTER_SEEDS,
    PRIMARY_SEED,
)
from fine_tuning.splits import FoldSpec, LabelAccessGuard


def count_trainable(module: nn.Module) -> tuple[int, int]:
    return (
        sum(p.numel() for p in module.parameters() if p.requires_grad),
        sum(p.numel() for p in module.parameters()),
    )


def assert_cnn_frozen_transformers_live(wavlm: nn.Module) -> None:
    for name, parameter in wavlm.named_parameters():
        expected = is_full_ft_trainable(name)
        if parameter.requires_grad != expected:
            raise AssertionError(
                f"{name}: requires_grad={parameter.requires_grad}, expected={expected}"
            )


def _set_encoder_trainable(model: FullWavLMClassifier, trainable: bool) -> None:
    for name, parameter in model.wavlm.named_parameters():
        if is_full_ft_trainable(name):
            parameter.requires_grad = trainable


def _pad_batch(
    waves: list[np.ndarray], processor, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = processor(
        waves,
        sampling_rate=16_000,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True,
    )
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        raise RuntimeError(
            "feature extractor returned no attention_mask; padded pooling is unsafe"
        )
    return inputs["input_values"].to(device), attention_mask.to(device)


def _audio_lengths_16k(paths: list[str]) -> np.ndarray:
    import soundfile as sf

    return np.asarray(
        [
            int(round(info.frames * 16_000 / info.samplerate))
            for info in (sf.info(path) for path in paths)
        ],
        dtype=np.int64,
    )


class WaveformBatcher:
    """Length-local, minority-aware batches returning positions into row_indices."""

    def __init__(
        self,
        *,
        waves: list[np.ndarray],
        processor,
        sample_lengths: np.ndarray,
    ):
        self.waves = waves
        self.processor = processor
        self.sample_lengths = np.asarray(sample_lengths, dtype=np.int64)

    def batches(
        self,
        row_indices: np.ndarray,
        labels: np.ndarray,
        batch_size: int,
        rng: np.random.Generator,
    ) -> list[np.ndarray]:
        return label_aware_duration_batches(
            self.sample_lengths[np.asarray(row_indices, dtype=np.int64)],
            np.asarray(labels, dtype=np.int64),
            batch_size,
            rng,
            minority_label=NEG_ID,
        )

    def collate(
        self, batch_rows: np.ndarray, device: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _pad_batch(
            [self.waves[int(row)] for row in batch_rows],
            self.processor,
            device,
        )


def _train_config(
    *,
    encoder_lr: float,
    head_lr: float,
    encoder_epochs: int,
    seed: int,
    batch_size: int,
) -> TrainConfig:
    return TrainConfig(
        head_lr=head_lr,
        encoder_lr=encoder_lr,
        max_epochs=int(encoder_epochs),
        seed=int(seed),
        effective_batch=int(batch_size),
        warmup_steps=FULL_FT_WARMUP_STEPS,
    )


def _cache_features(
    pooled_stack: np.ndarray, rows: np.ndarray, pool_mode: str
) -> np.ndarray:
    stack = pooled_stack[np.asarray(rows, dtype=np.int64)]
    if pool_mode == "final":
        return stack[:, LAST_LAYER_IDX, :]
    if pool_mode == "uniform":
        return stack.mean(axis=1)
    raise ValueError("cached matched control supports uniform/final only")


def _seeded_model(ctx: dict, pool_mode: str, seed: int) -> FullWavLMClassifier:
    # Seed before construction: head initialization must not depend on resume order.
    seed_everything(seed)
    return ctx["make_model"](pool_mode)


def pretrain_head_on_cache(
    model: FullWavLMClassifier,
    *,
    pooled_stack: np.ndarray,
    rows: np.ndarray,
    labels: np.ndarray,
    pool_mode: str,
    head_lr: float,
    epochs: int,
    seed: int,
    device: str,
    batch_size: int,
) -> dict:
    """Fit head on frozen fit-row features before either matched branch."""
    if epochs <= 0:
        return {"epochs": 0}
    seed_everything(seed)
    features = torch.from_numpy(
        np.ascontiguousarray(_cache_features(pooled_stack, rows, pool_mode))
    ).float().to(device)
    targets = torch.from_numpy(np.asarray(labels, dtype=np.int64)).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(
            class_weights_from_y(labels), device=device, dtype=torch.float32
        )
    )
    optimizer = torch.optim.AdamW(
        model.head.parameters(), lr=head_lr, weight_decay=0.01
    )
    rng = np.random.default_rng(seed)
    model.head.train()
    final_loss = float("nan")
    for _ in range(epochs):
        order = rng.permutation(len(labels))
        for start in range(0, len(labels), batch_size):
            positions = order[start : start + batch_size]
            idx = torch.from_numpy(positions).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model.head(features[idx]), targets[idx])
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu())
    return {
        "epochs": int(epochs),
        "n_rows": int(len(labels)),
        "final_loss": final_loss,
    }


def snapshot_encoder(model: FullWavLMClassifier) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.wavlm.named_parameters()
        if is_full_ft_trainable(name)
    }


def layer_update_ratios(
    before: dict[str, torch.Tensor], model: FullWavLMClassifier
) -> dict[str, float]:
    numerators: dict[str, float] = defaultdict(float)
    denominators: dict[str, float] = defaultdict(float)
    for name, parameter in model.wavlm.named_parameters():
        if name not in before:
            continue
        key = (
            ".".join(name.split(".")[:3])
            if name.startswith("encoder.layers.")
            else ".".join(name.split(".")[:2])
        )
        old = before[name].float()
        new = parameter.detach().float().cpu()
        numerators[key] += float(torch.sum((new - old) ** 2))
        denominators[key] += float(torch.sum(old**2))
    return {
        key: math.sqrt(numerators[key]) / max(math.sqrt(denominators[key]), 1e-12)
        for key in sorted(numerators)
    }


def train_full_ft_fixed(
    model: FullWavLMClassifier,
    *,
    row_indices: np.ndarray,
    labels: np.ndarray,
    class_weights: np.ndarray,
    batcher: WaveformBatcher,
    config: TrainConfig,
    device: str,
    eval_rows: np.ndarray | None = None,
    eval_labels: np.ndarray | None = None,
    epoch_checkpoints: tuple[int, ...] = FULL_FT_EPOCH_CHECKPOINTS,
    head_warmup_epochs: int = FULL_FT_HEAD_WARMUP_EPOCHS,
    encoder_updates: bool = True,
    progress_label: str | None = None,
) -> dict:
    """Train full branch; max_epochs and checkpoints count encoder epochs."""
    rows = np.asarray(row_indices, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    if len(rows) != len(labels):
        raise ValueError("labels must align with row_indices")
    if epoch_checkpoints and max(epoch_checkpoints) > config.max_epochs:
        raise ValueError("checkpoint exceeds configured encoder epochs")

    seed_everything(config.seed)
    model = model.to(device)
    _set_encoder_trainable(model, encoder_updates)
    encoder_parameters = [
        parameter
        for name, parameter in model.wavlm.named_parameters()
        if is_full_ft_trainable(name)
    ]
    encoder_ids = {id(parameter) for parameter in encoder_parameters}
    optimizer = build_optimizer(
        head=model.head,
        mix=model.mix if model.pool_mode == "mix" else None,
        encoder_params=encoder_parameters,
        cfg=config,
    )
    encoder_groups = [
        any(id(parameter) in encoder_ids for parameter in group["params"])
        for group in optimizer.param_groups
    ]
    if not any(encoder_groups):
        raise RuntimeError("optimizer has no encoder parameter group")
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, device=device, dtype=torch.float32)
    )
    rng = np.random.default_rng(config.seed)
    encoder_step = 0
    optimizer_steps = 0
    nonfinite = 0
    history: list[dict] = []
    checkpoint_scores: dict[int, float] = {}
    coverage: list[float] = []

    for epoch in range(1, head_warmup_epochs + config.max_epochs + 1):
        encoder_live = encoder_updates and epoch > head_warmup_epochs
        _set_encoder_trainable(model, encoder_live)
        model.train()
        batches = batcher.batches(rows, labels, config.effective_batch, rng)
        coverage.append(minority_coverage(batches, labels, NEG_ID))
        epoch_loss = 0.0
        completed_batches = 0
        for positions in batches:
            targets = torch.from_numpy(labels[positions]).to(device)
            input_values, attention_mask = batcher.collate(rows[positions], device)
            encoder_scale = (
                linear_warmup_factor(encoder_step, config.warmup_steps)
                if encoder_live
                else 0.0
            )
            for group, base_lr, is_encoder in zip(
                optimizer.param_groups, base_lrs, encoder_groups
            ):
                group["lr"] = base_lr * encoder_scale if is_encoder else base_lr
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(input_values, attention_mask), targets)
            if not torch.isfinite(loss):
                nonfinite += 1
                raise RuntimeError(
                    f"non-finite loss on paired branch step; aborting ({nonfinite})"
                )
            loss.backward()
            head_parameters = list(model.head.parameters())
            if model.pool_mode == "mix":
                head_parameters += list(model.mix.parameters())
            # Separate clipping keeps head optimization branch-invariant.
            torch.nn.utils.clip_grad_norm_(head_parameters, config.grad_clip)
            if encoder_live:
                torch.nn.utils.clip_grad_norm_(encoder_parameters, config.grad_clip)
            optimizer.step()
            optimizer_steps += 1
            # Logical encoder-stage step drives the same schedule in both branches.
            if epoch > head_warmup_epochs:
                encoder_step += 1
            epoch_loss += float(loss.item())
            completed_batches += 1

        encoder_epoch = epoch - head_warmup_epochs
        macro_f1 = None
        if (
            epoch > head_warmup_epochs
            and eval_rows is not None
            and eval_labels is not None
            and encoder_epoch in set(epoch_checkpoints)
        ):
            predictions = predict_full_ft(
                model,
                row_indices=eval_rows,
                batcher=batcher,
                device=device,
                batch_size=config.effective_batch,
            )
            macro_f1 = float(
                f1_score(eval_labels, predictions, average="macro", zero_division=0)
            )
            checkpoint_scores[encoder_epoch] = macro_f1
            if progress_label:
                print(
                    f"[{progress_label}] encoder_epoch {encoder_epoch}/"
                    f"{config.max_epochs} macro_f1={macro_f1:.4f}"
                )
        history.append(
            {
                "epoch": epoch,
                "encoder_epoch": encoder_epoch if encoder_live else 0,
                "train_loss": epoch_loss / max(1, completed_batches),
                "macro_f1": macro_f1,
            }
        )

    _set_encoder_trainable(model, encoder_updates)
    return {
        "checkpoint_scores": {
            int(epoch): float(score) for epoch, score in checkpoint_scores.items()
        },
        "history": history,
        "optimizer_steps": optimizer_steps,
        "encoder_steps": encoder_step,
        "minority_batch_coverage": float(np.mean(coverage)) if coverage else None,
    }


@torch.no_grad()
def predict_full_ft(
    model: FullWavLMClassifier,
    *,
    row_indices: np.ndarray,
    batcher: WaveformBatcher,
    device: str,
    batch_size: int = FULL_FT_BATCH_SIZE,
) -> np.ndarray:
    model = model.to(device)
    model.eval()
    rows = np.asarray(row_indices, dtype=np.int64)
    order = np.argsort(batcher.sample_lengths[rows], kind="stable")
    predictions = np.empty(len(rows), dtype=np.int64)
    for start in range(0, len(order), batch_size):
        positions = order[start : start + batch_size]
        inputs, masks = batcher.collate(rows[positions], device)
        predictions[positions] = model(inputs, masks).argmax(dim=1).cpu().numpy()
    return predictions


def make_full_ft_context(
    *,
    frame: pd.DataFrame,
    corpora_root: Path,
    model_id: str,
    device: str,
    pooled_stack: np.ndarray,
    wav_loader: Callable[[Path], np.ndarray] | None = None,
) -> dict[str, Any]:
    if len(pooled_stack) != len(frame):
        raise ValueError("pooled cache and manifest row counts differ")
    loader = wav_loader or load_wav_mono_16k
    remapped = remap_audio_paths(frame.copy(), corpora_root, strict=True)
    waves = [loader(Path(path)) for path in remapped["audio_path"].tolist()]
    lengths = np.asarray([len(wave) for wave in waves], dtype=np.int64)
    processor, wavlm = load_wavlm(model_id, device)
    configure_wavlm_for_partial(wavlm)
    base_state = {
        name: value.detach().cpu().clone()
        for name, value in wavlm.state_dict().items()
    }

    def make_model(pool_mode: str) -> FullWavLMClassifier:
        wavlm.load_state_dict(base_state)
        configure_wavlm_for_partial(wavlm)
        return FullWavLMClassifier(wavlm, pool_mode=pool_mode)

    def make_batcher() -> WaveformBatcher:
        return WaveformBatcher(
            waves=waves,
            processor=processor,
            sample_lengths=lengths,
        )

    return {
        "device": device,
        "make_model": make_model,
        "make_batcher": make_batcher,
        "pooled_stack": pooled_stack,
        "sample_lengths": lengths,
        "durations": duration_summary(lengths),
    }


def select_full_ft_config_inner(
    *,
    ctx: dict,
    y: np.ndarray,
    inners: list[FoldSpec],
    guard: LabelAccessGuard,
    seed: int,
    pool_mode: str = FULL_FT_PRIMARY_POOL,
    encoder_lrs: tuple[float, ...] = FULL_FT_ENCODER_LRS,
    head_lrs: tuple[float, ...] = FULL_FT_HEAD_LRS,
    epoch_checkpoints: tuple[int, ...] = FULL_FT_EPOCH_CHECKPOINTS,
    batch_size: int = FULL_FT_BATCH_SIZE,
    head_pretrain_epochs: int = FULL_FT_HEAD_PRETRAIN_EPOCHS,
    store: dict | None = None,
    persist: Callable[[], None] | None = None,
    progress_label: str | None = None,
) -> dict:
    """Select one executable (encoder LR, head LR, epoch) triple."""
    store = {} if store is None else store
    triple_scores: dict[tuple[float, float, int], list[float]] = {}
    for encoder_lr in encoder_lrs:
        for head_lr in head_lrs:
            for epoch in epoch_checkpoints:
                triple_scores[(encoder_lr, head_lr, epoch)] = []
            for inner_index, inner in enumerate(inners):
                run_key = (
                    f"{pool_mode}|enc{encoder_lr:g}|head{head_lr:g}|inner{inner_index}"
                )
                if run_key not in store:
                    fit_labels = guard.labels(y, inner.fit, "fit")
                    validation_labels = guard.labels(y, inner.score, "select")
                    run_seed = seed + inner_index * 100
                    model = _seeded_model(ctx, pool_mode, run_seed)
                    pretrain_head_on_cache(
                        model,
                        pooled_stack=ctx["pooled_stack"],
                        rows=inner.fit,
                        labels=fit_labels,
                        pool_mode=pool_mode,
                        head_lr=head_lr,
                        epochs=head_pretrain_epochs,
                        seed=run_seed,
                        device=ctx["device"],
                        batch_size=batch_size,
                    )
                    result = train_full_ft_fixed(
                        model,
                        row_indices=inner.fit,
                        labels=fit_labels,
                        class_weights=class_weights_from_y(fit_labels),
                        batcher=ctx["make_batcher"](),
                        config=_train_config(
                            encoder_lr=encoder_lr,
                            head_lr=head_lr,
                            encoder_epochs=max(epoch_checkpoints),
                            seed=run_seed,
                            batch_size=batch_size,
                        ),
                        device=ctx["device"],
                        eval_rows=inner.score,
                        eval_labels=validation_labels,
                        epoch_checkpoints=epoch_checkpoints,
                        progress_label=progress_label,
                    )
                    if set(result["checkpoint_scores"]) != set(epoch_checkpoints):
                        raise RuntimeError(f"incomplete checkpoint curve for {run_key}")
                    store[run_key] = {
                        "checkpoint_scores": {
                            str(epoch): float(score)
                            for epoch, score in result["checkpoint_scores"].items()
                        },
                        "minority_batch_coverage": result[
                            "minority_batch_coverage"
                        ],
                    }
                    if persist:
                        persist()
                for epoch in epoch_checkpoints:
                    triple_scores[(encoder_lr, head_lr, epoch)].append(
                        float(store[run_key]["checkpoint_scores"][str(epoch)])
                    )
    guard.assert_clean()

    means = {
        triple: float(np.mean(scores)) for triple, scores in triple_scores.items()
    }
    selected = max(
        means,
        key=lambda triple: (
            means[triple],
            -triple[2],
            -triple[0],
            -triple[1],
        ),
    )
    return {
        "encoder_lr": float(selected[0]),
        "head_lr": float(selected[1]),
        "epoch": int(selected[2]),
        "mean_inner_macro_f1": means[selected],
        "triple_means": {
            f"enc{encoder_lr:g}_head{head_lr:g}_epoch{epoch}": score
            for (encoder_lr, head_lr, epoch), score in means.items()
        },
        "tie_break": "mean macro-F1, earlier epoch, lower encoder/head LR",
    }


def train_eval_full_ft_honest(
    *,
    ctx: dict,
    y: np.ndarray,
    outer: FoldSpec,
    inners: list[FoldSpec],
    selected: dict,
    seed: int,
    pool_mode: str,
    batch_size: int,
    head_pretrain_epochs: int,
    progress_label: str | None = None,
) -> dict:
    """Refit selected triple once; produce full and exactly matched frozen OOF."""
    guard = LabelAccessGuard(outer)
    fit_labels = guard.labels(y, outer.fit, "fit")
    refit_epochs = matched_encoder_epochs(
        inner_encoder_epochs=int(selected["epoch"]),
        n_inner_fit=int(round(np.mean([len(inner.fit) for inner in inners]))),
        n_outer_fit=len(outer.fit),
        batch_size=batch_size,
    )

    full_model = _seeded_model(ctx, pool_mode, seed)
    full_pretrain = pretrain_head_on_cache(
        full_model,
        pooled_stack=ctx["pooled_stack"],
        rows=outer.fit,
        labels=fit_labels,
        pool_mode=pool_mode,
        head_lr=float(selected["head_lr"]),
        epochs=head_pretrain_epochs,
        seed=seed,
        device=ctx["device"],
        batch_size=batch_size,
    )
    before = snapshot_encoder(full_model)
    full_result = train_full_ft_fixed(
        full_model,
        row_indices=outer.fit,
        labels=fit_labels,
        class_weights=class_weights_from_y(fit_labels),
        batcher=ctx["make_batcher"](),
        config=_train_config(
            encoder_lr=float(selected["encoder_lr"]),
            head_lr=float(selected["head_lr"]),
            encoder_epochs=refit_epochs,
            seed=seed,
            batch_size=batch_size,
        ),
        device=ctx["device"],
        epoch_checkpoints=(),
        progress_label=progress_label,
    )
    full_predictions = predict_full_ft(
        full_model,
        row_indices=outer.score,
        batcher=ctx["make_batcher"](),
        device=ctx["device"],
        batch_size=batch_size,
    )
    update_ratios = layer_update_ratios(before, full_model)

    matched_model = _seeded_model(ctx, pool_mode, seed)
    matched_pretrain = pretrain_head_on_cache(
        matched_model,
        pooled_stack=ctx["pooled_stack"],
        rows=outer.fit,
        labels=fit_labels,
        pool_mode=pool_mode,
        head_lr=float(selected["head_lr"]),
        epochs=head_pretrain_epochs,
        seed=seed,
        device=ctx["device"],
        batch_size=batch_size,
    )
    matched_result = train_full_ft_fixed(
        matched_model,
        row_indices=outer.fit,
        labels=fit_labels,
        class_weights=class_weights_from_y(fit_labels),
        batcher=ctx["make_batcher"](),
        config=_train_config(
            encoder_lr=float(selected["encoder_lr"]),
            head_lr=float(selected["head_lr"]),
            encoder_epochs=refit_epochs,
            seed=seed,
            batch_size=batch_size,
        ),
        device=ctx["device"],
        epoch_checkpoints=(),
        encoder_updates=False,
        progress_label=f"{progress_label or 'full-ft'} matched-frozen",
    )
    matched_predictions = predict_full_ft(
        matched_model,
        row_indices=outer.score,
        batcher=ctx["make_batcher"](),
        device=ctx["device"],
        batch_size=batch_size,
    )
    if full_result["optimizer_steps"] != matched_result["optimizer_steps"]:
        raise RuntimeError("full and matched-frozen branches took different step counts")

    score_labels = guard.labels(y, outer.score, "final_score")
    guard.assert_clean()
    return {
        "pred_full": full_predictions,
        "pred_matched_frozen": matched_predictions,
        "full_report": classification_report_dict(score_labels, full_predictions),
        "matched_frozen_report": classification_report_dict(
            score_labels, matched_predictions
        ),
        "selected": {
            **selected,
            "refit_encoder_epochs": int(refit_epochs),
            "seed": int(seed),
            "pool_mode": pool_mode,
            "head_pretrain_epochs": int(head_pretrain_epochs),
        },
        "diagnostics": {
            "full_encoder_steps": full_result["encoder_steps"],
            "full_optimizer_steps": full_result["optimizer_steps"],
            "matched_frozen_optimizer_steps": matched_result["optimizer_steps"],
            "full_minority_batch_coverage": full_result[
                "minority_batch_coverage"
            ],
            "matched_minority_batch_coverage": matched_result[
                "minority_batch_coverage"
            ],
            "layer_update_ratios": update_ratios,
            "full_head_pretrain": full_pretrain,
            "matched_head_pretrain": matched_pretrain,
        },
        "audit": guard.summary(),
    }


def _macro(y: np.ndarray, predictions: np.ndarray) -> float:
    return float(classification_report_dict(y, predictions)["macro_f1"])


def _paired_block(
    y,
    groups,
    baseline,
    candidate,
    seed_deltas,
    *,
    seed,
    point_margin,
    negative_f1_margin,
    n_resamples,
) -> dict:
    boot = paired_group_bootstrap(
        y,
        baseline,
        candidate,
        groups,
        seed=seed,
        n_boot=n_resamples,
    )
    randomization = paired_group_randomization(
        y,
        baseline,
        candidate,
        groups,
        seed=seed,
        n_perm=n_resamples,
    )
    evidence = evidence_criteria(
        boot,
        randomization,
        seed_deltas,
        point_margin=point_margin,
        negative_f1_margin=negative_f1_margin,
    )
    return {
        **evidence,
        "label": evidence_label(evidence),
        "bootstrap": boot,
        "randomization": randomization,
        "mcnemar_secondary": mcnemar_secondary(y, baseline, candidate),
    }


def _comparative_training_gain(
    y,
    groups,
    full,
    matched_full_frozen,
    partial,
    separately_tuned_partial_frozen,
    *,
    seed,
    n_boot,
) -> dict:
    full_gain = _macro(y, full) - _macro(y, matched_full_frozen)
    partial_gain = _macro(y, partial) - _macro(y, separately_tuned_partial_frozen)
    rng = np.random.default_rng(seed)
    buckets: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups.tolist()):
        buckets[str(group)].append(index)
    keys = np.asarray(list(buckets))
    samples: list[float] = []
    for _ in range(n_boot):
        idx = np.concatenate(
            [buckets[key] for key in rng.choice(keys, len(keys), replace=True)]
        )
        samples.append(
            (_macro(y[idx], full[idx]) - _macro(y[idx], matched_full_frozen[idx]))
            - (
                _macro(y[idx], partial[idx])
                - _macro(y[idx], separately_tuned_partial_frozen[idx])
            )
        )
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return {
        "name": "comparative_training_gain_contrast",
        "full_gain_vs_matched_frozen": full_gain,
        "partial_gain_vs_separately_tuned_frozen": partial_gain,
        "difference": {
            "point": full_gain - partial_gain,
            "ci95": [float(lower), float(upper)],
            "excludes_zero": bool(lower > 0 or upper < 0),
        },
        "interpretation": (
            "descriptive recipe-level contrast, not a causal depth effect; pooling, "
            "head training, batching, and trainable parameters differ"
        ),
    }


def apply_headline_gate(
    *,
    y: np.ndarray,
    groups: np.ndarray,
    full_by_seed: dict[int, np.ndarray],
    matched_frozen_by_seed: dict[int, np.ndarray],
    partial_by_seed: dict[int, np.ndarray],
    tuned_frozen: np.ndarray,
    partial_separately_tuned_frozen: np.ndarray,
    primary_seed: int = PRIMARY_SEED,
    point_margin: float = FULL_FT_MARGIN_DELTA,
    negative_f1_margin: float = FULL_FT_NEG_F1_MARGIN,
    n_resamples: int = FULL_FT_N_RESAMPLES,
) -> dict[str, Any]:
    seeds = sorted(
        set(full_by_seed) & set(matched_frozen_by_seed) & set(partial_by_seed)
    )
    if primary_seed not in seeds:
        raise ValueError(f"primary seed {primary_seed} missing; available={seeds}")
    full = full_by_seed[primary_seed]
    matched = matched_frozen_by_seed[primary_seed]
    partial = partial_by_seed[primary_seed]

    condition_a = _paired_block(
        y,
        groups,
        partial,
        full,
        [_macro(y, full_by_seed[s]) - _macro(y, partial_by_seed[s]) for s in seeds],
        seed=primary_seed,
        point_margin=point_margin,
        negative_f1_margin=negative_f1_margin,
        n_resamples=n_resamples,
    )
    condition_b = _paired_block(
        y,
        groups,
        matched,
        full,
        [
            _macro(y, full_by_seed[s]) - _macro(y, matched_frozen_by_seed[s])
            for s in seeds
        ],
        seed=primary_seed,
        point_margin=point_margin,
        negative_f1_margin=negative_f1_margin,
        n_resamples=n_resamples,
    )
    condition_c = _paired_block(
        y,
        groups,
        partial,
        tuned_frozen,
        [
            _macro(y, tuned_frozen) - _macro(y, partial_by_seed[s])
            for s in seeds
        ],
        seed=primary_seed,
        point_margin=point_margin,
        negative_f1_margin=negative_f1_margin,
        n_resamples=n_resamples,
    )
    return {
        "primary_seed": primary_seed,
        "predeclared_seeds": seeds,
        "point_margin": point_margin,
        "margin_interpretation": (
            "practical adoption threshold on point estimate; uncertainty tests zero"
        ),
        "negative_f1_margin": negative_f1_margin,
        "n_bootstrap": n_resamples,
        "n_permutations": n_resamples,
        "condition_a_full_vs_partial": condition_a,
        "condition_b_full_vs_matched_frozen": condition_b,
        "condition_c_tuned_frozen_vs_partial": condition_c,
        "comparative_training_gain": _comparative_training_gain(
            y,
            groups,
            full,
            matched,
            partial,
            partial_separately_tuned_frozen,
            seed=primary_seed,
            n_boot=n_resamples,
        ),
        "headline": decide_headline(
            a_pass=condition_a["pass"],
            b_pass=condition_b["pass"],
            c_pass=condition_c["pass"],
        ),
        "headline_rule": HEADLINE_RULE,
    }


def load_partial_oof(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"clip_id", "question", "y_true", "pred_baseline"} | {
        f"pred_seed_{seed}" for seed in OUTER_SEEDS
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"partial OOF missing columns {sorted(missing)} in {path}")
    if frame["clip_id"].astype(str).duplicated().any():
        raise ValueError(f"duplicate clip_id in {path}")
    return frame


def align_partial_preds(frame: pd.DataFrame, partial_oof: pd.DataFrame) -> dict:
    """Require exact clip, label, and question equality before paired inference."""
    ids = frame["clip_id"].astype(str)
    partial_ids = partial_oof["clip_id"].astype(str)
    expected = set(ids)
    observed = set(partial_ids)
    if expected != observed:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            f"partial OOF clip set differs: missing={missing[:1]}, extra={extra[:1]}"
        )
    table = partial_oof.assign(clip_id=partial_ids).set_index("clip_id").loc[ids]
    if not np.array_equal(
        table["y_true"].to_numpy(dtype=np.int64),
        frame["y"].to_numpy(dtype=np.int64),
    ):
        raise ValueError("partial OOF labels disagree with manifest")
    if not np.array_equal(
        table["question"].astype(str).to_numpy(),
        frame["question"].astype(str).to_numpy(),
    ):
        raise ValueError("partial OOF question groups disagree with manifest")
    return {
        "by_seed": {
            seed: table[f"pred_seed_{seed}"].to_numpy(dtype=np.int64)
            for seed in OUTER_SEEDS
        },
        "separately_tuned_frozen": table["pred_baseline"].to_numpy(dtype=np.int64),
    }


def paired_leakage_gap_bootstrap(
    *,
    y: np.ndarray,
    groups: np.ndarray,
    grouped_predictions: np.ndarray,
    random_predictions: np.ndarray,
    seed: int,
    n_boot: int = FULL_FT_N_RESAMPLES,
) -> dict:
    """Question-cluster bootstrap of random-CV minus grouped-CV macro-F1."""
    point = _macro(y, random_predictions) - _macro(y, grouped_predictions)
    buckets: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups.tolist()):
        buckets[str(group)].append(index)
    keys = np.asarray(list(buckets))
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(n_boot):
        idx = np.concatenate(
            [buckets[key] for key in rng.choice(keys, len(keys), replace=True)]
        )
        samples.append(
            _macro(y[idx], random_predictions[idx])
            - _macro(y[idx], grouped_predictions[idx])
        )
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return {
        "point": float(point),
        "ci95": [float(lower), float(upper)],
        "n_bootstrap": n_boot,
        "conditioning": (
            "paired over clips and clustered by question; conditional on chosen "
            "grouped/random fold assignments"
        ),
    }


def run_full_ft_smoke(
    *,
    frame: pd.DataFrame,
    corpora_root: Path,
    model_id: str,
    device: str,
    seed: int,
    n_clips: int,
    pool_modes: tuple[str, ...],
    pooled_stack: np.ndarray | None = None,
    timed_steps: int = 3,
) -> dict[str, Any]:
    """Worst-case padded fp32 smoke with optional live/cache parity."""
    seed_everything(seed)
    remapped = remap_audio_paths(frame.copy(), corpora_root, strict=True)
    lengths = _audio_lengths_16k(remapped["audio_path"].tolist())
    selected_rows = np.argsort(lengths, kind="stable")[-n_clips:]
    subset = remapped.iloc[selected_rows].reset_index(drop=True)
    waves = [load_wav_mono_16k(path) for path in subset["audio_path"]]
    targets = torch.tensor(subset["y"].to_numpy(), device=device, dtype=torch.long)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(
            class_weights_from_y(subset["y"].to_numpy()),
            device=device,
            dtype=torch.float32,
        )
    )
    processor, wavlm = load_wavlm(model_id, device)
    base_state = {
        name: value.detach().cpu().clone()
        for name, value in wavlm.state_dict().items()
    }
    inputs, masks = _pad_batch(waves, processor, device)
    arms: dict[str, Any] = {}
    for pool_mode in pool_modes:
        seed_everything(seed)
        wavlm.load_state_dict(base_state)
        configure_wavlm_for_partial(wavlm)
        model = FullWavLMClassifier(wavlm, pool_mode=pool_mode)
        assert_cnn_frozen_transformers_live(wavlm)
        parity = None
        if pooled_stack is not None:
            model.eval()
            with torch.no_grad():
                live = model.pooled(inputs, masks).float().cpu().numpy()
            cached = _cache_features(pooled_stack, selected_rows, pool_mode)
            difference = np.abs(live - cached)
            parity = {
                "max_abs_diff": float(difference.max()),
                "mean_abs_diff": float(difference.mean()),
                "tolerance": 1e-3,
                "passed": bool(difference.max() < 1e-3),
            }
            if not parity["passed"]:
                raise RuntimeError(f"live/cache parity failed for {pool_mode}: {parity}")
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-5
        )
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        times: list[float] = []
        losses: list[float] = []
        model.train()
        for _ in range(timed_steps):
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs, masks), targets)
            loss.backward()
            optimizer.step()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.perf_counter() - started)
            losses.append(float(loss.detach().cpu()))
        steady = times[1:] if len(times) > 1 else times
        trainable, total = count_trainable(model)
        arms[pool_mode] = {
            "losses": losses,
            "step_seconds": times,
            "seconds_per_clip_step_steady": float(np.mean(steady) / len(waves)),
            "peak_mem_mb": (
                float(torch.cuda.max_memory_allocated() / 1024**2)
                if torch.cuda.is_available()
                else None
            ),
            "trainable_params": trainable,
            "total_params": total,
            "cache_parity": parity,
        }
    return {
        "status": "ok",
        "protocol": "full_ft_smoke_fp32_longest_clips",
        "device": device,
        "model_id": model_id,
        "durations_all_clips": duration_summary(lengths),
        "smoke_clip_ids": subset["clip_id"].astype(str).tolist(),
        "smoke_clip_seconds": [len(wave) / 16_000 for wave in waves],
        "arms": arms,
    }
