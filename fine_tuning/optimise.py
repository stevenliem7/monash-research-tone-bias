"""Optimizers, duration batching, fixed-length trainers, and suffix parity."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch import amp

from fine_tuning.metrics import scalar_weight_diagnostics
from fine_tuning.model import (
    FrozenPooledClassifier,
    PartialWavLMClassifier,
    SuffixMixClassifier,
    call_wavlm_encoder_layer,
    mask_aware_mean_pool,
    seed_everything,
)
from fine_tuning.splits import (
    FoldSpec,
    LabelAccessGuard,
    epochs_for_matched_steps,
)


def _use_cuda(device: str) -> bool:
    return device.startswith("cuda") and torch.cuda.is_available()


def _grad_scaler(device: str) -> amp.GradScaler:
    if _use_cuda(device):
        return amp.GradScaler("cuda")
    return amp.GradScaler("cpu", enabled=False)


def _autocast(device: str):
    if _use_cuda(device):
        return amp.autocast("cuda")
    return amp.autocast("cpu", enabled=False)


def _ce_loss(
    criterion: nn.CrossEntropyLoss,
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy in fp32 so fp16 autocast logits match fp32 class weights."""
    return criterion(logits.float(), targets)


@dataclass
class BoundaryBatch:
    hidden: torch.Tensor
    frame_mask: torch.Tensor
    frozen_lower: torch.Tensor | None = None

    @property
    def lower_pooled(self) -> torch.Tensor | None:
        return self.frozen_lower


@dataclass
class TrainConfig:
    head_lr: float = 1e-4
    scalar_lr: float = 1e-3
    encoder_lr: float = 1e-5
    weight_decay: float = 0.01
    max_epochs: int = 15
    min_epochs: int = 5
    patience: int = 3
    grad_clip: float = 1.0
    effective_batch: int = 16
    physical_batch: int = 4
    warmup_ratio: float = 0.1
    warmup_steps: int = 30
    seed: int = 42
    max_nonfinite: int = 5


def build_optimizer(
    *,
    head: nn.Module,
    mix: nn.Module | None,
    encoder_params: list[nn.Parameter] | None,
    cfg: TrainConfig,
) -> torch.optim.AdamW:
    groups = [
        {
            "params": list(head.parameters()),
            "lr": cfg.head_lr,
            "weight_decay": cfg.weight_decay,
        }
    ]
    if mix is not None:
        groups.append(
            {
                "params": list(mix.parameters()),
                "lr": cfg.scalar_lr,
                "weight_decay": 0.0,
            }
        )
    if encoder_params:
        groups.append(
            {
                "params": encoder_params,
                "lr": cfg.encoder_lr,
                "weight_decay": cfg.weight_decay,
            }
        )
    return torch.optim.AdamW(groups)


def linear_warmup_factor(step: int, warmup_steps: int) -> float:
    """Constant LR after warmup (needed for epoch transfer to matched refits)."""
    return min(1.0, float(step + 1) / float(max(1, warmup_steps)))


def duration_batches(
    row_indices: np.ndarray,
    lengths: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Sort by frame length, form contiguous batches, then shuffle batch order.

    Contiguous length locality keeps padding small. Do not scatter minority
    labels across batches; that destroys the length sort.
    """
    if len(row_indices) == 0:
        return []
    order = np.argsort(lengths[row_indices], kind="stable")
    sorted_rows = row_indices[order]
    batches = [
        sorted_rows[start : start + batch_size]
        for start in range(0, len(sorted_rows), batch_size)
    ]
    rng.shuffle(batches)
    return batches


def collate_boundary_batch(
    row_indices: np.ndarray,
    *,
    boundary_bundle: dict,
    clip_id_to_row: dict[str, int],
    clip_ids: list[str],
    frozen_lower: np.ndarray | None = None,
    lower_pooled: np.ndarray | None = None,
    device: str,
) -> BoundaryBatch:
    """Pad variable-length boundary hiddens for one mini-batch."""
    lower_src = frozen_lower if frozen_lower is not None else lower_pooled
    hiddens: list[torch.Tensor] = []
    lengths: list[int] = []
    lower: list[torch.Tensor] = []
    for row_i in row_indices:
        clip = clip_ids[int(row_i)]
        b_idx = clip_id_to_row[clip]
        seq_len = int(boundary_bundle["lengths"][b_idx])
        hidden = torch.from_numpy(boundary_bundle["hiddens"][b_idx][:seq_len])
        hiddens.append(hidden)
        lengths.append(seq_len)
        if lower_src is not None:
            lower.append(torch.from_numpy(lower_src[int(row_i)]))
    max_len = max(lengths)
    hidden_size = hiddens[0].shape[-1]
    batch = torch.zeros(len(row_indices), max_len, hidden_size, dtype=hiddens[0].dtype)
    frame_mask = torch.zeros(len(row_indices), max_len, dtype=torch.long)
    for i, (h, seq_len) in enumerate(zip(hiddens, lengths)):
        batch[i, :seq_len] = h
        frame_mask[i, :seq_len] = 1
    lower_t = torch.stack(lower, dim=0) if lower else None
    return BoundaryBatch(
        hidden=batch.to(device),
        frame_mask=frame_mask.to(device),
        frozen_lower=lower_t.to(device) if lower_t is not None else None,
    )


class BoundaryBatcher:
    """Length-aware batching over a cached boundary bundle."""

    def __init__(
        self,
        *,
        boundary_bundle: dict,
        clip_id_to_row: dict[str, int],
        clip_ids: list[str],
        frame_lengths: np.ndarray,
        frozen_lower: np.ndarray | None = None,
    ):
        self.boundary_bundle = boundary_bundle
        self.clip_id_to_row = clip_id_to_row
        self.clip_ids = clip_ids
        self.frame_lengths = np.asarray(frame_lengths)
        self.frozen_lower = frozen_lower

    def batches(
        self,
        row_indices: np.ndarray,
        batch_size: int,
        rng: np.random.Generator,
    ) -> list[np.ndarray]:
        return duration_batches(row_indices, self.frame_lengths, batch_size, rng)

    def collate(self, row_indices: np.ndarray, device: str) -> BoundaryBatch:
        return collate_boundary_batch(
            row_indices,
            boundary_bundle=self.boundary_bundle,
            clip_id_to_row=self.clip_id_to_row,
            clip_ids=self.clip_ids,
            frozen_lower=self.frozen_lower,
            device=device,
        )


def eval_suffix_parity(
    wavlm: nn.Module,
    processor,
    waves: list[np.ndarray],
    *,
    active_start: int,
    device: str,
    tolerance: float = 2e-2,
) -> dict:
    """Padded-batch suffix must match full WavLM on the same padded inputs.

    Uses the real 2D frame mask (1=valid). Persists max/mean abs error for
    preflight.json.
    """
    wavlm.eval()
    inputs = processor(
        waves,
        sampling_rate=16_000,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        full = wavlm(**inputs, output_hidden_states=True)
        reference = full.hidden_states[-1]
        boundary = full.hidden_states[active_start]
        frame_mask = wavlm._get_feature_vector_attention_mask(
            boundary.shape[1], inputs["attention_mask"]
        )
        hidden = boundary.clone()
        expand = frame_mask.unsqueeze(-1).repeat(1, 1, hidden.shape[2])
        hidden = hidden.masked_fill(~expand.bool(), 0.0)
        attn0 = wavlm.encoder.layers[0].attention
        bias = attn0.compute_bias(hidden.shape[1], hidden.shape[1])
        bias = bias.unsqueeze(0).repeat(hidden.shape[0], 1, 1, 1)
        position_bias = bias.view(
            hidden.shape[0] * attn0.num_heads, hidden.shape[1], hidden.shape[1]
        ).to(device=hidden.device, dtype=hidden.dtype)
        n_layers = len(wavlm.encoder.layers)
        for i in range(active_start, n_layers):
            hidden, position_bias = call_wavlm_encoder_layer(
                wavlm.encoder.layers[i],
                hidden,
                attention_mask=frame_mask,
                position_bias=position_bias,
                layer_idx=i,
            )
        if hasattr(wavlm.encoder, "layer_norm"):
            hidden = wavlm.encoder.layer_norm(hidden)

    keep = frame_mask.to(torch.bool)
    diff = (hidden - reference)[keep].abs()
    result = {
        "active_start": active_start,
        "batch": len(waves),
        "seq_len": int(hidden.shape[1]),
        "max_abs_diff": float(diff.max()) if diff.numel() else 0.0,
        "mean_abs_diff": float(diff.mean()) if diff.numel() else 0.0,
        "tolerance": tolerance,
        "mask_contract": "2d_binary_1_valid",
    }
    result["passed"] = bool(result["max_abs_diff"] < tolerance)
    if not result["passed"]:
        raise RuntimeError(f"suffix parity failed: {result}")
    return result


def train_frozen_pooled_fixed(
    model: FrozenPooledClassifier,
    X: np.ndarray,
    y: np.ndarray,
    class_weights: np.ndarray,
    *,
    cfg: TrainConfig,
    device: str,
    X_eval: np.ndarray | None = None,
    y_eval: np.ndarray | None = None,
) -> dict:
    """Fixed-length pooled training. No early stopping on any held-out set."""
    seed_everything(cfg.seed)
    model = model.to(device)
    mix = model.mix if model.learn_mix else None
    opt = build_optimizer(head=model.head, mix=mix, encoder_params=None, cfg=cfg)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, device=device, dtype=torch.float32)
    )
    n = len(y)
    rng = np.random.default_rng(cfg.seed)
    x_t = torch.from_numpy(X).to(device)
    y_t = torch.from_numpy(y.astype(np.int64)).to(device)
    base_lrs = [g["lr"] for g in opt.param_groups]
    step = 0
    history: list[dict] = []
    curve: list[float] = []

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        order = rng.permutation(n)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, cfg.effective_batch):
            idx = order[start : start + cfg.effective_batch]
            scale = linear_warmup_factor(step, cfg.warmup_steps)
            for g, base in zip(opt.param_groups, base_lrs):
                g["lr"] = base * scale
            opt.zero_grad(set_to_none=True)
            logits = model(x_t[idx])
            loss = _ce_loss(criterion, logits, y_t[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], cfg.grad_clip
            )
            opt.step()
            step += 1
            epoch_loss += float(loss.item())
            n_batches += 1
        macro = None
        if X_eval is not None and y_eval is not None:
            preds = predict_frozen_pooled(model, X_eval, device)
            macro = float(f1_score(y_eval, preds, average="macro", zero_division=0))
            curve.append(macro)
        history.append(
            {
                "epoch": epoch,
                "train_loss": epoch_loss / max(1, n_batches),
                "macro_f1": macro,
            }
        )

    mix_diag = None
    if model.learn_mix:
        mix_diag = scalar_weight_diagnostics(model.mix.logits.detach().cpu().numpy())
    return {
        "best_epoch": cfg.max_epochs,
        "history": history,
        "curve": curve,
        "scalar_diagnostics": mix_diag,
        "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "optimizer_steps": step,
    }


# Backwards-compatible name used by older call sites (no early-stop on val).
def train_frozen_pooled(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    class_weights: np.ndarray,
    *,
    cfg: TrainConfig,
    device: str,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
) -> dict:
    return train_frozen_pooled_fixed(
        model,
        X,
        y,
        class_weights,
        cfg=cfg,
        device=device,
        X_eval=X_val,
        y_eval=y_val,
    )


@torch.no_grad()
def predict_frozen_pooled(
    model: nn.Module, X: np.ndarray, device: str
) -> np.ndarray:
    model = model.to(device)
    model.eval()
    logits = model(torch.from_numpy(X).to(device))
    return logits.argmax(dim=1).cpu().numpy()


def _encoder_params(model: SuffixMixClassifier) -> list[nn.Parameter]:
    return [p for p in model.wavlm.parameters() if p.requires_grad]


def train_partial_fixed(
    model: SuffixMixClassifier,
    *,
    row_indices: np.ndarray,
    y: np.ndarray,
    class_weights: np.ndarray,
    batcher: BoundaryBatcher,
    cfg: TrainConfig,
    device: str,
    eval_rows: np.ndarray | None = None,
    y_eval: np.ndarray | None = None,
    progress_label: str | None = None,
) -> dict:
    """Fixed-length partial FT. No checkpointing from an eval fold."""
    warnings.filterwarnings(
        "ignore",
        message=".*torch\\.cuda\\.amp\\.GradScaler.*",
        category=FutureWarning,
    )
    seed_everything(cfg.seed)
    model = model.to(device)
    opt = build_optimizer(
        head=model.head,
        mix=model.mix,
        encoder_params=_encoder_params(model),
        cfg=cfg,
    )
    scaler = _grad_scaler(device)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, device=device, dtype=torch.float32)
    )
    rng = np.random.default_rng(cfg.seed)
    base_lrs = [g["lr"] for g in opt.param_groups]
    step = 0
    nonfinite = 0
    history: list[dict] = []
    curve: list[float] = []

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch_rows in batcher.batches(row_indices, cfg.effective_batch, rng):
            # Map batch_rows (absolute indices) to labels via position in row_indices.
            pos = np.asarray(
                [int(np.where(row_indices == r)[0][0]) for r in batch_rows],
                dtype=np.int64,
            )
            yb = torch.from_numpy(y[pos].astype(np.int64)).to(device)
            batch = batcher.collate(batch_rows, device)
            scale = linear_warmup_factor(step, cfg.warmup_steps)
            for g, base in zip(opt.param_groups, base_lrs):
                g["lr"] = base * scale
            opt.zero_grad(set_to_none=True)
            with _autocast(device):
                logits = model(batch.hidden, batch.frame_mask, batch.frozen_lower)
                loss = _ce_loss(criterion, logits, yb)
            if not torch.isfinite(loss):
                nonfinite += 1
                if nonfinite >= cfg.max_nonfinite:
                    raise RuntimeError(f"Too many non-finite losses ({nonfinite})")
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], cfg.grad_clip
            )
            scaler.step(opt)
            scaler.update()
            step += 1
            epoch_loss += float(loss.item())
            n_batches += 1

        macro = None
        if eval_rows is not None and y_eval is not None:
            preds = predict_partial_wavlm(
                model,
                row_indices=eval_rows,
                batcher=batcher,
                device=device,
                batch_size=cfg.effective_batch,
            )
            macro = float(f1_score(y_eval, preds, average="macro", zero_division=0))
            curve.append(macro)
            if progress_label and (epoch == 1 or epoch % 2 == 0):
                print(
                    f"[{progress_label}] epoch {epoch}/{cfg.max_epochs} "
                    f"loss={epoch_loss / max(1, n_batches):.4f} macro_f1={macro:.4f}"
                )
        history.append(
            {
                "epoch": epoch,
                "train_loss": epoch_loss / max(1, n_batches),
                "macro_f1": macro,
            }
        )

    mix_diag = scalar_weight_diagnostics(model.mix.logits.detach().cpu().numpy())
    return {
        "best_epoch": int(np.argmax(curve)) + 1 if curve else cfg.max_epochs,
        "curve": curve,
        "history": history,
        "scalar_diagnostics": mix_diag,
        "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "optimizer_steps": step,
    }


def train_partial_wavlm(
    model: PartialWavLMClassifier,
    *,
    row_indices: np.ndarray,
    y: np.ndarray,
    class_weights: np.ndarray,
    boundary_bundle: dict,
    clip_id_to_row: dict[str, int],
    clip_ids: list[str],
    lower_pooled: np.ndarray | None,
    cfg: TrainConfig,
    device: str,
    val_row_indices: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    progress_label: str | None = None,
    frame_lengths: np.ndarray | None = None,
) -> dict:
    """Compatibility wrapper around fixed-length partial training."""
    if frame_lengths is None:
        # Fall back to boundary lengths aligned by clip order in the bundle.
        lengths = np.zeros(len(clip_ids), dtype=np.int64)
        for i, clip in enumerate(clip_ids):
            b_idx = clip_id_to_row[clip]
            lengths[i] = int(boundary_bundle["lengths"][b_idx])
        frame_lengths = lengths
    batcher = BoundaryBatcher(
        boundary_bundle=boundary_bundle,
        clip_id_to_row=clip_id_to_row,
        clip_ids=clip_ids,
        frame_lengths=frame_lengths,
        frozen_lower=lower_pooled,
    )
    return train_partial_fixed(
        model,
        row_indices=row_indices,
        y=y,
        class_weights=class_weights,
        batcher=batcher,
        cfg=cfg,
        device=device,
        eval_rows=val_row_indices,
        y_eval=y_val,
        progress_label=progress_label,
    )


@torch.no_grad()
def predict_partial_wavlm(
    model: SuffixMixClassifier,
    *,
    row_indices: np.ndarray,
    device: str,
    batch_size: int = 16,
    batcher: BoundaryBatcher | None = None,
    boundary_bundle: dict | None = None,
    clip_id_to_row: dict[str, int] | None = None,
    clip_ids: list[str] | None = None,
    lower_pooled: np.ndarray | None = None,
) -> np.ndarray:
    model = model.to(device)
    model.eval()
    if batcher is None:
        assert boundary_bundle is not None and clip_id_to_row is not None and clip_ids is not None
        lengths = np.zeros(len(clip_ids), dtype=np.int64)
        for i, clip in enumerate(clip_ids):
            lengths[i] = int(boundary_bundle["lengths"][clip_id_to_row[clip]])
        batcher = BoundaryBatcher(
            boundary_bundle=boundary_bundle,
            clip_id_to_row=clip_id_to_row,
            clip_ids=clip_ids,
            frame_lengths=lengths,
            frozen_lower=lower_pooled,
        )
    preds: list[np.ndarray] = []
    for start in range(0, len(row_indices), batch_size):
        batch_rows = row_indices[start : start + batch_size]
        batch = batcher.collate(batch_rows, device)
        with _autocast(device):
            logits = model(batch.hidden, batch.frame_mask, batch.frozen_lower)
        preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds, axis=0)


def select_physical_batch(
    try_fn,
    candidates: list[tuple[int, int]] | None = None,
) -> tuple[int, int]:
    """Try (physical, accumulation) pairs until try_fn succeeds."""
    if candidates is None:
        candidates = [(4, 4), (2, 8), (1, 16)]
    last_err: Exception | None = None
    for physical, accum in candidates:
        try:
            try_fn(physical, accum)
            return physical, accum
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() and "cuda" not in str(exc).lower():
                raise
            last_err = exc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    raise RuntimeError(f"All batch sizes OOM; last error: {last_err}")


def train_eval_pooled_honest(
    *,
    X: np.ndarray,
    y: np.ndarray,
    outer: FoldSpec,
    inners: list[FoldSpec],
    learn_mix: bool,
    scalar_lr: float | None,
    seed: int,
    device: str,
    max_epochs: int = 100,
    batch_size: int = 16,
    lr_grid: tuple[float, ...] | None = None,
) -> dict:
    """Inner epoch (+ optional LR) selection, matched-step outer refit, score once."""
    from fine_tuning.data import class_weights_from_y
    from fine_tuning.metrics import classification_report_dict

    guard = LabelAccessGuard(outer)
    hidden = int(X.shape[-1])
    n_layers = int(X.shape[1])
    grid = lr_grid if (learn_mix and lr_grid) else ((scalar_lr or 1e-3),)

    per_lr: dict[float, dict] = {}
    for lr in grid:
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
                model.mix.logits.requires_grad_(False)
            cfg = TrainConfig(
                scalar_lr=lr,
                max_epochs=max_epochs,
                seed=seed,
                effective_batch=batch_size,
            )
            out = train_frozen_pooled_fixed(
                model,
                X[inner.fit],
                y_fit,
                class_weights_from_y(y_fit),
                cfg=cfg,
                device=device,
                X_eval=X[inner.score],
                y_eval=y_val,
            )
            curve = out["curve"]
            best_epochs.append(int(np.argmax(curve)) + 1 if curve else max_epochs)
            best_scores.append(float(np.max(curve)) if curve else -1.0)
        per_lr[lr] = {
            "mean_macro_f1": float(np.mean(best_scores)),
            "median_epoch": float(np.median(best_epochs)),
            "epochs": best_epochs,
        }

    best_lr = max(per_lr, key=lambda k: per_lr[k]["mean_macro_f1"])
    chosen = per_lr[best_lr]
    refit_epochs = epochs_for_matched_steps(
        inner_epochs=chosen["median_epoch"],
        n_inner_fit=int(np.mean([len(i.fit) for i in inners])),
        n_outer_fit=len(outer.fit),
        batch_size=batch_size,
    )
    seed_everything(seed)
    model = FrozenPooledClassifier(
        n_layers=n_layers, hidden_size=hidden, learn_mix=learn_mix
    )
    if not learn_mix:
        model.mix.logits.requires_grad_(False)
    y_fit = guard.labels(y, outer.fit, "fit")
    cfg = TrainConfig(
        scalar_lr=best_lr,
        max_epochs=refit_epochs,
        seed=seed,
        effective_batch=batch_size,
    )
    result = train_frozen_pooled_fixed(
        model,
        X[outer.fit],
        y_fit,
        class_weights_from_y(y_fit),
        cfg=cfg,
        device=device,
    )
    model.load_state_dict(result["state_dict"])
    pred = predict_frozen_pooled(model, X[outer.score], device)
    guard.assert_clean()
    return {
        **result,
        "pred": pred,
        "val_report": classification_report_dict(y[outer.score], pred),
        "selected": {
            "scalar_lr": best_lr if learn_mix else None,
            "inner_median_epoch": chosen["median_epoch"],
            "refit_epochs": refit_epochs,
        },
        "inner_mean_macro_f1": chosen["mean_macro_f1"],
        "model": model,
        "audit": guard.summary(),
    }


def train_eval_partial_honest(
    *,
    ctx: dict,
    y: np.ndarray,
    outer: FoldSpec,
    inners: list[FoldSpec],
    endpoint: int,
    depth: int,
    encoder_lr: float,
    scalar_lr: float,
    seed: int,
    max_epochs: int = 15,
    progress_label: str | None = None,
) -> dict:
    """Inner epoch selection on partial FT, matched-step outer refit, score once."""
    from fine_tuning.data import class_weights_from_y
    from fine_tuning.metrics import classification_report_dict
    from fine_tuning.model import parameter_update_ratios, snapshot_trainable

    guard = LabelAccessGuard(outer)
    active_start = max(0, endpoint - depth)
    n_mix = depth if depth > 1 else 1
    batch_size = 8

    def make_model() -> SuffixMixClassifier:
        ctx["reset_wavlm"]()
        return SuffixMixClassifier(
            ctx["wavlm"],
            active_start=active_start,
            active_end=endpoint,
            n_frozen_lower=0,
        )

    def make_batcher() -> BoundaryBatcher:
        boundary = ctx["boundary_bundle"](active_start)
        return BoundaryBatcher(
            boundary_bundle=boundary,
            clip_id_to_row=ctx["clip_id_to_row"],
            clip_ids=ctx["clip_ids"],
            frame_lengths=ctx["frame_lengths"],
            frozen_lower=None,
        )

    best_epochs: list[int] = []
    best_scores: list[float] = []
    for inner in inners:
        y_fit = guard.labels(y, inner.fit, "fit")
        y_val = guard.labels(y, inner.score, "select")
        model = make_model()
        batcher = make_batcher()
        cfg = TrainConfig(
            scalar_lr=scalar_lr,
            encoder_lr=encoder_lr,
            max_epochs=max_epochs,
            seed=seed,
            effective_batch=batch_size,
        )
        out = train_partial_fixed(
            model,
            row_indices=inner.fit,
            y=y_fit,
            class_weights=class_weights_from_y(y_fit),
            batcher=batcher,
            cfg=cfg,
            device=ctx["device"],
            eval_rows=inner.score,
            y_eval=y_val,
            progress_label=progress_label,
        )
        curve = out["curve"]
        best_epochs.append(int(np.argmax(curve)) + 1 if curve else max_epochs)
        best_scores.append(float(np.max(curve)) if curve else -1.0)
        del model

    refit_epochs = epochs_for_matched_steps(
        inner_epochs=float(np.median(best_epochs)),
        n_inner_fit=int(np.mean([len(i.fit) for i in inners])),
        n_outer_fit=len(outer.fit),
        batch_size=batch_size,
    )
    model = make_model()
    batcher = make_batcher()
    before = snapshot_trainable(model)
    pred_before = predict_partial_wavlm(
        model,
        row_indices=outer.score,
        batcher=batcher,
        device=ctx["device"],
        batch_size=batch_size,
    )
    y_fit = guard.labels(y, outer.fit, "fit")
    cfg = TrainConfig(
        scalar_lr=scalar_lr,
        encoder_lr=encoder_lr,
        max_epochs=refit_epochs,
        seed=seed,
        effective_batch=batch_size,
    )
    result = train_partial_fixed(
        model,
        row_indices=outer.fit,
        y=y_fit,
        class_weights=class_weights_from_y(y_fit),
        batcher=batcher,
        cfg=cfg,
        device=ctx["device"],
        progress_label=progress_label,
    )
    model.load_state_dict(result["state_dict"])
    pred = predict_partial_wavlm(
        model,
        row_indices=outer.score,
        batcher=batcher,
        device=ctx["device"],
        batch_size=batch_size,
    )
    guard.assert_clean()
    return {
        **result,
        "pred": pred,
        "val_report": classification_report_dict(y[outer.score], pred),
        "inner_best_epochs": best_epochs,
        "inner_mean_macro_f1": float(np.mean(best_scores)),
        "refit_epochs": refit_epochs,
        "selected": {
            "inner_median_epoch": float(np.median(best_epochs)),
            "refit_epochs": refit_epochs,
            "encoder_lr": encoder_lr,
            "scalar_lr": scalar_lr,
            "endpoint": endpoint,
            "depth": depth,
        },
        "update_ratios": parameter_update_ratios(before, model),
        "prediction_shift": float(np.mean(pred != pred_before)),
        "model": model,
        "audit": guard.summary(),
        "fold_seed": seed,
    }
