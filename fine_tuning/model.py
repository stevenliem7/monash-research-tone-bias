"""Neural head, scalar mix, and WavLM truncated / partial-tuning wrappers."""

from __future__ import annotations

import inspect
import warnings
from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F

_MASK_WARNINGS_SILENCED = False


def silence_wavlm_attention_mask_warnings() -> None:
    """HF WavLM pairs a bool key_padding_mask with a float position bias.

    PyTorch deprecates that dtype mix and spams every forward; the behaviour is
    still correct for our 2D 1=valid mask contract.
    """
    global _MASK_WARNINGS_SILENCED
    if _MASK_WARNINGS_SILENCED:
        return
    warnings.filterwarnings(
        "ignore",
        message=r"Support for mismatched key_padding_mask and attn_mask is deprecated\. Use same type for both instead\.",
        category=UserWarning,
    )
    _MASK_WARNINGS_SILENCED = True


class ValenceHead(nn.Module):
    """Fixed 1024 -> 128 -> 3 MLP with LayerNorm and dropout."""

    def __init__(self, hidden_size: int = 1024, n_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ScalarMix(nn.Module):
    """Softmax-normalized trainable mix over layer vectors."""

    def __init__(self, n_layers: int):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(n_layers))

    def forward(self, layers: torch.Tensor) -> torch.Tensor:
        """Mix stacked layer features.

        Args:
            layers: (batch, n_layers, hidden) or (n_layers, hidden) for one row.
        """
        weights = F.softmax(self.logits, dim=-1)
        if layers.dim() == 2:
            return (weights.unsqueeze(-1) * layers).sum(dim=0)
        return (weights.view(1, -1, 1) * layers).sum(dim=1)

    def weight_vector(self) -> torch.Tensor:
        return F.softmax(self.logits.detach(), dim=-1)


def mask_aware_mean_pool(
    hidden: torch.Tensor,
    frame_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Mean-pool over valid frames. ``frame_mask`` is (B, T) bool/int."""
    if frame_mask is None:
        return hidden.mean(dim=1)
    mask = frame_mask.to(dtype=hidden.dtype).unsqueeze(-1)
    summed = (hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return summed / denom


def configure_wavlm_for_partial(model: nn.Module) -> None:
    """Disable LayerDrop and SpecAugment; assert both are off."""
    silence_wavlm_attention_mask_warnings()
    cfg = model.config
    cfg.layerdrop = 0.0
    cfg.apply_spec_augment = False
    assert float(cfg.layerdrop) == 0.0, "layerdrop must be 0.0"
    assert bool(cfg.apply_spec_augment) is False, "apply_spec_augment must be False"


def load_wavlm(model_id: str, device: str):
    """Load HF WavLM backbone + feature extractor with partial-FT safety flags."""
    from transformers import AutoFeatureExtractor, AutoModel

    processor = AutoFeatureExtractor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    configure_wavlm_for_partial(model)
    model.to(device)
    return processor, model


@dataclass
class FreezePlan:
    """Which transformer layers (0-indexed) are trainable."""

    active_start: int
    active_end: int  # exclusive
    retained_endpoint: int  # 1-indexed last kept transformer layer

    @property
    def active_depth(self) -> int:
        return self.active_end - self.active_start


@lru_cache(maxsize=32)
def _encoder_layer_accepts_index(layer_cls: type) -> bool:
    """Transformers >=5 uses WavLMEncoderLayerStableLayerNorm without ``index``."""
    return "index" in inspect.signature(layer_cls.forward).parameters


def call_wavlm_encoder_layer(
    layer: nn.Module,
    hidden: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None,
    position_bias: torch.Tensor | None,
    layer_idx: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Call a WavLM encoder layer across Transformers 4.x and 5.x.

    ``attention_mask`` must be ``(B, T)`` with 1=valid / 0=pad. WavLM turns that
    into ``key_padding_mask = attention_mask.ne(1)``. Do not pass a 4D additive
    mask; that is the wrong API for this model.
    """
    silence_wavlm_attention_mask_warnings()
    kwargs: dict = {
        "hidden_states": hidden,
        "attention_mask": attention_mask,
        "position_bias": position_bias,
        "output_attentions": False,
    }
    if _encoder_layer_accepts_index(type(layer)):
        kwargs["index"] = layer_idx
    layer_out = layer(**kwargs)
    return layer_out[0], layer_out[1]


def freeze_except_suffix(wavlm: nn.Module, plan: FreezePlan) -> list[str]:
    """Freeze everything; unfreeze encoder layers [active_start, active_end)."""
    for p in wavlm.parameters():
        p.requires_grad = False
    trainable: list[str] = []
    layers = wavlm.encoder.layers
    for i in range(plan.active_start, plan.active_end):
        for name, p in layers[i].named_parameters():
            p.requires_grad = True
            trainable.append(f"encoder.layers.{i}.{name}")
    return trainable


class FrozenPooledClassifier(nn.Module):
    """Classify from cached mean-pooled layer stack (no WavLM forward)."""

    def __init__(
        self,
        n_layers: int,
        hidden_size: int = 1024,
        n_classes: int = 3,
        learn_mix: bool = True,
    ):
        super().__init__()
        self.learn_mix = learn_mix
        self.mix = ScalarMix(n_layers)
        self.head = ValenceHead(hidden_size, n_classes)
        if not learn_mix:
            self.mix.logits.requires_grad_(False)

    def forward(self, layer_stack: torch.Tensor) -> torch.Tensor:
        """layer_stack: (B, L, H) pooled hidden states."""
        if self.learn_mix:
            pooled = self.mix(layer_stack)
        else:
            pooled = layer_stack.mean(dim=1)
        return self.head(pooled)


class SuffixMixClassifier(nn.Module):
    """Active WavLM block with live per-layer pools mixed into the valence head.

    Cached vectors may appear only for layers below ``active_start``. Layers
    inside the active block are pooled after each live forward, so the mix never
    sees stale copies of trainable layers.
    """

    def __init__(
        self,
        wavlm: nn.Module,
        *,
        active_start: int,
        active_end: int,
        n_frozen_lower: int = 0,
        hidden_size: int = 1024,
        n_classes: int = 3,
    ):
        super().__init__()
        self.wavlm = wavlm
        self.active_start = active_start
        self.active_end = active_end
        self.n_frozen_lower = n_frozen_lower
        self.n_total_layers = len(wavlm.encoder.layers)
        n_mix = n_frozen_lower + (active_end - active_start)
        device = next(wavlm.parameters()).device
        self.mix = ScalarMix(n_mix).to(device)
        self.head = ValenceHead(hidden_size, n_classes).to(device)
        freeze_except_suffix(
            wavlm,
            FreezePlan(
                active_start=active_start,
                active_end=active_end,
                retained_endpoint=active_end,
            ),
        )

    def _base_position_bias(self, batch_size: int, seq_len: int, device, dtype):
        attn0 = self.wavlm.encoder.layers[0].attention
        bias = attn0.compute_bias(seq_len, seq_len)
        bias = bias.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        bias = bias.view(batch_size * attn0.num_heads, seq_len, seq_len)
        return bias.to(device=device, dtype=dtype)

    def forward(
        self,
        boundary_hidden: torch.Tensor,
        frame_mask: torch.Tensor,
        frozen_lower: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Continue from cached boundary through active layers.

        Args:
            boundary_hidden: (B, T, H) states before ``active_start``.
            frame_mask: (B, T) with 1=valid, 0=pad.
            frozen_lower: optional (B, n_frozen_lower, H) pooled layers below
                the active block.
        """
        hidden = boundary_hidden
        bsz, seq_len, _ = hidden.shape
        # Zero pads for residual/value paths, then pass the binary frame mask so
        # padded keys are excluded via key_padding_mask.
        expand = frame_mask.unsqueeze(-1).repeat(1, 1, hidden.shape[2])
        hidden = hidden.clone()
        hidden = hidden.masked_fill(~expand.bool(), 0.0)

        position_bias = self._base_position_bias(
            bsz, seq_len, hidden.device, hidden.dtype
        )
        live: list[torch.Tensor] = []
        for i in range(self.active_start, self.active_end):
            layer = self.wavlm.encoder.layers[i]
            hidden, position_bias = call_wavlm_encoder_layer(
                layer,
                hidden,
                attention_mask=frame_mask,
                position_bias=position_bias,
                layer_idx=i,
            )
            state = hidden
            if i == self.n_total_layers - 1 and hasattr(self.wavlm.encoder, "layer_norm"):
                state = self.wavlm.encoder.layer_norm(state)
            live.append(mask_aware_mean_pool(state, frame_mask))

        stack = torch.stack(live, dim=1)
        if frozen_lower is not None and self.n_frozen_lower:
            lower = frozen_lower.to(device=stack.device, dtype=stack.dtype)
            if lower.shape[1] != self.n_frozen_lower:
                raise ValueError(
                    f"frozen_lower has {lower.shape[1]} layers, "
                    f"expected {self.n_frozen_lower}"
                )
            stack = torch.cat([lower, stack], dim=1)
        return self.head(self.mix(stack))


# Backwards-compatible alias used by older call sites / preflight.
class PartialWavLMClassifier(SuffixMixClassifier):
    """Alias wrapping FreezePlan construction for older callers."""

    def __init__(
        self,
        wavlm: nn.Module,
        plan: FreezePlan,
        n_mix_layers: int,
        hidden_size: int = 1024,
        n_classes: int = 3,
    ):
        n_frozen = max(0, n_mix_layers - plan.active_depth)
        super().__init__(
            wavlm,
            active_start=plan.active_start,
            active_end=plan.active_end,
            n_frozen_lower=n_frozen,
            hidden_size=hidden_size,
            n_classes=n_classes,
        )
        self.plan = plan

    def forward_from_boundary(
        self,
        boundary_hidden: torch.Tensor,
        frame_mask: torch.Tensor,
        lower_pooled: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward(boundary_hidden, frame_mask, frozen_lower=lower_pooled)


def parameter_update_ratios(
    named_before: dict[str, torch.Tensor],
    module: nn.Module,
) -> dict[str, float]:
    """Compute ||Δθ|| / ||θ|| for each parameter that had requires_grad."""
    ratios: dict[str, float] = {}
    for name, p in module.named_parameters():
        if name not in named_before:
            continue
        before = named_before[name]
        delta = (p.detach().cpu() - before).float()
        denom = before.float().norm().clamp(min=1e-12)
        ratios[name] = float(delta.norm() / denom)
    return ratios


def snapshot_trainable(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: p.detach().cpu().clone()
        for name, p in module.named_parameters()
        if p.requires_grad
    }


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and Torch before model construction."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
