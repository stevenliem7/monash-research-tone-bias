"""Frozen pooled-layer cache and lazy boundary-sequence caches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from typing import TYPE_CHECKING

import numpy as np
import torch

from fine_tuning.paths import CACHE_FORMAT_VERSION, WAVLM_MODEL_ID

if TYPE_CHECKING:
    import pandas as pd


def _meta_path(cache_dir: Path) -> Path:
    return cache_dir / "cache_meta.json"


def save_meta(cache_dir: Path, meta: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    _meta_path(cache_dir).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_meta(cache_dir: Path) -> dict | None:
    path = _meta_path(cache_dir)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def validate_meta(meta: dict, expected: dict) -> None:
    for key in (
        "cache_format_version",
        "model_id",
        "manifest_fingerprint",
        "hidden_size",
        "n_transformer_layers",
    ):
        if meta.get(key) != expected.get(key):
            raise ValueError(
                f"Stale cache for {key}: got {meta.get(key)!r}, expected {expected.get(key)!r}"
            )


@torch.inference_mode()
def extract_pooled_all_layers(
    processor,
    model,
    waves: list[np.ndarray],
    device: str,
) -> tuple[np.ndarray, list[int]]:
    """Return (n_layers+1, H) pooled stack and frame length for one clip.

    Index 0 is the post-projection / pre-transformer state; indices 1..24 are
    after each transformer layer for WavLM-large.
    """
    wav = waves[0]
    inputs = processor(
        wav,
        sampling_rate=16_000,
        return_tensors="pt",
        padding=False,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    inputs.pop("attention_mask", None)
    out = model(**inputs, output_hidden_states=True)
    stacks = []
    for h in out.hidden_states:
        stacks.append(h.squeeze(0).mean(dim=0).float().cpu().numpy())
    pooled = np.stack(stacks, axis=0)  # (25, H) for large
    n_frames = int(out.last_hidden_state.shape[1])
    return pooled.astype(np.float32), [n_frames]


@torch.inference_mode()
def extract_boundary_hidden(
    processor,
    model,
    wav: np.ndarray,
    device: str,
    boundary_layer_0idx: int,
) -> tuple[np.ndarray, int]:
    """Hidden state before transformer layer ``boundary_layer_0idx``.

    ``boundary_layer_0idx=0`` returns the post-projection encoder input.
    For active layers starting at s, pass ``boundary_layer_0idx=s``.
    """
    inputs = processor(
        wav,
        sampling_rate=16_000,
        return_tensors="pt",
        padding=False,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    inputs.pop("attention_mask", None)
    out = model(**inputs, output_hidden_states=True)
    # hidden_states[i] is input to transformer layer i (i==0: after pos/ln/drop).
    hidden = out.hidden_states[boundary_layer_0idx].squeeze(0)
    n_frames = int(hidden.shape[0])
    return hidden.half().cpu().numpy(), n_frames


def build_pooled_cache(
    *,
    clip_ids: list[str],
    pooled: np.ndarray,
    frame_lengths: np.ndarray,
    cache_dir: Path,
    manifest_fingerprint: str,
    model_id: str = WAVLM_MODEL_ID,
    model_revision: str | None = None,
    hidden_size: int = 1024,
    n_transformer_layers: int = 24,
) -> Path:
    """Write pooled_all_layers.pt and metadata."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "pooled_all_layers.pt"
    torch.save(
        {
            "clip_id": clip_ids,
            "pooled": torch.from_numpy(pooled),  # (N, 25, H)
            "frame_lengths": torch.from_numpy(frame_lengths.astype(np.int32)),
        },
        path,
    )
    save_meta(
        cache_dir,
        {
            "cache_format_version": CACHE_FORMAT_VERSION,
            "model_id": model_id,
            "model_revision": model_revision,
            "manifest_fingerprint": manifest_fingerprint,
            "hidden_size": hidden_size,
            "n_transformer_layers": n_transformer_layers,
            "pooled_path": str(path),
            "n_clips": len(clip_ids),
        },
    )
    return path


def load_pooled_cache(cache_dir: Path, expected: dict) -> dict[str, Any]:
    meta = load_meta(cache_dir)
    if meta is None:
        raise FileNotFoundError(f"No cache meta in {cache_dir}")
    validate_meta(meta, expected)
    path = cache_dir / "pooled_all_layers.pt"
    return torch.load(path, map_location="cpu", weights_only=False)


def boundary_cache_path(cache_dir: Path, active_start: int) -> Path:
    return cache_dir / f"boundary_before_layer_{active_start}.pt"


def save_boundary_cache(
    cache_dir: Path,
    active_start: int,
    clip_ids: list[str],
    hiddens: list[np.ndarray],
    lengths: list[int],
) -> Path:
    path = boundary_cache_path(cache_dir, active_start)
    torch.save(
        {
            "clip_id": clip_ids,
            "lengths": lengths,
            "hiddens": hiddens,
            "active_start": active_start,
        },
        path,
    )
    return path


def load_boundary_cache(cache_dir: Path, active_start: int) -> dict:
    path = boundary_cache_path(cache_dir, active_start)
    if not path.is_file():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=False)


def ensure_boundary_cache(
    *,
    cache_dir: Path,
    active_start: int,
    frame: "pd.DataFrame",
    processor,
    model,
    device: str,
    wav_loader,
) -> dict:
    """Build boundary hidden cache for ``active_start`` if missing."""
    path = boundary_cache_path(cache_dir, active_start)
    if path.is_file():
        return load_boundary_cache(cache_dir, active_start)

    hiddens: list[np.ndarray] = []
    lengths: list[int] = []
    clip_ids: list[str] = []
    for _, row in frame.iterrows():
        wav = wav_loader(Path(str(row["audio_path"])))
        hidden, n_frames = extract_boundary_hidden(
            processor,
            model,
            wav,
            device,
            boundary_layer_0idx=active_start,
        )
        hiddens.append(hidden)
        lengths.append(n_frames)
        clip_ids.append(str(row["clip_id"]))
    save_boundary_cache(cache_dir, active_start, clip_ids, hiddens, lengths)
    return load_boundary_cache(cache_dir, active_start)
