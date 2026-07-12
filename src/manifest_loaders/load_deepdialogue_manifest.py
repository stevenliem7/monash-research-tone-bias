#!/usr/bin/env python3
"""
Authors:
    Steven Liem (steven.liem@sydney.edu.au)

Extract a metadata-only manifest from Hugging Face DeepDialogue-xtts (243k rows). Does not download the 189 GB of audio of the actual corpus. The full pipeline is as follows:
  1. Stream HF metadata (default/train) and keep path/text/label columns
  2. Prefer audio_ravdess_emotion for SER; keep dialogue emotion separately
  3. Map to 3-class valence; drop surprised/shocked and rows with no audio path
  4. Write manifests/manifest_deepdialogue_hf.csv

Usage:
    uv run python load_deepdialogue_manifest.py
    uv run python load_deepdialogue_manifest.py --dry-run

References:
    https://huggingface.co/datasets/SALT-Research/DeepDialogue-xtts
"""

import argparse
from pathlib import Path

import pandas as pd

REPO_ID = "SALT-Research/DeepDialogue-xtts"
CONFIG = "default"
SPLIT = "train"
OUT = Path("manifests/manifest_deepdialogue_hf.csv") # Path to the output manifest file.
DRY_RUN_ROWS = 200 # Number of rows to extract for dry run.

KEEP_COLUMNS = (
    "conversation_id",
    "domain",
    "turn_index",
    "speaker",
    "text",
    "emotion",
    "segment_audio_path",
    "audio_ravdess_emotion",
    "audio_dialogue_emotion",
)

# RAVDESS-style audio labels (+ calm). Surprised excluded as valence-ambiguous.
VALENCE_MAP = {
    "happy": "positive",
    "neutral": "neutral",
    "calm": "neutral",
    "sad": "negative",
    "angry": "negative",
    "fearful": "negative",
    "disgusted": "negative",
    "surprised": None,
}


def load_hf_deepdialogue(max_rows: int | None = None) -> pd.DataFrame:
    """Stream DeepDialogue-xtts metadata from Hugging Face.

    Args:
        max_rows: Optional cap on number of rows to extract (used by --dry-run).

    Returns:
        pd.DataFrame: Raw metadata rows with KEEP_COLUMNS only.
    """
    from datasets import load_dataset

    print(f"[deepdialogue] loading {REPO_ID} (streaming)")
    ds = load_dataset(REPO_ID, name=CONFIG, split=SPLIT, streaming=True)
    keep = [c for c in KEEP_COLUMNS if c in ds.column_names]
    ds = ds.remove_columns([c for c in ds.column_names if c not in keep])

    rows: list[dict] = []
    for i, rec in enumerate(ds):
        rows.append({c: rec.get(c) for c in keep})
        if (i + 1) % 25000 == 0:
            print(f"[deepdialogue] streamed {i + 1} rows...")
        if max_rows is not None and len(rows) >= max_rows:
            break

    print(f"[deepdialogue] collected {len(rows)} rows")
    return pd.DataFrame(rows)


def build_manifest(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize labels, map valence, and drop unusable rows.

    Args:
        raw: Raw HF metadata DataFrame from load_hf_deepdialogue.

    Returns:
        pd.DataFrame: Manifest columns ready to write to CSV.
    """
    df = raw.copy()
    df["split"] = SPLIT
    df["source_hf_repo"] = REPO_ID
    df["dialogue_emotion"] = df["emotion"].str.strip().str.lower()
    df["native_emotion_label"] = df["audio_ravdess_emotion"].str.strip().str.lower()

    missing = df["native_emotion_label"].isna()
    if missing.any():
        df.loc[missing, "native_emotion_label"] = df.loc[missing, "dialogue_emotion"]

    path = df["segment_audio_path"].astype(str).str.strip().str.lower()
    df = df[df["segment_audio_path"].notna() & (path != "") & (path != "null")].reset_index(drop=True)

    df["valence_label"] = df["native_emotion_label"].map(VALENCE_MAP)
    df = df.dropna(subset=["valence_label"]).reset_index(drop=True)

    return df[
        [
            "segment_audio_path",
            "text",
            "native_emotion_label",
            "dialogue_emotion",
            "valence_label",
            "conversation_id",
            "domain",
            "speaker",
            "turn_index",
            "split",
            "source_hf_repo",
        ]
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help=f"Stream only {DRY_RUN_ROWS} rows")
    args = ap.parse_args()

    max_rows = DRY_RUN_ROWS if args.dry_run else None
    df = build_manifest(load_hf_deepdialogue(max_rows=max_rows))
    print(f"[deepdialogue] final rows: {len(df)}")
    print(f"[deepdialogue] valence: {df['valence_label'].value_counts().to_dict()}")

    if args.dry_run:
        print("[deepdialogue] dry-run complete (CSV not written)")
        return

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"[deepdialogue] wrote CSV: {OUT}")


if __name__ == "__main__":
    main()
