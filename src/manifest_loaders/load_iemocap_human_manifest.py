"""
Authors:
    Steven Liem (steven.liem@sydney.edu.au)

Extract a metadata-only manifest from Hugging Face's IEMOCAP (human) corpus. Does not download the audio files. The full pipeline is as follows:
  1. Stream HF metadata (train/validation/test) with audio decode disabled
  2. Keep audio path, text, and major_emotion as native labels
  3. Map to 3-class valence; keep surprise/other with empty valence
  4. Write manifests/manifest_iemocap_hf.csv

Usage:
    python load_iemocap_human_manifest.py
    python load_iemocap_human_manifest.py --dry-run

References:
    https://huggingface.co/datasets/AbstractTTS/IEMOCAP
"""

import argparse
from pathlib import Path

import pandas as pd

REPO_ID = "AbstractTTS/IEMOCAP"
OUT = Path("manifests/manifest_iemocap_hf.csv")
DRY_RUN_ROWS = 200
SPLIT_NAMES = ("train", "validation", "test")

# Preserve all native labels; surprise/other get empty valence by design.
VALENCE_MAP = {
    "happy": "positive",
    "excited": "positive",
    "neutral": "neutral",
    "angry": "negative",
    "frustrated": "negative",
    "sad": "negative",
    "fear": "negative",
    "disgust": "negative",
    "surprise": None,
    "other": None,
}


def load_hf_iemocap(max_rows: int | None = None) -> pd.DataFrame:
    """Stream IEMOCAP metadata from Hugging Face (no waveform decode).

    Args:
        max_rows: Optional cap on extracted rows (used by --dry-run).

    Returns:
        pd.DataFrame: Raw rows with audio_path, text, native_emotion_label, split.
    """
    from datasets import Audio, load_dataset

    print(f"[iemocap] loading {REPO_ID} (streaming)")
    ds = load_dataset(REPO_ID, streaming=True)

    rows: list[dict] = []
    for split in SPLIT_NAMES:
        if split not in ds:
            continue
        dsplit = ds[split]
        if "audio" in dsplit.column_names:
            dsplit = dsplit.cast_column("audio", Audio(decode=False))

        for rec in dsplit:
            audio = rec.get("audio")
            audio_path = str(audio.get("path", "")) if isinstance(audio, dict) else str(audio or "")
            rows.append(
                {
                    "audio_path": audio_path,
                    "text": str(rec.get("text", "")).strip(),
                    "native_emotion_label": str(rec.get("major_emotion", "")).strip().lower(),
                    "split": split,
                }
            )
            if max_rows is not None and len(rows) >= max_rows:
                print(f"[iemocap] collected {len(rows)} rows")
                return pd.DataFrame(rows)

    print(f"[iemocap] collected {len(rows)} rows")
    return pd.DataFrame(rows)


def build_manifest(raw: pd.DataFrame) -> pd.DataFrame:
    """Map valence and attach source metadata.

    Args:
        raw: Raw HF metadata DataFrame from load_hf_iemocap.

    Returns:
        pd.DataFrame: Manifest columns ready to write to CSV.
    """
    df = raw.copy()
    df["valence_label"] = df["native_emotion_label"].map(VALENCE_MAP)
    df["source_hf_repo"] = REPO_ID
    return df[
        [
            "audio_path",
            "text",
            "native_emotion_label",
            "split",
            "valence_label",
            "source_hf_repo",
        ]
    ]


def main() -> None:
    """Run the IEMOCAP human manifest extraction pipeline.

    Returns:
        None
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help=f"Stream only {DRY_RUN_ROWS} rows")
    args = ap.parse_args()

    max_rows = DRY_RUN_ROWS if args.dry_run else None
    df = build_manifest(load_hf_iemocap(max_rows=max_rows))
    print(f"[iemocap] final rows: {len(df)}")
    print(f"[iemocap] valence: {df['valence_label'].value_counts(dropna=False).to_dict()}")

    if args.dry_run:
        print("[iemocap] dry-run complete (CSV not written)")
        return

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"[iemocap] wrote CSV: {OUT}")


if __name__ == "__main__":
    main()
