#!/usr/bin/env python3
"""
Authors:
    Steven Liem (steven.liem@sydney.edu.au)

Create a balanced 1,200-file EmoVoice-DB subset for SER / tone-bias analysis.
The full pipeline is as follows:
  1. Load train/val/test JSONL metadata and deduplicate by audio path
  2. Stratified sample with seed=42: 400 happy, 400 neutral, 100 x 4 negative
  3. Copy wavs into corpora_cleaned/emovoice_cleaned/
  4. Rename as emovoice_{n}_{emotion}_{valence}.wav

Usage:
    uv run python extract_emovoice_cleaned.py
    uv run python extract_emovoice_cleaned.py --source <path> --output <path> --seed 42

References:
    None
"""

import argparse
import json
import random
import shutil
from pathlib import Path


DEFAULT_SOURCE = Path("/home/steve/tmp/Monash Research Code/corpora/EmoVoice-DB")
DEFAULT_OUTPUT = Path(
    "/home/steve/tmp/Monash Research Code/corpora_cleaned/emovoice_cleaned"
)

SAMPLE_COUNTS = {
    "happy": (400, "positive"),
    "neutral": (400, "neutral"),
    "angry": (100, "negative"),
    "sad": (100, "negative"),
    "fearful": (100, "negative"),
    "disgusted": (100, "negative"),
}


def load_unique_rows(source: Path) -> dict[str, list[dict]]:
    """Load train/val/test metadata and deduplicate repeated audio paths.

    Args:
        source: Path to the local EmoVoice-DB root (JSONL + audio/).

    Returns:
        dict[str, list[dict]]: Emotion --> unique metadata rows for that emotion.
    """
    rows_by_emotion = {emotion: [] for emotion in SAMPLE_COUNTS}
    seen_paths: set[str] = set()

    for split in ("train", "val", "test"):
        with (source / f"{split}.jsonl").open(encoding="utf-8") as file:
            for line in file:
                row = json.loads(line)
                emotion = row["emotion"].strip().lower()
                audio_path = row["target_wav"]

                if emotion in rows_by_emotion and audio_path not in seen_paths:
                    rows_by_emotion[emotion].append(row)
                    seen_paths.add(audio_path)

    return rows_by_emotion


def extract_subset(source: Path, output: Path, seed: int) -> None:
    """Sample and copy a balanced EmoVoice subset to the output directory.

    Args:
        source: Path to the local EmoVoice-DB root.
        output: Destination directory for cleaned wav files.
        seed: Random seed for reproducible stratified sampling.

    Returns:
        None
    """
    rows_by_emotion = load_unique_rows(source)
    rng = random.Random(seed)
    output.mkdir(parents=True, exist_ok=True)

    number = 1
    for emotion, (count, valence) in SAMPLE_COUNTS.items():
        available = rows_by_emotion[emotion]
        if len(available) < count:
            raise ValueError(f"{emotion}: need {count} files, found {len(available)}")

        for row in rng.sample(available, count):
            source_audio = source / row["target_wav"]
            filename = f"emovoice_{number}_{emotion}_{valence}.wav"
            shutil.copy2(source_audio, output / filename)
            number += 1

        print(f"{emotion}: copied {count} files")

    print(f"Done: {number - 1} files written to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    extract_subset(args.source, args.output, args.seed)


if __name__ == "__main__":
    main()
