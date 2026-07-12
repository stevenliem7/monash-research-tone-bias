#!/usr/bin/env python3
"""
Authors:
    Steven Liem (steven.liem@sydney.edu.au)

Create a balanced 1,200-file human TESS subset for SER / tone-bias analysis.
Native TESS has 7 emotions (pleasant surprise excluded as valence-ambiguous).
The full pipeline is as follows:
  1. Glob local wavs by filename emotion suffix
  2. Stratified sample with seed=42: 400 happy, 400 neutral, 100 x 4 negative
  3. Copy wavs into corpora_cleaned/tess_human_cleaned/
  4. Rename as tess_human_{n}_{emotion}_{valence}.wav

Usage:
    uv run python extract_tess_human_cleaned.py

References:
    None
"""

import random
import shutil
from pathlib import Path


SOURCE = Path("/home/steve/tmp/Monash Research Code/corpora/TESS/data/tess/tess")
OUTPUT = Path("/home/steve/tmp/Monash Research Code/corpora_cleaned/tess_human_cleaned")
SAMPLE_COUNTS = {
    "happy": (400, "positive"),
    "neutral": (400, "neutral"),
    "angry": (100, "negative"),
    "sad": (100, "negative"),
    "fear": (100, "negative"),
    "disgust": (100, "negative"),
}


def main() -> None:
    rows = {
        emotion: list(SOURCE.glob(f"*_{emotion}.wav"))
        for emotion in SAMPLE_COUNTS
    }

    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)

    rng = random.Random(42)
    number = 1
    for emotion, (count, valence) in SAMPLE_COUNTS.items():
        for audio in rng.sample(rows[emotion], count):
            name = f"tess_human_{number}_{emotion}_{valence}.wav"
            shutil.copy2(audio, OUTPUT / name)
            number += 1
        print(f"{emotion}: copied {count}")

    print(f"Done: {number - 1} files written to {OUTPUT}")


if __name__ == "__main__":
    main()
