#!/usr/bin/env python3
"""
Authors:
    Steven Liem (steven.liem@sydney.edu.au)

Create a balanced 1,200-file IndexTTS2 TESS subset for SER / tone-bias analysis.
Surprised is excluded as valence-ambiguous. The full pipeline is as follows:
  1. Load All_sample_indextts2.json and resolve local tts_path wavs
  2. Stratified sample with seed=42: 400 happy, 400 neutral, 100 x 4 negative
  3. Copy wavs into corpora_cleaned/tess_indextts_cleaned/
  4. Rename as tess_indextts_{n}_{emotion}_{valence}.wav

Usage:
    uv run python extract_tess_indextts_cleaned.py

References:
    None
"""

import json
import random
import shutil
from pathlib import Path


SOURCE = Path("/home/steve/tmp/Monash Research Code/corpora/TESS_SYN/indextts2")
OUTPUT = Path("/home/steve/tmp/Monash Research Code/corpora_cleaned/tess_indextts_cleaned")
SAMPLE_COUNTS = {
    "happy": (400, "positive"),
    "neutral": (400, "neutral"),
    "angry": (100, "negative"),
    "sad": (100, "negative"),
    "fearful": (100, "negative"),
    "disgust": (100, "negative"),
}


def main() -> None:
    with (SOURCE / "All_sample_indextts2.json").open() as file:
        manifest = json.load(file)

    rows = {emotion: [] for emotion in SAMPLE_COUNTS}
    for row in manifest:
        emotion = row["emotion"].lower()
        if emotion in rows:
            rows[emotion].append(
                SOURCE / "TESS-indextts2" / Path(row["tts_path"]).name
            )

    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)

    rng = random.Random(42)
    number = 1
    for emotion, (count, valence) in SAMPLE_COUNTS.items():
        for audio in rng.sample(rows[emotion], count):
            name = f"tess_indextts_{number}_{emotion}_{valence}.wav"
            shutil.copy2(audio, OUTPUT / name)
            number += 1
        print(f"{emotion}: copied {count}")

    print(f"Done: {number - 1} files written to {OUTPUT}")


if __name__ == "__main__":
    main()
