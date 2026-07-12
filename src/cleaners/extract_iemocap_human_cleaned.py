#!/usr/bin/env python3
"""
Authors:
    Steven Liem (steven.liem@sydney.edu.au)

Create a balanced 1,200-file human IEMOCAP subset (standard 6-class) for tone bias analysis. The full pipeline is as follows:
  1. Read manifest_iemocap_hf.csv and resolve local wav paths
  2. Stratified sample with seed=42: 400 happy, 400 neutral, 134 angry + 133 sad + 133 frustrated (400 negative)
  3. Copy wavs into corpora_cleaned/iemocap_human_cleaned/
  4. Rename as iemocap_human_{n}_{emotion}_{valence}.wav

Usage:
    uv run python extract_iemocap_human_cleaned.py

References:
    https://huggingface.co/datasets/ak0255/Synthesis_SER
"""

import csv
import random
import shutil
from pathlib import Path


WAVS = Path("/home/steve/tmp/Monash Research Code/corpora/IEMOCAP/wavs")
MANIFEST = Path(
    "/home/steve/tmp/Monash Research Code/"
    "monash-research-tone-bias/manifests/manifest_iemocap_hf.csv"
)
OUTPUT = Path(
    "/home/steve/tmp/Monash Research Code/corpora_cleaned/iemocap_human_cleaned"
)

# Standard IEMOCAP 6-class; 400/400/400 valence bins.
SAMPLE_COUNTS = {
    "happy": (400, "positive"),
    "neutral": (400, "neutral"),
    "angry": (134, "negative"),
    "sad": (133, "negative"),
    "frustrated": (133, "negative"),
}


def main() -> None:
    rows = {emotion: [] for emotion in SAMPLE_COUNTS}

    with MANIFEST.open() as file:
        for row in csv.DictReader(file):
            emotion = row["native_emotion_label"].lower()
            if emotion in rows:
                rows[emotion].append(WAVS / Path(row["audio_path"]).name)

    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)

    rng = random.Random(42)
    number = 1
    for emotion, (count, valence) in SAMPLE_COUNTS.items():
        for audio in rng.sample(rows[emotion], count):
            name = f"iemocap_human_{number}_{emotion}_{valence}.wav"
            shutil.copy2(audio, OUTPUT / name)
            number += 1
        print(f"{emotion}: copied {count}")

    print(f"Done: {number - 1} files written to {OUTPUT}")


if __name__ == "__main__":
    main()
