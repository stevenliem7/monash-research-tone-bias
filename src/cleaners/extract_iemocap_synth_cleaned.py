#!/usr/bin/env python3
"""
Authors:
    Steven Liem (steven.liem@sydney.edu.au)

Create a balanced 1,200-file CosyVoice2 IEMOCAP subset for SER / tone-bias
analysis. Upstream CosyVoice IEMOCAP only ships 4 emotions. The full pipeline is as follows:
  1. Load train.json / test.json and resolve local wav paths
  2. Stratified sample with seed=42: 400 happy, 400 neutral,
     200 angry + 200 sad (400 negative)
  3. Copy wavs into corpora_cleaned/iemocap_synth_cleaned/
  4. Rename as iemocap_synth_{n}_{emotion}_{valence}.wav

Usage:
    uv run python extract_iemocap_synth_cleaned.py

References:
    None
"""

import json
import random
import shutil
from pathlib import Path


SOURCE = Path("/home/steve/tmp/Monash Research Code/corpora/IEMOCAP_SYN/cosyvoice2")
OUTPUT = Path(
    "/home/steve/tmp/Monash Research Code/corpora_cleaned/iemocap_synth_cleaned"
)
SAMPLE_COUNTS = {
    "happy": (400, "positive"),
    "neutral": (400, "neutral"),
    "angry": (200, "negative"),
    "sad": (200, "negative"),
}


def main() -> None:
    rows = {emotion: [] for emotion in SAMPLE_COUNTS}

    for split in ("train", "test"):
        with (SOURCE / f"{split}.json").open() as file:
            for row in json.load(file):
                emotion = row["emotion"].lower()
                if emotion in rows:
                    audio = SOURCE / split / Path(row["wav_path"]).name
                    rows[emotion].append(audio)

    rng = random.Random(42)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    number = 1

    for emotion, (count, valence) in SAMPLE_COUNTS.items():
        for audio in rng.sample(rows[emotion], count):
            name = f"iemocap_synth_{number}_{emotion}_{valence}.wav"
            shutil.copy2(audio, OUTPUT / name)
            number += 1
        print(f"{emotion}: copied {count}")

    print(f"Done: {number - 1} files written to {OUTPUT}")


if __name__ == "__main__":
    main()
