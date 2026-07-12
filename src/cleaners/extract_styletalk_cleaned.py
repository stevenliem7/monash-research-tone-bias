#!/usr/bin/env python3
"""
Authors:
    Steven Liem (steven.liem@sydney.edu.au)

Create a StyleTalk subset for SER / tone-bias analysis using only the core emotion labels from the paper: neutral, cheerful, sad 
(friendly/unfriendly excluded). Sad is scarce (~202 unique clips), so the subset is 1,000 files.
The full pipeline is as follows:
  1. Load train/eval CSVs and collect curr + res audio clips by emotion
  2. Stratified sample with seed=42: 400 cheerful, 400 neutral, 200 sad
  3. Copy wavs into corpora_cleaned/styletalk_cleaned/
  4. Rename as styletalk_{n}_{emotion}_{valence}.wav

Usage:
    uv run python extract_styletalk_cleaned.py

References:
    https://github.com/DanielLin94144/StyleTalk
"""

import csv
import random
import shutil
from pathlib import Path


SOURCE = Path("/home/steve/tmp/Monash Research Code/corpora/StyleTalk")
OUTPUT = Path("/home/steve/tmp/Monash Research Code/corpora_cleaned/styletalk_cleaned")
CSV_FILES = ("train (1).csv", "eval.csv")
SAMPLE_COUNTS = {
    "cheerful": (400, "positive"),
    "neutral": (400, "neutral"),
    "sad": (200, "negative"),
}


def main() -> None:
    rows = {emotion: [] for emotion in SAMPLE_COUNTS}
    seen: set[str] = set()

    for name in CSV_FILES:
        with (SOURCE / name).open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                for audio_key, emotion_key in ( ("curr_audio_id", "curr_emotion"), ("res_audio_id", "res_emotion"), ):
                    emotion = row[emotion_key].strip().lower()
                    audio_id = row[audio_key].strip()
                    if emotion in rows and audio_id not in seen:
                        rows[emotion].append(SOURCE / "audio" / audio_id)
                        seen.add(audio_id)

    rng = random.Random(42)
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)
    number = 1

    for emotion, (count, valence) in SAMPLE_COUNTS.items():
        for audio in rng.sample(rows[emotion], count):
            name = f"styletalk_{number}_{emotion}_{valence}.wav"
            shutil.copy2(audio, OUTPUT / name)
            number += 1
        print(f"{emotion}: copied {count}")

    print(f"Done: {number - 1} files written to {OUTPUT}")


if __name__ == "__main__":
    main()
