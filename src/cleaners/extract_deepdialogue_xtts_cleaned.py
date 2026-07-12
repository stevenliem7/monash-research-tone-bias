#!/usr/bin/env python3
"""
Authors:
    Steven Liem (steven.liem@sydney.edu.au)

Download a balanced 1,200-file DeepDialogue-XTTS subset for SER / tone-bias analysis. The full corpus is not downloaded as it is 189 GB. 
Only sampled segment wavs are fetched from Hugging Face and downloaded into the cleaned output directory.

Usage:
    uv run python extract_deepdialogue_xtts_cleaned.py

References:
    https://huggingface.co/datasets/SALT-Research/DeepDialogue-xtts
"""

import csv
import os
import random
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


REPO = "SALT-Research/DeepDialogue-xtts"
MANIFEST = Path(
    "/home/steve/tmp/Monash Research Code/"
    "monash-research-tone-bias/manifests/manifest_deepdialogue_hf.csv"
)
OUTPUT = Path(
    "/home/steve/tmp/Monash Research Code/corpora_cleaned/deepdialogue_xtts_cleaned"
)
SAMPLE_COUNTS = {
    "happy": (400, "positive"),
    "neutral": (400, "neutral"),
    "angry": (100, "negative"),
    "sad": (100, "negative"),
    "fearful": (100, "negative"),
    "disgust": (100, "negative"),
}


def main() -> None:
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    from huggingface_hub import hf_hub_download

    rows = {emotion: [] for emotion in SAMPLE_COUNTS}
    with MANIFEST.open() as file:
        for row in csv.DictReader(file):
            emotion = row["native_emotion_label"].lower()

            # For our research, we only consider the first turn of each dialogue which is the closest match to our
            # Gemini Live pipeline: one prompt --> one line assistant response
            if emotion in rows and int(row["turn_index"]) == 1:
                rows[emotion].append(row["segment_audio_path"])

    rng = random.Random(42)  # Change seed to get a different random sample.
    selected = []
    number = 1
    for emotion, (count, valence) in SAMPLE_COUNTS.items():
        for path in rng.sample(rows[emotion], count):
            name = f"deepdialogue_xtts_{number}_{emotion}_{valence}.wav"
            selected.append((path, name))
            number += 1

    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)


    def download(item: tuple[str, str]) -> None:
        """Download one HF segment wav and copy it into the cleaned output dir.

        Args:
            item: (repo-relative segment_audio_path, destination filename).

        Returns:
            None
        """
        path, name = item
        cached = hf_hub_download(
            repo_id=REPO,
            repo_type="dataset",
            filename=f"data/{path}",
        )
        shutil.copy2(cached, OUTPUT / name)

    # Download the selected segment wavs concurrently to save time.
    with ThreadPoolExecutor(max_workers=8) as pool:
        for done, _ in enumerate(pool.map(download, selected), 1):
            if done % 100 == 0:
                print(f"Downloaded {done}/1200")

    print(f"Done: 1200 files written to {OUTPUT}")


if __name__ == "__main__":
    main()
