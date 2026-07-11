"""
Authors: 
    Steven Liem (steven.liem@sydney.edu.au)

Clean Heet's dataset to prepare for emotion classification and tone bias analysis. The full pipeline is as follows:
  1. Clean dirty text
  2. Randomly assign reproducible emotion labels (default seed=42)
  3. Stratified sample: 400 happy, 400 neutral, 100 × 4 negative = 1,200 total
  4. Add valence_label column (positive/negative/neutral)
  5. Output heet_dataset_clean.csv

Usage:
    python clean_heet_dataset.py
    python clean_heet_dataset.py --input heet_dataset.csv --output heet_dataset_clean.csv --seed 42

References:
    None
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

# EDIT the TARGET_COUNTS map to change the number of rows to extract from Heet's dataset for each emotion.

# For our current research, we are aiming for 400 of each valence bin to match it with other corpora.
# 400 positive (happy) + 400 neutral + 100 each * 4 negative = 1,200
TARGET_COUNTS = {
    "happy": 400,
    "neutral": 400,
    "angry": 100,
    "fearful": 100,
    "sad": 100,
    "disgust": 100,
}

# Valence mapping (Ekman's 6 emotions to 3 class bins)
VALENCE_MAP = {
    "happy": "positive",
    "neutral": "neutral",
    "angry": "negative",
    "fearful": "negative",
    "sad": "negative",
    "disgust": "negative",
}

# Ordered replacements — specific patterns first to avoid partial matches
TEXT_REPLACEMENTS = [
    # Corrupted encoding artifacts. For example, row 88 has "Äôs" instead of "'s".
    ("‚Äô", "'"),
    ("‚Äò", "'"),
    ("'", "'"),
    ("'", "'"),
    ("'", "'"),
    # Degree symbol corruption --> °
    ("¬∞", "°"),
    # Em dash corruption --> —
    ("¬†", "—"),
    # Bullet corruption
    ("‚Ä¢", "•"),
    # Non-breaking space --> regular space
    ("\xa0", " "),
    # Ellipsis corruption
    ("‚Ä¶", "..."),
    # En-dash corruption
    ("‚Äì", "–"),
    # Zero-width joiner corruption (emoji sequences)
    ("‚Äç", ""),
    # Left/right double quotes
    ("‚Äú", '"'),
    ("‚Äù", '"'),
    # Common latin-1 --> UTF-8 artifacts
    ("‚Ç¨", "€"),
    ("√©", "é"),
    ("√∂", "ö"),
    ("√º", "ú"),
    ("√°", "á"),
    ("√±", "ñ"),
    ("√Å", "Å"),
    ("√Ñ", "Ñ"),
]


def clean_text(text: str) -> str:
    """Replace known encoding artifacts and normalize whitespace.
    
    Args:
        text: string to clean

    Returns:
        str: cleaned text
    """
    if not isinstance(text, str):
        return text
    for pattern, replacement in TEXT_REPLACEMENTS:
        text = text.replace(pattern, replacement)
    return " ".join(text.split())


def assign_emotion_labels(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Randomly assign emotion labels with reproducible stratified sampling.

    Shuffles the full dataset, then takes N rows per emotion from the top.
    Remaining rows are discarded.

    Args:
        df: pd.DataFrame to assign emotion labels to
        seed: random seed for reproducibility. Default seed = 42

    Returns:
        pd.DataFrame: DataFrame with emotion_label and valence_label columns
    """
    total = sum(TARGET_COUNTS.values())
    if len(df) < total:
        raise ValueError(f"Need {total} rows, found {len(df)}")

    # Shuffle indices
    indices = list(df.index)
    random.Random(seed).shuffle(indices)
    result = df.loc[indices[:total]].copy().reset_index(drop=True)

    labels = []
    for emotion, count in TARGET_COUNTS.items():
        labels.extend([emotion] * count)
    result["emotion_label"] = labels
    result["valence_label"] = result["emotion_label"].map(VALENCE_MAP)

    print("\nEmotion distribution:", result["emotion_label"].value_counts().to_dict())
    print("Valence distribution:", result["valence_label"].value_counts().to_dict())

    return result



def main(
    input_path: str | Path = "heet_dataset.csv",
    output_path: str | Path = "heet_dataset_clean.csv",
    seed: int = 42,
) -> pd.DataFrame:
    """Full cleaning pipeline.
    
    Args:
        input_path: path to the input CSV file
        output_path: path to the output CSV file
        seed: random seed for reproducibility. Default seed = 42

    Returns:
        pd.DataFrame: DataFrame with cleaned text, emotion_label, and valence_label columns
    """

    input_path, output_path = Path(input_path), Path(output_path)
    print(f"Loading {input_path}")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows x {len(df.columns)} columns")

    print("\nCleaning dirty text...")
    for col in ("Question", "Answer"):
        before = df[col].copy()
        df[col] = df[col].map(clean_text)
        changed = before.fillna("").ne(df[col].fillna("")).sum()
        print(f"  {col}: {changed} rows cleaned")

    print(f"\nAssigning emotion labels (seed={seed})...")
    df = assign_emotion_labels(df, seed=seed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows x {len(df.columns)} columns to {output_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="heet_dataset.csv")
    parser.add_argument("--output", default="heet_dataset_clean.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(args.input, args.output, args.seed)
