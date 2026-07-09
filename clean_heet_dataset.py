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
import re
from pathlib import Path

import pandas as pd

# Ordered replacements — specific patterns first to avoid partial matches
TEXT_REPLACEMENTS = [
    # Corrupted encoding artifacts. For example, row 88 has "Äôs" instead of "'s".
    ("‚Äô", "'"),
    ("‚Äò", "'"),
    ("'", "'"),
    ("'", "'"),
    ("'", "'"),
    # Degree symbol corruption → °
    ("¬∞", "°"),
    # Em dash corruption → —
    ("¬†", "—"),
    # Bullet corruption
    ("‚Ä¢", "•"),
    # Non-breaking space → regular space
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
    # Common latin-1 → UTF-8 artifacts
    ("‚Ç¨", "€"),
    ("√©", "é"),
    ("√∂", "ö"),
    ("√º", "ú"),
    ("√°", "á"),
    ("√±", "ñ"),
    ("√Å", "Å"),
    ("√Ñ", "Ñ"),
]

# Regex-based cleanup applied after replacements
REGEX_CLEANUPS = [
    (r"\s+", " "),           # Collapse whitespace
    (r"[ \t]+$", ""),        # Trailing whitespace
    (r"^[ \t]+", ""),        # Leading whitespace
]


def clean_text(text: str) -> str:
    """Apply all dirty text replacements and regex cleanups.
    
    Args:
        text: string to clean

    Returns:
        str: cleaned text
    """
    if not isinstance(text, str) or text.strip() == "":
        return text
    for pattern, replacement in TEXT_REPLACEMENTS:
        text = text.replace(pattern, replacement)
    for pattern, replacement in REGEX_CLEANUPS:
        text = re.sub(pattern, replacement, text)
    return text.strip()

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
    rng = random.Random(seed) 

    # Shuffle indices
    indices = list(df.index)
    rng.shuffle(indices)

    labeled = []
    offset = 0
    for emotion, count in TARGET_COUNTS.items():
        sample_indices = indices[offset : offset + count]
        offset += count
        if len(sample_indices) < count:
            print(f"  ⚠️  Not enough rows for emotion={emotion}: need {count}, got {len(sample_indices)}")
        for idx in sample_indices:
            labeled.append((idx, emotion, VALENCE_MAP[emotion]))

    # Build labeled DataFrame
    rows = []
    for idx, emotion, valence in labeled:
        row = df.loc[idx].copy()
        row["emotion_label"] = emotion
        row["valence_label"] = valence
        rows.append(row)

    result = pd.DataFrame(rows).reset_index(drop=True)

    # Verify counts
    print("\n  Emotion distribution:")
    for emotion, count in TARGET_COUNTS.items():
        actual = (result["emotion_label"] == emotion).sum()
        print(f"{emotion} : {actual}")

    print(f"\n  Valence distribution:")
    for valence in ["positive", "neutral", "negative"]:
        count = (result["valence_label"] == valence).sum()
        print(f"{valence} : {count}")

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

    input_path = Path(input_path)
    output_path = Path(output_path)

    # 1. Load the dataset
    print(f"Loading {input_path}")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows × {len(df.columns)} columns: {list(df.columns)}")

    # 2. Clean text
    print(f"\nCleaning dirty text...")
    for col in ["Question", "Answer"]:
        before = df[col].copy()
        df[col] = df[col].apply(clean_text)
        changed = (before != df[col]).sum()
        if changed > 0:
            print(f"  {col}: {changed} rows cleaned")
        else:
            print(f"  {col}: no changes needed")

    # 3. Assign emotion labels + stratified sample
    print(f"\nAssigning emotion labels (seed={seed})...")
    df = assign_emotion_labels(df, seed=seed)

    # 4. Save
    print(f"\nSaving to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows × {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean heet_dataset and assign emotion labels")
    parser.add_argument("--input", type=str, default="heet_dataset.csv", help="Input CSV path")
    parser.add_argument("--output", type=str, default="heet_dataset_clean.csv", help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    main(input_path=args.input, output_path=args.output, seed=args.seed)
