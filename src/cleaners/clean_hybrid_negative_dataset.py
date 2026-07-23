"""
Authors:
    Steven Liem (steven.liem@sydney.edu.au)

Convert the new hybrid_negative_voice_assistant_2000.jsonl to match Heet's dataset manifest for Gemini Live generation.

Uses only `human` (→ Question) and optional `category`. Existing JSONL emotion tags and AI responses are ignored. Emotion labels are experimental input-tone
conditions assigned with a seeded shuffle: 500 angry / sad / disgust / fearful.

Usage:
    uv run python src/cleaners/clean_hybrid_negative_dataset.py
    uv run python src/cleaners/clean_hybrid_negative_dataset.py --seed 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

WORKSPACE = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT = WORKSPACE / "hybrid_negative_voice_assistant_2000.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "heet_dataset_new_negative_clean.csv"

EMOTIONS = ("angry", "sad", "disgust", "fearful")
PER_EMOTION = 500
EXPECTED_N = len(EMOTIONS) * PER_EMOTION


def load_jsonl(path: Path) -> list[dict]:
    """Load and validate hybrid-negative JSONL records.

    Args:
        path: Path to the JSONL file.

    Returns:
        list[dict]: Parsed records with non-empty human questions.
    """
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no}: {e}") from e
            human = str(rec.get("human") or "").strip()
            if not human:
                raise ValueError(f"Missing human question on line {line_no}")
            rows.append(rec)
    if len(rows) != EXPECTED_N:
        raise ValueError(f"Expected {EXPECTED_N} records, got {len(rows)}")
    return rows


def build_manifest(records: list[dict], seed: int) -> pd.DataFrame:
    """Build a HEET-compatible CSV frame with balanced emotion assignment.

    Args:
        records: Validated JSONL records.
        seed: RNG seed for reproducible shuffle.

    Returns:
        pd.DataFrame: Manifest ready for Gemini generation.
    """
    df = pd.DataFrame(
        {
            "Question": [str(r["human"]).strip() for r in records],
            "category": [str(r.get("category") or "").strip() for r in records],
        }
    )
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    labels: list[str] = []
    for emotion in EMOTIONS:
        labels.extend([emotion] * PER_EMOTION)
    shuffled["emotion_label"] = labels
    shuffled["valence_label"] = "negative"
    shuffled["audio_path"] = ""
    shuffled["response_text"] = ""
    shuffled["ground_truth_label"] = ""

    return shuffled[
        [
            "Question",
            "emotion_label",
            "valence_label",
            "audio_path",
            "response_text",
            "ground_truth_label",
            "category",
        ]
    ]


def print_validation(df: pd.DataFrame) -> None:
    """Print row counts and emotion × category balance.

    Args:
        df: Manifest DataFrame.

    Returns:
        None
    """
    print(f"rows: {len(df)}")
    print(f"blank questions: {int(df['Question'].str.strip().eq('').sum())}")
    print("emotion_label counts:")
    print(df["emotion_label"].value_counts().reindex(list(EMOTIONS)).to_string())
    print(f"valence_label unique: {sorted(df['valence_label'].unique().tolist())}")
    for col in ("audio_path", "response_text", "ground_truth_label"):
        nonempty = int(df[col].fillna("").astype(str).str.strip().ne("").sum())
        print(f"non-empty {col}: {nonempty}")
    print("\nemotion × category:")
    print(pd.crosstab(df["emotion_label"], df["category"]).reindex(index=list(EMOTIONS)).to_string())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = load_jsonl(args.input)
    df = build_manifest(records, seed=args.seed)
    print_validation(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nWrote: {args.output}")


if __name__ == "__main__":
    main()
