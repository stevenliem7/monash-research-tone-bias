"""
Authors:
    Steven Liem (steven.liem@sydney.edu.au)

Aggregate Corpus A + Corpus B WAVs into a single folder: corpora_cleaned/our_speech_corpus_final/

Filenames are prefixed A_ / B_ because ~100 basenames collide across the two source directories.

Usage:
    uv run python src/our_speech_corpus_final/aggregate_our_speech_corpus_final.py
    uv run python src/our_speech_corpus_final/aggregate_our_speech_corpus_final.py --dry-run
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[3]
CORPORA_ROOT = WORKSPACE / "corpora_cleaned"

SOURCES = (
    ("A", CORPORA_ROOT / "our_speech_corpus_cleaned"),
    ("B", CORPORA_ROOT / "our_speech_corpus_new_negative_cleaned"),
)
DEST = CORPORA_ROOT / "our_speech_corpus_final"


def aggregate(dest: Path = DEST, dry_run: bool = False) -> int:
    """Copy both corpora into dest with A_/B_ filename prefixes.

    Args:
        dest: Output directory under corpora_cleaned/.
        dry_run: If True, only print planned copies.

    Returns:
        int: Number of files planned/copied.
    """
    if not dry_run:
        dest.mkdir(parents=True, exist_ok=True)

    n = 0
    for prefix, src in SOURCES:
        if not src.is_dir():
            raise FileNotFoundError(f"Missing source corpus: {src}")
        for wav in sorted(src.glob("*.wav")):
            out = dest / f"{prefix}_{wav.name}"
            n += 1
            if dry_run:
                if n <= 5:
                    print(f"[dry-run] {wav.name} -> {out.name}")
                continue
            shutil.copy2(wav, out)

    return n


def main() -> None:
    """CLI entrypoint: aggregate Corpus A/B WAVs into our_speech_corpus_final.

    Args:
        None. Command-line flags: ``--dest`` and ``--dry-run``.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description="Aggregate our_speech corpora into our_speech_corpus_final/"
    )
    parser.add_argument("--dest", type=Path, default=DEST)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    n = aggregate(dest=args.dest, dry_run=args.dry_run)
    action = "would copy" if args.dry_run else "copied"
    print(f"[{action}] {n} wavs -> {args.dest}")


if __name__ == "__main__":
    main()
