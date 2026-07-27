"""
Authors:
    Steven Liem (steven.liem@sydney.edu.au)

Kaggle runner for our_speech_corpus emotion2vec embedding extraction. Remaps local absolute audio_path values in the split manifest onto a Kaggle
corpora_cleaned root, then calls extract_emotion2vec_embeddings.py.

Extract and cache utterance-level emotion2vec embeddings for our_speech_corpus through Kaggle. Run the following command in a Kaggle notebook (GPU enabled):

    !pip -q install funasr modelscope

    !python /kaggle/input/datasets/stevenliiem/extract-emotion2vec-embeddings-final-corpus/extract_emotion2vec_embeddings_kaggle.py \
      --corpora-root /kaggle/input/datasets/stevenliiem/extract-emotion2vec-embeddings-final-corpus \
      --manifest /kaggle/input/datasets/stevenliiem/extract-emotion2vec-embeddings-final-corpus/our_speech_corpus_manifest_split_80_20.csv \
      --unified /kaggle/input/datasets/stevenliiem/extract-emotion2vec-embeddings-final-corpus/our_speech_corpus_unified.csv \
      --output-dir /kaggle/working/our_speech_corpus \
      --splits train test

Dry run test:

    !python .../extract_emotion2vec_embeddings_kaggle.py \
      --corpora-root /kaggle/input/datasets/stevenliiem/extract-emotion2vec-embeddings-final-corpus \
      --manifest /kaggle/input/datasets/stevenliiem/extract-emotion2vec-embeddings-final-corpus/our_speech_corpus_manifest_split_80_20.csv \
      --output-dir /kaggle/working/our_speech_corpus \
      --limit 20

Note: in a notebook cell use a single backslash for line breaks (not a doubled backslash).

--corpora-root may be either:
  - the dataset/parent dir that contains our_speech_corpus_final/, or
  - the our_speech_corpus_final/ directory itself (A_*.wav / B_*.wav inside)

References:
    https://huggingface.co/emotion2vec/emotion2vec_plus_large
"""


from __future__ import annotations

import importlib.util
from pathlib import Path

for _p in Path(__file__).resolve().parents:
    _loader = _p / "load_config.py"
    if _loader.is_file():
        _spec = importlib.util.spec_from_file_location("load_config", _loader)
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        _mod.bootstrap(__file__)
        break


import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

from config import bootstrap_imports, extend_our_speech_corpus_paths

bootstrap_imports(__file__)
extend_our_speech_corpus_paths()
from extract_emotion2vec_embeddings import extract_embeddings
from config import CORPUS_FINAL_SUBDIR


def resolve_final_audio_dir(corpora_root: Path, sample_clip_id: str | None = None) -> Path:
    """Resolve the directory that actually contains A_/B_ WAV files.

    Accepts either the parent of ``our_speech_corpus_final`` or that folder itself.
    If ``sample_clip_id`` is given, prefer a directory that contains that file.

    Args:
        corpora_root: User-supplied --corpora-root path.
        sample_clip_id: Optional clip_id used to pick the correct nested folder.

    Returns:
        Path: Directory containing ``{clip_id}.wav`` files.

    Raises:
        FileNotFoundError: If neither layout can be resolved.
    """
    candidates: list[Path] = [
        corpora_root / CORPUS_FINAL_SUBDIR,
        corpora_root,
        corpora_root / CORPUS_FINAL_SUBDIR / CORPUS_FINAL_SUBDIR,
    ]
    # Also search one level of subdirs for a folder that holds WAVs.
    if corpora_root.is_dir():
        for child in sorted(corpora_root.iterdir()):
            if child.is_dir():
                candidates.append(child)
                nested = child / CORPUS_FINAL_SUBDIR
                if nested.is_dir():
                    candidates.append(nested)

    seen: set[Path] = set()
    wav_dirs: list[Path] = []
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved in seen or not path.is_dir():
            continue
        seen.add(resolved)
        if any(path.glob("*.wav")):
            wav_dirs.append(path)

    if sample_clip_id:
        target = f"{sample_clip_id}.wav"
        for path in wav_dirs:
            if (path / target).is_file():
                return path

    if wav_dirs:
        # Prefer a dir that looks like the aggregated corpus (A_*.wav present).
        for path in wav_dirs:
            if any(path.glob("A_*.wav")):
                return path
        return wav_dirs[0]

    raise FileNotFoundError(
        f"Could not find WAV directories under {corpora_root}. "
        f"Checked: {[str(p) for p in candidates[:6]]}"
    )


def remap_audio_paths(
    df: pd.DataFrame,
    corpora_root: Path,
    *,
    strict: bool = True,
) -> pd.DataFrame:
    """Rewrite audio_path to ``{final_dir}/{clip_id}.wav`` (ignore local abs paths).

    The CSV may contain machine-local paths such as
    ``/home/steve/.../A_1_happy_positive.wav``. On Kaggle those are discarded;
    only ``clip_id`` is used to rebuild paths under --corpora-root.

    Args:
        df: Split or unified manifest with clip_id.
        corpora_root: Parent of ``our_speech_corpus_final`` or the folder itself.
        strict: If True, raise when any remapped path is missing. If False,
            keep rows with missing audio and only print a warning.

    Returns:
        pd.DataFrame: Copy with remapped audio_path values.

    Raises:
        FileNotFoundError: If ``strict`` and one or more WAVs are missing.
    """
    out = df.copy()
    sample_clip = str(out.iloc[0]["clip_id"]).strip() if len(out) else None
    final_dir = resolve_final_audio_dir(corpora_root, sample_clip_id=sample_clip)

    remapped: list[str] = []
    missing_ids: list[str] = []
    for _, row in out.iterrows():
        clip_id = str(row["clip_id"]).strip()
        candidate = final_dir / f"{clip_id}.wav"
        remapped.append(str(candidate))
        if not candidate.is_file():
            missing_ids.append(clip_id)

    out["audio_path"] = remapped
    print(
        f"[kaggle] remapped {len(out)} paths under {final_dir} "
        f"(missing={len(missing_ids)})"
    )

    if missing_ids and strict:
        on_disk = sorted(p.name for p in final_dir.glob("*.wav"))[:8]
        raise FileNotFoundError(
            "Audio remapping failed: expected files named {clip_id}.wav under "
            f"{final_dir}, but {len(missing_ids)}/{len(out)} were missing.\n"
            f"  example expected: {final_dir / (missing_ids[0] + '.wav')}\n"
            f"  example files on disk: {on_disk or '(none)'}\n"
            "Fix: upload the aggregated corpus (A_*.wav / B_*.wav) and set "
            "--corpora-root to the dataset root that contains "
            "our_speech_corpus_final/, or to that folder itself."
        )
    if missing_ids:
        print(
            f"[kaggle] warning: {len(missing_ids)} clips missing on disk "
            f"(e.g. {missing_ids[:5]}); continuing"
        )
    return out


def main() -> None:
    """CLI entrypoint for Kaggle emotion2vec embedding extraction.

    Args:
        None. Command-line flags: ``--corpora-root``, ``--manifest``, ``--unified``,
        ``--output-dir``, ``--device``, ``--limit``, and ``--splits``.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description="Kaggle wrapper for extract_emotion2vec_embeddings.py"
    )
    parser.add_argument(
        "--corpora-root",
        type=Path,
        required=True,
        help="Parent of our_speech_corpus_final/, or that folder itself",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to our_speech_corpus_manifest_split_80_20.csv",
    )
    parser.add_argument(
        "--unified",
        type=Path,
        default=None,
        help="Optional unified manifest for QC report refresh",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/kaggle/working/our_speech_corpus"),
        help="Directory for remapped CSVs, embeddings.npz, and QC JSON",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--splits",
        nargs="*",
        default=["train", "test"],
        help="Split values to include (default: train test)",
    )
    args = parser.parse_args()

    if not args.corpora_root.is_dir():
        raise FileNotFoundError(f"corpora-root not found: {args.corpora_root}")
    if not args.manifest.is_file():
        raise FileNotFoundError(f"manifest not found: {args.manifest}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Remap BEFORE loading the model so bad audio layouts fail immediately.
    split_df = remap_audio_paths(pd.read_csv(args.manifest), args.corpora_root)
    remapped_split = args.output_dir / "our_speech_corpus_manifest_split_80_20.csv"
    split_df.to_csv(remapped_split, index=False)

    remapped_unified = args.output_dir / "our_speech_corpus_unified.csv"
    if args.unified is not None and args.unified.is_file():
        unified_df = remap_audio_paths(pd.read_csv(args.unified), args.corpora_root)
        unified_df.to_csv(remapped_unified, index=False)
    elif remapped_unified.exists():
        pass
    else:
        shutil.copy(remapped_split, remapped_unified)
        print(f"[kaggle] no --unified provided; copied remapped split -> {remapped_unified}")

    embeddings_out = args.output_dir / "our_speech_corpus_embeddings.npz"
    qc_out = args.output_dir / "our_speech_corpus_qc_report.json"

    summary = extract_embeddings(
        manifest_path=remapped_split,
        output_path=embeddings_out,
        device=args.device,
        limit=args.limit,
        splits=tuple(args.splits) if args.splits else None,
        unified_path=remapped_unified,
        qc_report_path=qc_out,
    )

    print(f"[kaggle] embeddings: {embeddings_out}")
    print(f"[kaggle] qc report:  {qc_out}")


if __name__ == "__main__":
    main()
