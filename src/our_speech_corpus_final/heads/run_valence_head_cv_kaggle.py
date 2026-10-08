#!/usr/bin/env python3
"""
Authors:
    Steven Liem (steven.liem@sydney.edu.au)

Kaggle: re-run the frozen trained-head grid under question-grouped CV.

Why this exists
---------------
The early emotion2vec+ / WavLM / wav2vec2 head bake-off used
RepeatedStratifiedKFold (random folds). The WavLM headline (0.589), text
baselines, Gate 0, and nested fine-tuning all use StratifiedGroupKFold on
``question``. This script re-runs the same nine-head grid with
``--group-by question`` so those encoders can be compared on the same
protocol.

Embeddings for emotion2vec+ are already cached; pass that NPZ. WavLM-large
NPZ is often missing from the Kaggle input — extract it with
``run_ssl_backbone_kaggle.py`` (do not point this script at a missing NPZ).

Run both arms on a **GPU** notebook: classical heads ignore the device, but
``focal_mlp`` uses CUDA when torch sees a GPU, and WavLM extract needs one.

Kaggle (GPU notebook):

    !pip -q install scikit-learn pandas numpy transformers soundfile scipy
    !pip -q install torch  # Kaggle GPU image; lets focal_mlp use CUDA

    # emotion2vec+ large — question-grouped CV on cached NPZ
    !python /kaggle/input/.../run_valence_head_cv_kaggle.py \
      --embeddings /kaggle/input/.../our_speech_corpus_embeddings.npz \
      --manifest /kaggle/input/.../our_speech_corpus_manifest_split_80_20.csv \
      --output-dir /kaggle/working/e2v_group_question \
      --encoder-tag emotion2vec_plus_large \
      --group-by question

    # Optional WavLM-large — extract NPZ then same question-grouped CV
    !python /kaggle/input/.../run_ssl_backbone_kaggle.py \
      --model wavlm-large \
      --corpora-root /kaggle/input/.../our_speech_corpus_final \
      --manifest /kaggle/input/.../our_speech_corpus_manifest_split_80_20.csv \
      --output-dir /kaggle/working/ssl_wavlm \
      --run-cv --group-by question
    # Writes:
    #   /kaggle/working/ssl_wavlm/our_speech_corpus_microsoft_wavlm-large.npz
    #   /kaggle/working/ssl_wavlm/our_speech_corpus_valence_head_cv_microsoft_wavlm-large_group_question.json

Smoke test:

    !python .../run_valence_head_cv_kaggle.py \
      --embeddings ... --manifest ... --output-dir /kaggle/working/smoke \
      --encoder-tag smoke --group-by question --folds 2 --repeats 1
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
import json
from pathlib import Path

from config import SPLIT_SEED, bootstrap_imports, extend_our_speech_corpus_paths

bootstrap_imports(__file__)
extend_our_speech_corpus_paths()

from train_valence_head_cv import evaluate_all  # noqa: E402


def main() -> None:
    """Run the matched trained-head grid under group-aware CV on Kaggle.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description=(
            "Kaggle: frozen trained-head CV with StratifiedGroupKFold "
            "(default group_by=question)"
        )
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        required=True,
        help="Cached utterance embeddings NPZ (clip_id + embeddings)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="our_speech_corpus_manifest_split_80_20.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/kaggle/working/valence_head_cv_group_question"),
        help="Directory for the CV JSON",
    )
    parser.add_argument(
        "--encoder-tag",
        type=str,
        default="encoder",
        help="Short name baked into the output filename",
    )
    parser.add_argument(
        "--group-by",
        choices=("question", "response_text"),
        default="question",
        help="Grouping column (default: question, matches WavLM headline)",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Optional subset of head names, e.g. --only rbf_svc shrinkage_lda",
    )
    args = parser.parse_args()

    if not args.embeddings.is_file():
        raise FileNotFoundError(f"embeddings not found: {args.embeddings}")
    if not args.manifest.is_file():
        raise FileNotFoundError(f"manifest not found: {args.manifest}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = (
        args.output_dir
        / f"valence_head_cv_{args.encoder_tag}_group_{args.group_by}.json"
    )

    print(
        f"[kaggle] encoder={args.encoder_tag}  group_by={args.group_by}  "
        f"folds={args.folds}  repeats={args.repeats}  seed={args.seed}"
    )
    print(f"[kaggle] embeddings={args.embeddings}")
    print(f"[kaggle] manifest={args.manifest}")

    results = evaluate_all(
        embeddings_path=args.embeddings,
        manifest_path=args.manifest,
        output_path=out,
        folds=args.folds,
        repeats=args.repeats,
        seed=args.seed,
        acoustic=False,
        only=tuple(args.only) if args.only else None,
        group_by=args.group_by,
    )

    # Print a compact headline so the notebook cell shows the answer.
    balanced = results.get("grid", {}).get("balanced", {})
    best_name = None
    best_mac = -1.0
    for name, cell in balanced.items():
        mac = cell.get("macro_f1", {})
        mean = mac.get("mean") if isinstance(mac, dict) else None
        if mean is not None and mean > best_mac:
            best_mac = float(mean)
            best_name = name
    print(f"[kaggle] wrote {out}")
    if best_name is not None:
        print(
            f"[kaggle] best balanced macro-F1 under group_by={args.group_by}: "
            f"{best_name} = {best_mac:.4f}"
        )
    # Keep a tiny sidecar for quick download without opening the full grid.
    summary_path = args.output_dir / f"summary_{args.encoder_tag}_group_{args.group_by}.json"
    summary = {
        "encoder_tag": args.encoder_tag,
        "group_by": args.group_by,
        "embeddings": str(args.embeddings),
        "manifest": str(args.manifest),
        "cv_json": str(out),
        "best_balanced_head": best_name,
        "best_balanced_macro_f1": best_mac if best_name is not None else None,
        "protocol": results.get("protocol"),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[kaggle] summary {summary_path}")


if __name__ == "__main__":
    main()
