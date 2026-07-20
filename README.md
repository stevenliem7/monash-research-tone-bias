# Monash Research: Tone Bias

Building on the paper "Bias Beneath the Tone: Empirical Characterisation of Tone Bias in LLM-Driven UX Systems" (UISE 2026, arXiv:2512.19950)

Speech emotion recognition (SER) evaluation on eight cleaned corpora, running inference on **emotion2vec** and **Empathic-Insight-Voice (EIV)**, and then fine-tuning the better model for our speech corpus. Our-speech human labels come from HEET (`heet_dataset_clean.csv`).

## Setup (uv)

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+.

```bash
uv sync                    # create .venv and install dependencies
cp .env.example .env       # optional: Gemini API key to run the Gemini pipeline.
uv run python src/inference/emotion2vec.py --dry-run # test the pipeline for the first 10 rows.
```

Run any script with `uv run python <path>`. Evaluators in `src/evaluators/` are run the same way (see below).

For GPU inference (EIV / emotion2vec), install a CUDA build of PyTorch separately if needed, e.g. `uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124`.

### Dependencies

Declared in `pyproject.toml`. Full stack used by this project:

| Package | Used for |
|---------|----------|
| `pandas`, `numpy`, `scipy`, `pyarrow` | Data handling, cleaners, metrics |
| `python-dotenv` | Load `GEMINI_API_KEY` from `.env` |
| `google-genai` | Gemini Live API (`src/gemini_response_pipeline.py`) |
| `matplotlib`, `seaborn` | Confusion-matrix plots |
| `scikit-learn` | Accuracy, F1, classification reports |
| `huggingface-hub` | Download EIV MLP weights |
| `transformers` | Whisper encoder (EIV) |
| `torch`, `torchaudio` | EIV inference; emotion2vec backend |
| `funasr` | emotion2vec_plus_large (`src/inference/emotion2vec.py`) |
| `librosa`, `soundfile` | Audio loading (EIV) |

`funasr` also pulls `modelscope` for model hosting. Cleaners and manifest loaders need only the pandas/numpy stack.

## Corpora

Eight cleaned sets under `corpora_cleaned/` (not in git):

| Corpus | Description |
|--------|-------------|
| `emovoice_cleaned` | EmoVoice-DB subset |
| `iemocap_human_cleaned` | Human IEMOCAP |
| `iemocap_synth_cleaned` | Synthetic IEMOCAP (CosyVoice2) |
| `tess_human_cleaned` | Human TESS |
| `tess_indextts_cleaned` | Synthetic TESS (IndexTTS) |
| `deepdialogue_xtts_cleaned` | DeepDialogue XTTS |
| `styletalk_cleaned` | StyleTalk |
| `our_speech_corpus_cleaned` | HEET / our-speech (filename valence + optional human GT) |

## Data files

| File | Purpose |
|------|---------|
| `heet_dataset.csv` | Raw HEET questions and metadata |
| `heet_dataset_clean.csv` | Cleaned HEET (1,200 rows; `ground_truth_label` for human valence) |
| `manifests/manifest_deepdialogue_hf.csv` | DeepDialogue metadata from Hugging Face |
| `manifests/manifest_iemocap_hf.csv` | IEMOCAP metadata from Hugging Face |
| `.env.example` | Template for `GEMINI_API_KEY` (copy to `.env`) |

## Project layout

```
monash-research-tone-bias/
├── README.md
├── heet_dataset.csv
├── heet_dataset_clean.csv
├── empathic_insight_voice.py          # EIV valence-head only (legacy local eval)
├── our_speech_emotion_variants_colab.py
├── manifests/
├── src/
│   ├── gemini_response_pipeline.py    # Generate our-speech audio via Gemini Live API
│   ├── cleaners/                      # Build corpora_cleaned/ subsets
│   ├── manifest_loaders/              # HF metadata → manifests/
│   ├── inference/                     # GPU inference (writes predictions + metrics)
│   └── evaluators/                    # Summary tables, diagrams, re-aggregation
└── corpora_cleaned/                   # Audio (local only)
```

Results are written to `../results/` (sibling of this repo).

## Inference

### emotion2vec

`src/inference/emotion2vec.py` — pretrained `emotion2vec_plus_large`, 3-way valence (pos/neu/neg).

```bash
uv run python src/inference/emotion2vec.py --dry-run
uv run python src/inference/emotion2vec.py
uv run python src/inference/emotion2vec.py --from-cache --workers 8
```

Output: `../results/emotion2vec/<corpus>/predictions.csv`, `metrics.json`

### Empathic-Insight-Voice (36 emotion heads)

`src/inference/empathic_insight_voice_all36_kaggle.py` — Whisper + Valence MLP + 36 EMONET emotion heads. Three valence methods:

| Method | Rule |
|--------|------|
| `valence_head` | Valence MLP score, binned with neutral band (default 0.5) |
| `top1` | Argmax emotion head → polarity map |
| `mass` | Softmax → count-normalized pos/neu/neg mass → argmax polarity |

```bash
uv run python src/inference/empathic_insight_voice_all36_kaggle.py --dry-run
uv run python src/inference/empathic_insight_voice_all36_kaggle.py
```

Output: `../results/empathic_insight_voice_all36/<corpus>/predictions.csv`, `metrics.json`

## Evaluators

Run after inference. All live in `src/evaluators/`.

| Script | Purpose |
|--------|---------|
| `emotion2vec_summary_table.py` | Valence summary CSV; our-speech prompted + human GT rows |
| `emotion2vec_all_emotions.py` | Full-emotion confusion matrices and metrics |
| `eiv_summary_table.py` | Per-method valence tables (`valence_head`, `top1`, `mass`) + `summary_comparison.csv` |
| `eiv_all_emotions.py` | EIV full-emotion confusion matrices |
| `human_gt_comparable.py` | Shared human-GT set (n=229) for cross-instrument comparison |
| `compare_heet_ground_truth.py` | Human GT vs emotion2vec confusion matrix PNG |

```bash
uv run python src/evaluators/emotion2vec_summary_table.py
uv run python src/evaluators/emotion2vec_all_emotions.py
uv run python src/evaluators/eiv_summary_table.py
uv run python src/evaluators/eiv_all_emotions.py
```

Our-speech summary rows:

- `*_prompted` — vs filename / synthesis target valence
- `*_human_GT` — vs HEET human labels on the shared n=229 set (human-labelled ∩ emotion2vec-mappable)

## Corpus builders

### Manifest loaders (`src/manifest_loaders/`)

| Script | Output |
|--------|--------|
| `load_deepdialogue_manifest.py` | `manifests/manifest_deepdialogue_hf.csv` |
| `load_iemocap_human_manifest.py` | `manifests/manifest_iemocap_hf.csv` |

See `src/manifest_loaders/README.md` for usage.

### Cleaners (`src/cleaners/`)

| Script | Corpus output |
|--------|---------------|
| `clean_heet_dataset.py` | `heet_dataset_clean.csv` |
| `extract_emovoice_cleaned.py` | `emovoice_cleaned` |
| `extract_iemocap_human_cleaned.py` | `iemocap_human_cleaned` |
| `extract_iemocap_synth_cleaned.py` | `iemocap_synth_cleaned` |
| `extract_tess_human_cleaned.py` | `tess_human_cleaned` |
| `extract_tess_indextts_cleaned.py` | `tess_indextts_cleaned` |
| `extract_deepdialogue_xtts_cleaned.py` | `deepdialogue_xtts_cleaned` |
| `extract_styletalk_cleaned.py` | `styletalk_cleaned` |

## Other scripts

| Script | Purpose |
|--------|---------|
| `empathic_insight_voice.py` | Early EIV valence-head-only local pipeline (metrics + plots in-script) |
| `our_speech_emotion_variants_colab.py` | Colab helper: compare HEET human GT vs EIV predictions |
| `src/gemini_response_pipeline.py` | Generate our-speech wavs from HEET via Gemini Live API (feeds `our_speech_corpus_cleaned`) |

## Results layout

```
../results/
├── emotion2vec/
│   ├── summary_metrics_table.csv
│   ├── <corpus>/predictions.csv, metrics.json
│   ├── all_emotions/
│   └── result_diagrams/
└── empathic_insight_voice_all36/
    ├── summary_metrics_table_{valence_head,top1,mass}.csv
    ├── summary_comparison.csv
    ├── metrics_all.json
    ├── <corpus>/predictions.csv, metrics.json
    ├── all_emotions/
    └── result_diagrams/
```

## References

- [emotion2vec_plus_large](https://huggingface.co/emotion2vec/emotion2vec_plus_large)
- [Empathic-Insight-Voice-Small](https://huggingface.co/laion/Empathic-Insight-Voice-Small)
- [EmoWhisper-AnS-Small](https://huggingface.co/mkrausio/EmoWhisper-AnS-Small-v0.1)
