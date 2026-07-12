# Manifest loaders

Scripts that stream **metadata only** from Hugging Face (no full audio download) and write CSVs into `manifests/`.

| Script | HF source | Output |
|---|---|---|
| `load_deepdialogue_manifest.py` | [SALT-Research/DeepDialogue-xtts](https://huggingface.co/datasets/SALT-Research/DeepDialogue-xtts) | `manifests/manifest_deepdialogue_hf.csv` |
| `load_iemocap_human_manifest.py` | [AbstractTTS/IEMOCAP](https://huggingface.co/datasets/AbstractTTS/IEMOCAP) | `manifests/manifest_iemocap_hf.csv` |

```bash
uv run python load_deepdialogue_manifest.py --dry-run   # 200 rows, no CSV write
uv run python load_deepdialogue_manifest.py             # full extract

uv run python load_iemocap_human_manifest.py --dry-run # Similarly, only extract 200 rows, no CSV write
uv run python load_iemocap_human_manifest.py
```

These two manifests are consumed by the matching cleaners in `src/cleaners/` to build the 1,200-file subsets under `corpora_cleaned/`.

### Summary: Which cleaners need a manifest?

| Cleaner | Needs `manifests/` CSV? | Data source |
|---|---|---|
| `extract_deepdialogue_xtts_cleaned.py` | Yes, `manifest_deepdialogue_hf.csv` | HF paths via manifest, then selective download |
| `extract_iemocap_human_cleaned.py` | Yes, `manifest_iemocap_hf.csv` | Local IEMOCAP wavs, selected via manifest |
| `extract_emovoice_cleaned.py` | No | Local EmoVoice-DB JSONL + wavs |
| `extract_iemocap_synth_cleaned.py` | No | Local IEMOCAP_SYN CosyVoice2 JSON + wavs |
| `extract_tess_human_cleaned.py` | No | Local TESS wavs |
| `extract_tess_indextts_cleaned.py` | No | Local TESS_SYN IndexTTS JSON + wavs |
| `clean_heet_dataset.py` | No | Its own Heet CSV |
