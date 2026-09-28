# AXEN AI

AXEN AI is a small educational language model built from first principles.

This project demonstrates the full pipeline from raw text to token IDs, embeddings, causal attention, transformer blocks, and next-token prediction.

## Project layout
- `src/axen/` package with config, data, tokenizer, model, training, inference, evaluation, utils, and CLI modules.
- `configs/` YAML configs.
- `data/samples/` small example corpus.
- `tests/` verification suite.
- `docs/` architecture notes.

## Quick start
```
python main.py setup
python main.py prepare-data
python main.py test
python main.py desktop
python main.py data-preview
python main.py prepare-data --config configs/my_data_training.yaml
python main.py data-stats
```

This is intentionally small and educational. It is not meant to replace large pretrained models.

## Desktop chat

The desktop client uses Tkinter and communicates only with the local
`AXENChatEngine`. It does not use cloud APIs or generate placeholder replies.
If `checkpoints/overfit.pt` is absent, the window remains usable and reports
`MODEL NOT TRAINED`.

The same window can be launched with:

```text
python scripts/launch_desktop.py
```

Chat history is stored locally in `data/axen_chats.sqlite3`. A future icon can
be placed at `assets/axen.ico`; no icon is required for the development build.

## Local data pipeline

Put `.txt`, `.md`, `.json`, `.jsonl`, or `.csv` files in `data/raw/`.
`data-preview` scans metadata and small previews without training. 
`prepare-data` cleans records, removes duplicates, deterministically splits
90/5/5 train/validation/test records, fits the existing AXEN tokenizer only on
training records, writes encoded splits, and creates
`data/processed/manifest.json`. Files remain local and their contents are not
written to the manifest.
