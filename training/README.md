# AXEN AI Training & Knowledge Pipeline

This directory contains the pipeline to fine-tune the Qwen2.5-Coder model for AXEN, making it better at multiple programming languages.

## Workflow Overview

### Step 1: Prepare Dataset (Runs Locally)
Run the dataset preparation script locally on your machine. This script downloads required datasets from Hugging Face, formats them, balances the languages, and splits them into training and evaluation sets.
```bash
# Install dependencies
pip install datasets

# Run the preparation script
python training/prepare_dataset.py
```
- **Expected Time:** 2-5 minutes depending on download speed.
- **Expected Download Size:** ~500 MB (Hugging Face datasets).
- **Outputs:** `train.jsonl` (95%) and `eval.jsonl` (5%).

### Step 2: Fine-Tuning (Runs on Google Colab / Kaggle)
Since a 4GB GPU cannot handle fine-tuning, you must run this step in a cloud notebook with at least a T4 GPU (free on Colab or Kaggle).
1. Upload `training/train_qlora.ipynb`, `train.jsonl`, and `eval.jsonl` to Google Colab.
2. Run all cells in the notebook.
- **Expected Time:** 1 to 2 hours for 1 epoch on a T4 GPU.
- **Expected Download Size:** Model weights are downloaded directly to the Colab instance (~2-3 GB).
- **Outputs:** A folder named `axen-coder-gguf/` containing `unsloth.Q4_K_M.gguf`. Download this folder back to your local `training/` directory.

### Step 3: Import into Ollama (Runs Locally)
Once you have the `.gguf` file, use the provided `Modelfile` to register it locally in Ollama.
```bash
cd training
ollama create axen-coder -f Modelfile
```
- **Expected Time:** < 1 minute.
- **Outputs:** A new Ollama model named `axen-coder`.

### Step 4: Evaluate (Runs Locally)
You can compare the base model and your fine-tuned model across different languages.
```bash
python training/evaluate.py
```
- **Expected Time:** ~2-5 minutes depending on Ollama inference speed.

### Step 5: Knowledge RAG (Runs Locally)
To give AXEN access to your specific codebase and current documentation, index your files.
```bash
# Install dependencies
pip install chromadb langchain langchain-community sentence-transformers

# Index your files (run this from the root directory)
python knowledge/ingest.py
```
This stores the vector database in `knowledge/chroma_db`. AXEN will now automatically use this database when planning and answering code questions via the `search_knowledge` tool.
