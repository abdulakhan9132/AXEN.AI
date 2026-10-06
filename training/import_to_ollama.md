# Importing the model to Ollama

After downloading the `axen-coder-gguf` directory (which contains `unsloth.Q4_K_M.gguf`) from Google Colab/Kaggle, place it in the `training` directory next to the `Modelfile`.

Run the following command from the `training` directory to import the model into Ollama:

```bash
ollama create axen-coder -f Modelfile
```

To test it manually:
```bash
ollama run axen-coder
```
