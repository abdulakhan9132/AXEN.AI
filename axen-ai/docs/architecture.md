# AXEN AI architecture

AXEN AI follows a compact educational pipeline:

1. Raw text is cleaned and split into train/validation files.
2. A tokenizer converts characters to integer IDs and back.
3. A dataset creates supervised next-token prediction windows.
4. A decoder-only transformer produces hidden states and logits.
5. Training minimizes cross-entropy against the next token.
6. Generation samples from logits autoregressively.

This design keeps the model understandable and runnable on CPU-first systems with optional CUDA acceleration.
