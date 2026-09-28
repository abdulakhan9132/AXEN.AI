from __future__ import annotations

import torch
from threading import Event

from axen.model.language_model import AXENLanguageModel
from axen.tokenizer.tokenizer import Tokenizer


def generate_text(model: AXENLanguageModel, tokenizer: Tokenizer, prompt: str, max_new_tokens: int = 24, temperature: float = 1.0, stop_event: Event | None = None) -> str:
    encoded = tokenizer.encode(prompt, add_special_tokens=True)
    vocab_size = model.transformer.token_embedding.num_embeddings
    invalid = [token for token in encoded if token < 0 or token >= vocab_size]
    if invalid:
        raise ValueError(
            "Tokenizer/model vocabulary mismatch: "
            f"token IDs {invalid[:3]} are outside vocabulary size {vocab_size}; "
            f"prompt={prompt!r}, tokenizer vocabulary size={tokenizer.vocab_size}."
        )
    context_length = model.transformer.positional_embedding.num_embeddings
    if len(encoded) > context_length:
        encoded = encoded[-context_length:]
    device = next(model.parameters()).device
    token_ids = torch.tensor([encoded], dtype=torch.long, device=device)
    output = model.generate(token_ids, max_new_tokens=max_new_tokens, temperature=temperature, stop_event=stop_event)
    decoded = tokenizer.decode(output[0].tolist())
    decoded = decoded.replace(tokenizer.bos_token, '').replace(tokenizer.eos_token, '')
    return decoded
