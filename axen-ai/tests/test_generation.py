import torch

from axen.model.language_model import AXENLanguageModel


def test_language_model_generation_length():
    model = AXENLanguageModel(
        vocab_size=32,
        context_length=8,
        embedding_dim=16,
        num_layers=1,
        num_heads=4,
        ff_dim=32,
    )
    token_ids = torch.randint(0, 32, (1, 4))

    generated = model.generate(token_ids, max_new_tokens=3, temperature=1.0)

    assert generated.shape == (1, 7)
