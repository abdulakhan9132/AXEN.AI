import torch

from axen.model.transformer import Transformer


def test_transformer_forward_shape():
    model = Transformer(
        vocab_size=32,
        context_length=8,
        embedding_dim=16,
        num_layers=2,
        num_heads=4,
        ff_dim=32,
    )
    token_ids = torch.randint(0, 32, (2, 8))

    hidden = model(token_ids)

    assert hidden.shape == (2, 8, 16)
