import torch

from axen.model.attention import MultiHeadAttention


def test_attention_shape_and_mask():
    attn = MultiHeadAttention(embedding_dim=8, num_heads=2)
    x = torch.randn(2, 5, 8)

    y = attn(x)

    assert y.shape == x.shape
    assert attn._causal_mask(5, x.device).shape == (5, 5)
    assert torch.all(attn._causal_mask(5, x.device)[0, 1:] == True)
