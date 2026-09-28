import torch

from axen.data.dataset import TextDataset


def test_text_dataset_creates_x_y_pairs():
    dataset = TextDataset([1, 2, 3, 4, 5], context_length=3)

    x, y = dataset[0]

    assert x.shape == torch.Size([3])
    assert y.shape == torch.Size([3])
    assert x.tolist() == [1, 2, 3]
    assert y.tolist() == [2, 3, 4]
