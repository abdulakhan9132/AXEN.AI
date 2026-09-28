from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from axen.model.language_model import AXENLanguageModel


def evaluate_model(model: AXENLanguageModel, data_loader) -> Dict[str, float]:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for x, y in data_loader:
            logits = model(x)
            loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=0)
            total += float(loss.item())
            count += 1
    avg_loss = total / max(count, 1)
    return {"loss": avg_loss, "perplexity": float(torch.exp(torch.tensor(avg_loss)).item())}
