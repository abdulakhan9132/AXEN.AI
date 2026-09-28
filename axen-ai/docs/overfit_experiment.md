# Tiny-corpus overfit experiment

This experiment validates the AXEN training path before increasing model size.

## Configuration

- Config: `configs/overfit.yaml`
- Context length: 32
- Embedding dimension: 32
- Layers / heads: 1 / 2
- Feed-forward dimension: 64
- Dropout: 0.0
- Batch size: 4
- Learning rate: 0.003
- Maximum steps: 250
- Seed: 123

## Dataset

The deterministic corpus is `data/samples/overfit_corpus.txt`:

```text
hello world
hello axen
axen is learning
the cat is small
the dog is running
```

The character tokenizer is fitted only on this corpus. The dataset creates
next-token windows, allowing the small model to memorize the examples.

## Results

- Initial loss: 4.4736
- Final loss: 0.2547
- Steps: 250
- Training time: 2.94 seconds
- Device: CPU (`detect_device()` selected CPU because CUDA was unavailable)
- GPU memory: 0 MiB
- Gradient behavior: gradients were required, non-zero, finite, and parameters changed after `optimizer.step()`
- Checkpoint: `checkpoints/overfit.pt`
- Generation example from the reloaded checkpoint: `hello axen\naxen i`

## Conclusion

This is a pipeline correctness milestone only. A decreasing loss and successful
checkpoint reload show that the tiny Transformer can learn the tiny corpus;
they do not demonstrate general language ability or intelligence.
