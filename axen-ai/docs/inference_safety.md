# Inference safety

The first desktop generation failure (`index out of range in self`) came from
the autoregressive loop passing a sequence longer than the checkpoint's
`context_length` to the positional embedding. The model had 32 positional
entries, but generation kept appending tokens without limiting the input.

Inference now:

- validates every prompt token against the model vocabulary;
- reports tokenizer/model vocabulary mismatches instead of clamping IDs;
- truncates the oldest prompt tokens when the prompt exceeds context length;
- uses only the latest context window on every generation step;
- validates generated token IDs;
- stops when the EOS token is produced.

The checkpoint loader also verifies tokenizer IDs, token embedding size, and
language-model head size against the authoritative checkpoint configuration.
