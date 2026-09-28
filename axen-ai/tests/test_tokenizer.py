from axen.tokenizer.tokenizer import Tokenizer


def test_tokenizer_round_trip():
    tokenizer = Tokenizer()
    tokenizer.fit(["hello world"])

    encoded = tokenizer.encode("hello", add_special_tokens=True)
    decoded = tokenizer.decode(encoded)

    assert encoded[0] == tokenizer.vocab[tokenizer.bos_token]
    assert encoded[-1] == tokenizer.vocab[tokenizer.eos_token]
    assert decoded.startswith(tokenizer.bos_token)
    assert decoded.endswith(tokenizer.eos_token)
    assert "h" in tokenizer.vocab
