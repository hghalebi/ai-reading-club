# Rust BPE Tokenizer Tutorial

This folder contains a Rust version of the beginner BPE notebook tutorial.

## Run

```bash
cd sections/tokenization/rust_bpe_tokenizer
cargo run
```

The binary prints:

- byte-level intuition on a short string
- training a custom BPE tokenizer (vocab size 1000)
- encode/decode round trip
- an optional GPT-2 file summary, including encoder size and merge count, if the model files exist
