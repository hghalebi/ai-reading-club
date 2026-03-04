# Rust BPE tokenizer tutorial

This folder contains a Rust version of the beginner BPE notebook tutorial.

## Run

```bash
cd rust_bpe_tokenizer
cargo run
```

The binary prints:
- byte-level intuition on a short string
- training a custom BPE tokenizer (vocab size 1000)
- encode/decode roundtrip
- optional GPT-2 files summary (encoder size + merge count) if model files exist
