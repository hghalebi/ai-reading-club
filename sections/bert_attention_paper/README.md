# Rust BERT Attention Paper Tutorial

This section is a pedagogical Rust reimplementation of:

> What Does BERT Look At? An Analysis of BERT's Attention
> Kevin Clark, Urvashi Khandelwal, Omer Levy, Christopher D. Manning (2019)

The project is organised as a sequence of runnable steps. Each step compiles and runs on its own, and the final binary combines them into one paper-style walkthrough.

## What This Project Reimplements

- loading a pretrained BERT model and exposing all self-attention heads
- converting token-level attention maps to word-level attention maps
- surface analyses from section 3:
  - previous/self/next-token attention
  - attention to `[CLS]`, `[SEP]`, and punctuation
  - attention entropy
- head-level syntax probing from section 4.2
- head-level coreference probing from section 4.3
- attention-only and attention-plus-word probes from section 5
- Jensen-Shannon head distances plus 2D head clustering from section 6

## Important Scope Decision

The original paper used licensed Penn Treebank and CoNLL-2012 data. This tutorial ships with small open pedagogical datasets instead, so every step runs from a clean checkout. The code paths are the same style as the paper, but the default numbers are demo-scale rather than benchmark-scale.

## Run

```bash
cd sections/bert_attention_paper
cargo run --bin 01_extract_attention
cargo run --bin 02_surface_patterns
cargo run --bin 03_syntax_heads
cargo run --bin 04_coreference_heads
cargo run --bin 05_attention_probes
cargo run --bin 06_head_clustering
cargo run --bin 07_full_paper
```

All binaries default to `google-bert/bert-base-uncased`.
Model files are cached under `sections/bert_attention_paper/.cache/huggingface/` by default.
If you prefer a different cache location, set `HF_HOME` before running.

## Implementation Notes

- The Hugging Face checkpoint is loaded through a minimal Candle-based BERT implementation in `src/bert.rs`.
- Token-to-word aggregation matches the original analysis code's rule: sum attention into split target wordpieces, then average over split source wordpieces.
- The section-5 probe comparison keeps the paper's attention-based probes intact, but the distance-plus-word baseline is a compact linear tutorial baseline rather than the original larger neural classifier.

## Files

- `src/bert.rs`: minimal BERT forward pass with attention outputs
- `src/tokenization.rs`: tokenizer plus wordpiece alignment
- `src/attention.rs`: token-to-word aggregation and common math
- `src/analysis.rs`: paper analyses
- `src/probe.rs`: section-5 probes
- `data/`: bundled tutorial corpora and a tiny GloVe-style embedding file
- `src/bin/`: the staged executable walkthrough

## Verification

```bash
cargo fmt
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```
