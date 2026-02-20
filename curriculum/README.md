# Curriculum

This curriculum is organized so earlier papers provide prerequisites for later ones.

## Module 1: Foundations & Architecture

1. Neural Machine Translation of Rare Words with Subword Units (2015)
   - Why here: introduces subword tokenization (BPE) and why tokenization matters.
   - Paper: https://arxiv.org/abs/1508.07909
2. Attention Is All You Need (2017)
   - Why here: introduces the Transformer architecture (self-attention + FFN blocks).
   - Paper: https://arxiv.org/abs/1706.03762

## Module 2: Interpretability (Inside the Black Box)

3. What Does BERT Look At? An Analysis of BERT's Attention (2019)
   - Why here: concrete visualizations and analysis of attention patterns.
   - Paper: https://arxiv.org/abs/1906.04341
4. Attention is not Explanation (2019)
   - Why here: why attention weights are not a faithful explanation by default.
   - Paper: https://arxiv.org/abs/1902.10186
5. Transformer Feed-Forward Layers Are Key-Value Memories (2020)
   - Why here: FFN layers store a surprising amount of "facts"; attention is not the whole story.
   - Paper: https://arxiv.org/abs/2012.14913

## Module 3: Generation & Decoding

6. The Curious Case of Neural Text Degeneration (2019)
   - Why here: decoding pathologies; introduces nucleus sampling (top-p).
   - Paper: https://arxiv.org/abs/1904.09751

## Module 4: The Data Foundation

7. Datasheets for Datasets (2018)
   - Why here: documentation, bias, accountability; a strong baseline for responsible data practice.
   - Paper: https://arxiv.org/abs/1803.09010

## Module 5: Efficiency & Scaling

8. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (2022)
   - Why here: makes attention faster and more memory-efficient with IO-aware tiling.
   - Paper: https://arxiv.org/abs/2205.14135
9. LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale (2022)
   - Why here: 8-bit quantization that enables large model inference with less memory.
   - Paper: https://arxiv.org/abs/2208.07339

## Module 6: Fine-Tuning & Alignment

10. LoRA: Low-Rank Adaptation of Large Language Models (2021)
   - Why here: parameter-efficient fine-tuning via low-rank adapters.
   - Paper: https://arxiv.org/abs/2106.09685
11. QLoRA: Efficient Finetuning of Quantized LLMs (2023)
   - Why here: combine quantization + LoRA to fine-tune bigger models on limited hardware.
   - Paper: https://arxiv.org/abs/2305.14314
12. The Flan Collection: Designing Data and Methods for Effective Instruction Tuning (2023)
   - Why here: instruction-tuning data design and methods.
   - Paper: https://arxiv.org/abs/2301.13688
13. LIMA: Less Is More for Alignment (2023)
   - Why here: shows that small, curated datasets can teach chat-style behavior.
   - Paper: https://arxiv.org/abs/2305.11206

## Supplemental Papers (Added, Not in 2026 Schedule Yet)

1. Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning (2021)
   - Why here: shows that fine-tuning often lives in a low-dimensional subspace, which motivates low-rank adaptation methods like LoRA.
   - Paper: https://arxiv.org/abs/2012.13255
2. Deduplicating Training Data Makes Language Models Better (2022)
   - Why here: demonstrates that data deduplication improves both efficiency and quality, reinforcing the importance of dataset curation.
   - Paper: https://arxiv.org/abs/2107.06499

