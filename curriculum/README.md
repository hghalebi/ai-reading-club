# Curriculum

This curriculum is ordered so each paper gives useful context for the papers that follow.

## Module 1: Foundations and Architecture

1. Neural Machine Translation of Rare Words with Subword Units (2015)
   - Why it belongs here: introduces subword tokenisation (BPE) and explains why tokenisation matters.
   - Paper: https://arxiv.org/abs/1508.07909
2. Attention Is All You Need (2017)
   - Why it belongs here: introduces the Transformer architecture, including self-attention and feed-forward blocks.
   - Paper: https://arxiv.org/abs/1706.03762

## Module 2: Interpretability (Inside the Black Box)

3. What Does BERT Look At? An Analysis of BERT's Attention (2019)
   - Why it belongs here: provides concrete visualisations and analyses of attention patterns.
   - Paper: https://arxiv.org/abs/1906.04341
4. Attention is not Explanation (2019)
   - Why it belongs here: explains why attention weights are not faithful explanations by default.
   - Paper: https://arxiv.org/abs/1902.10186
5. Transformer Feed-Forward Layers Are Key-Value Memories (2020)
   - Why it belongs here: shows that feed-forward layers store a surprising amount of factual information; attention is not the whole model.
   - Paper: https://arxiv.org/abs/2012.14913

## Module 3: Generation and Decoding

6. The Curious Case of Neural Text Degeneration (2019)
   - Why it belongs here: explains common decoding failures and introduces nucleus sampling (top-p).
   - Paper: https://arxiv.org/abs/1904.09751

## Module 4: The Data Foundation

7. Datasheets for Datasets (2018)
   - Why it belongs here: establishes documentation, bias, and accountability as core parts of responsible data practice.
   - Paper: https://arxiv.org/abs/1803.09010
8. Croissant: A Metadata Format for ML-Ready Datasets (2024)
   - Why it belongs here: follows Datasheets by turning dataset documentation into machine-readable, interoperable metadata that ML tools can consume directly.
   - Paper: https://arxiv.org/abs/2403.19546

## Module 5: Efficiency and Scaling

9. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (2022)
   - Why it belongs here: makes attention faster and more memory-efficient through IO-aware tiling.
   - Paper: https://arxiv.org/abs/2205.14135
10. LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale (2022)
   - Why it belongs here: introduces 8-bit quantisation for large-model inference with lower memory use.
   - Paper: https://arxiv.org/abs/2208.07339

## Module 6: Fine-Tuning and Alignment

11. LoRA: Low-Rank Adaptation of Large Language Models (2021)
   - Why it belongs here: introduces parameter-efficient fine-tuning through low-rank adapters.
   - Paper: https://arxiv.org/abs/2106.09685
12. QLoRA: Efficient Finetuning of Quantized LLMs (2023)
   - Why it belongs here: combines quantisation and LoRA to fine-tune larger models on limited hardware.
   - Paper: https://arxiv.org/abs/2305.14314
13. The Flan Collection: Designing Data and Methods for Effective Instruction Tuning (2023)
   - Why it belongs here: examines instruction-tuning data design and methodology.
   - Paper: https://arxiv.org/abs/2301.13688
14. LIMA: Less Is More for Alignment (2023)
   - Why it belongs here: shows that small, carefully curated datasets can teach chat-style behaviour.
   - Paper: https://arxiv.org/abs/2305.11206

## Supplemental Papers

1. Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning (2021)
   - Why it is useful: shows that fine-tuning often lives in a low-dimensional subspace, which motivates low-rank adaptation methods such as LoRA.
   - Paper: https://arxiv.org/abs/2012.13255
2. Deduplicating Training Data Makes Language Models Better (2022)
   - Why it is useful: demonstrates that data deduplication can improve both efficiency and quality, reinforcing the importance of dataset curation.
   - Paper: https://arxiv.org/abs/2107.06499
