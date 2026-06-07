# Session Notes

- Paper: The Curious Case of Neural Text Degeneration (2019)
- Link: https://arxiv.org/abs/1904.09751
- Date: 2026-06-04 (Thursday)
- Venue: Schoolab
- Material: https://rust-ml.com/materials/neural-text-degeneration.html
- Discussion lead:
- Attendees:

## 10-15 Minute Overview

- Problem: high-likelihood decoding can still produce repetitive, low-quality open-ended text.
- Key idea: decoding is where probability becomes product behavior.
- Method: compare greedy decoding, beam search, sampling, top-k sampling, and nucleus/top-p sampling.
- Results: beam search can degenerate for open-ended generation, while truncation-based sampling can better preserve diversity and coherence.
- What changed after this paper: decoding became a product and systems concern, not just an implementation detail hidden behind an API parameter.

## Discussion

1. Why does beam search work for some sequence tasks but fail for open-ended text generation?
2. What does top-p control that temperature alone does not?
3. Which decoding settings should product builders understand before exposing LLM output to users?

## Actions

- [ ]
