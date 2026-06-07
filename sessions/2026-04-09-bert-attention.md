# Session Notes

- Paper: What Does BERT Look At? An Analysis of BERT's Attention (2019)
- Link: https://arxiv.org/abs/1906.04341
- Date: 2026-04-09
- Discussion lead:
- Attendees:

## 10-15 Minute Overview

- Problem: attention is often visualized, but raw attention patterns need systematic analysis before we infer model behavior from them.
- Key idea: BERT attention heads show recurring positional, syntactic, delimiter, and coreference-like patterns across layers.
- Method: analyze attention heads by layer, token relation, syntactic dependency, and qualitative examples.
- Results: some heads specialize in interpretable behaviors, while many heads remain diffuse or redundant.
- What changed after this paper: it gave the club a concrete way to discuss model internals with engineering discipline rather than treating attention maps as decoration.

## Discussion

1. Which attention patterns look genuinely functional, and which might be artifacts of tokenization or positional structure?
2. How much evidence do we need before calling an attention head "interpretable"?
3. What would we measure if we wanted to reproduce this analysis in Rust?

## Actions

- [ ]
