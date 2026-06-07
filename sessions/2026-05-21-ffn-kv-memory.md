# Session Notes

- Paper: Transformer Feed-Forward Layers Are Key-Value Memories (2020)
- Link: https://arxiv.org/abs/2012.14913
- Date: 2026-05-21 (Thursday)
- Original date: 2026-05-14; postponed to 2026-05-21
- Material: https://rust-ml.com/materials/transformer-ffn-key-value-memories.html
- Interactive lab: https://rust-ml.com/#/lab/kvmemory
- Discussion lead:
- Attendees:

## 10-15 Minute Overview

- Problem: Transformer explanations often over-focus on attention and under-explain the role of feed-forward layers.
- Key idea: feed-forward layers can be interpreted as key-value memories: keys detect patterns, values contribute output directions.
- Method: inspect activations, neurons, and output contributions to connect FFN behavior to stored information.
- Results: FFN layers carry substantial memory-like structure and participate directly in retrieval-like behavior.
- What changed after this paper: it strengthened the club's mechanistic thread by connecting Transformer internals to concrete memory and retrieval intuitions.

## Discussion

1. What does the key-value interpretation explain better than a generic MLP description?
2. Where does the analogy break down?
3. How would this interpretation change the way we debug LLM behavior?

## Actions

- [ ]
