# Workshop & AI Reading Club History

This archive separates confirmed or public history from the forward-looking schedule. It includes AI Reading Club sessions and related Rust/ML workshops that extend the club's papers into executable understanding.

## Planned / Registration Links

These are upcoming or registration-only links that have not yet been folded into confirmed history.

| Date | Track | Title / Topic | Link | Status |
| --- | --- | --- | --- | --- |
| TBD | Rust / ML workshop | Category Theory for Tiny ML in Rust - Workshop 3 | Google Meet | Planned / calendar event provided |
| TBD | AI Reading Club / Rust-ML | Next event | https://luma.com/0ufoyoka | Planned / registration link |
| TBD | AI Reading Club / Rust-ML | Next event | https://luma.com/rwwlmo0u | Planned / latest provided registration link |

## Confirmed Sessions

| Date | Format | Title / Topic | Context | Status |
| --- | --- | --- | --- | --- |
| 2026-03 to 2026-04 | AI Reading Club | Attention Is All You Need | Foundational Transformer paper in the early curriculum. | Confirmed as part of early curriculum/history |
| 2026-04-09 | AI Reading Club | What Does BERT Look At? An Analysis of BERT's Attention | Interpretability session around attention heads, layers, syntax, and positional structure. | Confirmed planned/announced session |
| 2026-05 | AI Reading Club | Attention Is Not Explanation | Session in the newer Schoolab/Paris AI Reading Club cycle. | Confirmed held |
| 2026-05-21 | AI Reading Club | Transformer Feed-Forward Layers Are Key-Value Memories | Originally planned for 2026-05-14 and postponed to 2026-05-21. Material: https://rust-ml.com/materials/transformer-ffn-key-value-memories.html. Lab: https://rust-ml.com/#/lab/kvmemory | Confirmed event/session |
| 2026-05-21 | Rust / ML workshop | Category Theory for Tiny ML in Rust - Public Workshop | Originally planned for 2026-05-14 at Schoolab Saint-Lazare, postponed to 2026-05-21 and moved fully online. Streamed, not recorded. | Confirmed scheduled/postponed |
| 2026-06-02 | Rust / ML workshop | Category Theory for Tiny ML in Rust - Workshop 2: Build a Typed Tiny Neuron in Rust | Paris + online / Google Meet, 18:00-19:00 CEST, not recorded. | Confirmed held |
| 2026-06-04 | AI Reading Club / workshop-style session | The Curious Case of Neural Text Degeneration | AI Reading Club at Schoolab. Study page: https://rust-ml.com/materials/neural-text-degeneration.html | Confirmed held |

## AI Reading Club Arc

The original positioning remains:

> Read one foundational AI paper. Discuss what it really means. Connect it to modern LLM systems.

Confirmed or reconstructed paper sequence:

1. Attention Is All You Need
2. What Does BERT Look At? An Analysis of BERT's Attention
3. Attention Is Not Explanation
4. Transformer Feed-Forward Layers Are Key-Value Memories
5. The Curious Case of Neural Text Degeneration

The strongest historical arc so far:

```text
Attention / BERT / FFN memory
        -> how Transformers represent and retrieve information
Neural text degeneration
        -> how model probabilities become product behavior
Rust + category theory
        -> how to rebuild the concepts with executable types
```

## Rust / ML Workshop Branch

Repositories and materials:

- https://github.com/hghalebi/rust-ml
- https://github.com/hghalebi/category_theory_transformer_rs#runnable-examples
- https://rust-ml.com/materials/rust-syntax-intuition-primer.html
- https://rust-ml.com/materials/one-training-step-end-to-end.html

Purpose:

```text
Tiny ML from first principles
-> Rust types
-> typed transformations
-> category-theory intuition
-> executable learning
```

Confirmed workshops:

1. Category Theory for Tiny ML in Rust - Public Workshop
2. Category Theory for Tiny ML in Rust - Workshop 2: Build a Typed Tiny Neuron in Rust

Core teaching mapping:

```text
Object       ~= Type
Morphism     ~= Function / transformation
Composition  ~= pipeline
Identity     ~= do-nothing transformation
Training     ~= endomorphism on ModelState
```

Workshop 2 concepts:

- `Input`
- `Weight`
- `Bias`
- `Prediction`
- `Target`
- `Loss`
- `Gradient`
- `LearningRate`
- `ModelState`

Retrospective improvements to carry forward:

- Map code to category theory in real time.
- Add mini-exercises.
- Ask participants about identity morphisms.
- Ask how to compose morphisms represented as functions.
- Strengthen the theory-code bridge earlier.
- Show tiny Rust snippets right after the theory introduction.

Next workshop direction:

```text
Workshop 3:
Morphisms, Composition, Identity, and Training as Endomorphism
```

Tiny exercise seed:

```rust
fn identity_prediction(p: Prediction) -> Prediction {
    p
}
```

Prompt:

> What is the identity morphism here?

## Neural Text Degeneration Branch

Purpose:

```text
Classic generation papers
-> decoding intuition
-> product behavior
-> API-level implications
```

Confirmed session:

1. The Curious Case of Neural Text Degeneration

Material:

- https://rust-ml.com/materials/neural-text-degeneration.html

Core lesson:

> Decoding is where probability becomes product behavior.

Core concepts:

- likelihood vs generation quality
- greedy decoding
- beam search degeneration
- sampling
- top-k sampling
- nucleus / top-p sampling
- repetition
- entropy
- probability mass
- why API users still need to understand decoding behavior

Potential mini-series:

- Why LLMs repeat themselves
- Why beam search fails for open-ended text
- Why temperature is not "creativity"
- Why top-p is a product decision
- How decoding turns probabilities into UX
