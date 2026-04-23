//! Shared attention map types and transformations.

use crate::tokenization::EncodedInput;

/// Nested attention tensor representation: `[layer][head][query][key]`.
pub type AttentionTensor = Vec<Vec<Vec<Vec<f32>>>>;

/// A single analyzed example.
#[derive(Debug, Clone)]
pub struct AttentionExample {
    pub encoded: EncodedInput,
    pub token_attentions: AttentionTensor,
}

/// A word-level attention view derived from a token-level example.
#[derive(Debug, Clone)]
pub struct WordAttentionExample {
    pub words: Vec<String>,
    pub tokens: Vec<String>,
    pub word_attentions: AttentionTensor,
}

impl AttentionExample {
    /// Converts token-level attention to word-level attention using the paper's
    /// "sum over target tokens, mean over source tokens" convention.
    pub fn to_word_level(&self) -> WordAttentionExample {
        WordAttentionExample {
            words: self.encoded.words.clone(),
            tokens: self.encoded.tokens.clone(),
            word_attentions: collapse_to_word_level(
                &self.token_attentions,
                &self.encoded.word_spans,
            ),
        }
    }
}

/// Collapses token-level attention maps to word-level attention maps.
pub fn collapse_to_word_level(
    attentions: &AttentionTensor,
    word_spans: &[Vec<usize>],
) -> AttentionTensor {
    attentions
        .iter()
        .map(|layer| {
            layer
                .iter()
                .map(|head| collapse_head_to_word_level(head, word_spans))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>()
}

fn collapse_head_to_word_level(head: &[Vec<f32>], word_spans: &[Vec<usize>]) -> Vec<Vec<f32>> {
    let word_count = word_spans.len();
    let mut collapsed = vec![vec![0.0_f32; word_count]; word_count];

    for (query_word_index, query_span) in word_spans.iter().enumerate() {
        if query_span.is_empty() {
            continue;
        }

        for (key_word_index, key_span) in word_spans.iter().enumerate() {
            if key_span.is_empty() {
                continue;
            }

            let mut total = 0.0_f32;
            for &query_token in query_span {
                let key_mass = key_span
                    .iter()
                    .map(|&key_token| head[query_token][key_token])
                    .sum::<f32>();
                total += key_mass;
            }
            collapsed[query_word_index][key_word_index] = total / query_span.len() as f32;
        }
    }

    collapsed
}

/// Returns the average entropy of the distributions in a head.
pub fn average_entropy(head: &[Vec<f32>]) -> f64 {
    if head.is_empty() {
        return 0.0;
    }

    let total = head
        .iter()
        .map(|row| {
            row.iter()
                .copied()
                .filter(|value| *value > 0.0)
                .map(|value| {
                    let value = value as f64;
                    -value * value.ln()
                })
                .sum::<f64>()
        })
        .sum::<f64>();
    total / head.len() as f64
}

/// Jensen-Shannon divergence between two categorical distributions.
pub fn jensen_shannon_divergence(left: &[f32], right: &[f32]) -> f64 {
    let midpoint = left
        .iter()
        .copied()
        .zip(right.iter().copied())
        .map(|(lhs, rhs)| 0.5_f64 * (lhs as f64 + rhs as f64))
        .collect::<Vec<_>>();

    0.5 * kl_divergence(left, &midpoint) + 0.5 * kl_divergence(right, &midpoint)
}

fn kl_divergence(left: &[f32], right: &[f64]) -> f64 {
    left.iter()
        .copied()
        .zip(right.iter().copied())
        .filter(|(lhs, rhs)| *lhs > 0.0 && *rhs > 0.0)
        .map(|(lhs, rhs)| {
            let lhs = lhs as f64;
            lhs * (lhs / rhs).ln()
        })
        .sum::<f64>()
}

/// Returns the index and value of the largest element while skipping `exclude`.
pub fn argmax_excluding(values: &[f32], exclude: usize) -> Option<(usize, f32)> {
    values
        .iter()
        .copied()
        .enumerate()
        .filter(|(index, _)| *index != exclude)
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn word_level_collapse_matches_sum_then_mean_rule() {
        let head = vec![
            vec![0.0, 0.1, 0.2, 0.7],
            vec![0.0, 0.3, 0.3, 0.4],
            vec![0.0, 0.2, 0.4, 0.4],
            vec![0.0, 0.5, 0.25, 0.25],
        ];
        let collapsed = collapse_head_to_word_level(&head, &[vec![0], vec![1, 2], vec![3]]);
        assert_eq!(collapsed.len(), 3);
        assert!((collapsed[1][2] - 0.4).abs() < 1e-6);
        assert!((collapsed[1][1] - 0.6).abs() < 1e-6);
    }

    #[test]
    fn jensen_shannon_is_zero_for_identical_distributions() {
        let distribution = vec![0.25, 0.25, 0.5];
        assert!(jensen_shannon_divergence(&distribution, &distribution) < 1e-12);
    }
}
