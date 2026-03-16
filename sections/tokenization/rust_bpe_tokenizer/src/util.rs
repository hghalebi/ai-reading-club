//! Shared tokenizer helper functions.
//!
//! These helpers stay small so each step in training and encoding can be
//! followed without unpacking long control flow.
use crate::tokenizer::{Text, TokenId, TokenIdPair, TokenIdSequence, TokenSymbol};
use std::collections::HashMap;

/// Convert spaces into GPT-2 style leading-word markers.
///
/// The first character is preserved as-is so a sentence can start with a
/// literal leading space only when that is literally present in the raw text.
///
/// This function maps every interior `' '` to `GPT2_SPACE_MARKER`, then trims
/// nothing else so token boundaries stay faithful to the original text.
pub fn preprocess_spaces_as_leading_marker(text: &Text) -> Text {
    let mut output = String::new();

    for (index, ch) in text.as_str().chars().enumerate() {
        if ch == ' ' && index != 0 {
            output.push('Ġ');
            continue;
        }
        if ch != ' ' {
            output.push(ch);
        }
    }

    Text::new(output)
}

/// Return all unique non-control characters found in the given text.
///
/// Output is sorted and deduplicated to stabilize deterministic symbol
/// initialization.
pub fn unique_char_symbols(text: &str) -> Vec<char> {
    let mut chars: Vec<char> = text.chars().collect();
    chars.sort_unstable();
    chars.dedup();
    chars
}

/// Keep newline separators as standalone chunks to avoid accidental cross-merges.
///
/// Newlines and carriage returns are emitted as separate chunk symbols so later
/// merge passes treat line boundaries consistently.
pub fn split_with_newline_keepers(text: &str) -> Vec<TokenSymbol> {
    let chars: Vec<char> = text.chars().collect();
    let mut output = Vec::new();
    let mut index = 0usize;

    while index < chars.len() {
        if chars[index] == '\r' {
            if index + 1 < chars.len() && chars[index + 1] == '\n' {
                output.push(TokenSymbol::new("\r\n"));
                index += 2;
            } else {
                output.push(TokenSymbol::new("\r"));
                index += 1;
            }
            continue;
        }

        if chars[index] == '\n' {
            output.push(TokenSymbol::new("\n"));
            index += 1;
            continue;
        }

        let start = index;
        while index < chars.len() && chars[index] != '\r' && chars[index] != '\n' {
            index += 1;
        }
        let piece: String = chars[start..index].iter().collect();
        output.push(TokenSymbol::new(piece));
    }

    output
}

/// Split a chunk by space markers, preserving GPT-2 marker semantics.
///
/// This keeps each word boundary explicit by expanding runs of spaces into
/// explicit marker tokens (`"Ġ"`).
pub fn split_by_leading_spaces_marker(chunk: &str) -> Vec<TokenSymbol> {
    let chars: Vec<char> = chunk.chars().collect();
    let mut output = Vec::new();
    let mut index = 0usize;
    let mut pending_spaces = 0usize;

    while index < chars.len() {
        if chars[index] == ' ' {
            pending_spaces += 1;
            index += 1;
            continue;
        }

        let start = index;
        while index < chars.len() && chars[index] != ' ' {
            index += 1;
        }
        let word: String = chars[start..index].iter().collect();

        if pending_spaces > 0 {
            for _ in 0..(pending_spaces.saturating_sub(1)) {
                output.push(TokenSymbol::new("Ġ"));
            }
            output.push(TokenSymbol::new(format!("Ġ{word}")));
            pending_spaces = 0;
        } else {
            output.push(TokenSymbol::new(word));
        }
    }

    for _ in 0..pending_spaces {
        output.push(TokenSymbol::new("Ġ"));
    }

    output
}

/// Find the most frequent adjacent pair; on ties, prefer the first seen pair.
///
/// The tie-break rule uses earliest first-seen index in the input scan.
pub fn find_most_frequent_pair(token_ids: &TokenIdSequence) -> Option<TokenIdPair> {
    let ids = token_ids.as_slice();
    if ids.len() < 2 {
        return None;
    }

    let mut frequency: HashMap<TokenIdPair, usize> = HashMap::new();
    let mut first_seen: HashMap<TokenIdPair, usize> = HashMap::new();

    for (index, pair_slice) in ids.windows(2).enumerate() {
        let pair = TokenIdPair::new(pair_slice[0], pair_slice[1]);
        *frequency.entry(pair).or_insert(0) += 1;
        first_seen.entry(pair).or_insert(index);
    }

    frequency
        .into_iter()
        .max_by_key(|(pair, count)| {
            let first = first_seen.get(pair).copied().unwrap_or(usize::MAX);
            (*count, usize::MAX - first)
        })
        .map(|(pair, _)| pair)
}

/// Replace all non-overlapping occurrences of `pair` with `replacement`.
///
/// This intentionally skips overlapping pairs to preserve deterministic merge
/// behavior used by the training loop.
pub fn replace_pair(
    token_ids: &TokenIdSequence,
    pair: TokenIdPair,
    replacement: TokenId,
) -> TokenIdSequence {
    let mut output = Vec::with_capacity(token_ids.as_slice().len());
    let ids = token_ids.as_slice();
    let mut index = 0usize;

    while index < ids.len() {
        if index + 1 < ids.len() && ids[index] == pair.left() && ids[index + 1] == pair.right() {
            output.push(replacement);
            index += 2;
            continue;
        }

        output.push(ids[index]);
        index += 1;
    }

    TokenIdSequence::from_vec(output)
}
