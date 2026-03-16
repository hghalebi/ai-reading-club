//! Encoding logic split out for readability.
//!
//! The public tokenizer delegates here for the two cases:
//! - plain text path (no explicit specials), and
//! - special-token aware path that preserves configured markers.

use crate::tokenizer::{
    BPETokenizerSimple, MergePriority, Result, SpecialTokenSet, Text, TokenId, TokenIdSequence,
    TokenSymbol, TokenizerError,
};
use crate::util::{
    find_most_frequent_pair, replace_pair, split_by_leading_spaces_marker,
    split_with_newline_keepers,
};

/// Encode text while allowing a set of pre-tokenized special tokens.
///
/// Matching special tokens are emitted directly as IDs and are not passed through
/// BPE splitting.
pub(crate) fn encode_with_special_tokens(
    tokenizer: &BPETokenizerSimple,
    text: &Text,
    allowed_special: &SpecialTokenSet,
) -> Result<TokenIdSequence> {
    let mut output = TokenIdSequence::empty();
    let text = text.as_str();
    let mut cursor = 0usize;

    while cursor < text.len() {
        let remainder = &text[cursor..];
        let next_special = allowed_special
            .iter()
            .filter_map(|token| {
                remainder
                    .find(token.as_str())
                    .map(|relative_position| (cursor + relative_position, token))
            })
            .min_by(|(pos_a, token_a), (pos_b, token_b)| {
                pos_a
                    .cmp(pos_b)
                    .then_with(|| token_b.as_str().len().cmp(&token_a.as_str().len()))
            });

        match next_special {
            Some((position, token)) => {
                if position > cursor {
                    let prefix = &text[cursor..position];
                    check_for_disallowed_special_tokens(tokenizer, prefix, allowed_special)?;
                    output.extend(
                        encode_without_special_tokens(tokenizer, &Text::new(prefix))?.into_inner(),
                    );
                }

                let special_id = tokenizer
                    .inverse_vocab
                    .get(token)
                    .copied()
                    .ok_or_else(|| TokenizerError::UnknownSpecialMissing(token.clone()))?;
                output.push(special_id);
                cursor = position + token.as_str().len();
            }
            None => {
                let remainder = &text[cursor..];
                check_for_disallowed_special_tokens(tokenizer, remainder, allowed_special)?;
                output.extend(
                    encode_without_special_tokens(tokenizer, &Text::new(remainder))?.into_inner(),
                );
                cursor = text.len();
            }
        }
    }

    Ok(output)
}

fn check_for_disallowed_special_tokens(
    tokenizer: &BPETokenizerSimple,
    text: &str,
    allowed_special: &SpecialTokenSet,
) -> Result<()> {
    for token in tokenizer.inverse_vocab.keys() {
        if token.as_str().starts_with("<|")
            && token.as_str().ends_with("|>")
            && !allowed_special.contains(token)
            && text.contains(token.as_str())
        {
            return Err(TokenizerError::UnknownSpecialToken(token.clone()));
        }
    }
    Ok(())
}

/// Encode text after removing special-token handling from the flow.
///
/// This path keeps newline chunks as separate tokens and then applies GPT-2 merge
/// rules or local merge rules depending on whether `bpe_ranks` is populated.
pub(crate) fn encode_without_special_tokens(
    tokenizer: &BPETokenizerSimple,
    text: &Text,
) -> Result<TokenIdSequence> {
    if text.as_str().is_empty() {
        return Ok(TokenIdSequence::empty());
    }

    let mut output = TokenIdSequence::empty();
    for chunk in split_with_newline_keepers(text.as_str()) {
        let tokens = match chunk.as_str() {
            "\r\n" => vec![TokenSymbol::new("\r"), TokenSymbol::new("\n")],
            "\r" => vec![TokenSymbol::new("\r")],
            "\n" => vec![TokenSymbol::new("\n")],
            _ => split_by_leading_spaces_marker(chunk.as_str()),
        };

        for token in tokens {
            if let Some(&id) = tokenizer.inverse_vocab.get(&token) {
                output.push(id);
            } else {
                let tokenized = tokenize_with_bpe(tokenizer, &token)?;
                output.extend(tokenized.into_inner());
            }
        }
    }

    Ok(output)
}

/// Apply BPE merge logic to a pre-split token.
///
/// The algorithm is intentionally faithful to the original classroom version:
/// - local merge mode repeatedly replaces pairs from frequency ranking;
/// - GPT-2 mode repeatedly applies highest-priority ranked symbol pairs.
pub(crate) fn tokenize_with_bpe(
    tokenizer: &BPETokenizerSimple,
    token: &TokenSymbol,
) -> Result<TokenIdSequence> {
    let token_ids: Vec<TokenId> = token
        .as_str()
        .chars()
        .map(|ch| {
            tokenizer
                .inverse_vocab
                .get(&TokenSymbol::from_char(ch))
                .copied()
                .ok_or_else(|| TokenizerError::MissingToken(TokenSymbol::from_char(ch)))
        })
        .collect::<Result<Vec<_>>>()?;

    // A local training tokenizer uses token-ID merge rules.
    if tokenizer.bpe_ranks.is_empty() {
        let mut encoded = TokenIdSequence::from_vec(token_ids);
        while encoded.as_slice().len() > 1 {
            let Some(pair) = find_most_frequent_pair(&encoded) else {
                break;
            };
            let Some(&merged_id) = tokenizer.bpe_merges.get(&pair) else {
                break;
            };
            let replaced = replace_pair(&encoded, pair, merged_id);
            if replaced == encoded {
                break;
            }
            encoded = replaced;
        }
        return Ok(encoded);
    }

    // GPT-2 path: repeatedly merge the highest-priority pair.
    let mut symbols: Vec<TokenSymbol> = token_ids
        .into_iter()
        .map(|token_id| {
            tokenizer
                .vocab
                .get(&token_id)
                .cloned()
                .ok_or(TokenizerError::MissingTokenId(token_id))
        })
        .collect::<Result<Vec<_>>>()?;

    while symbols.len() > 1 {
        let mut best_pair: Option<(TokenSymbol, TokenSymbol)> = None;
        let mut best_rank = MergePriority::new(usize::MAX);

        for pair_slice in symbols.windows(2) {
            let left = pair_slice[0].clone();
            let right = pair_slice[1].clone();
            if let Some(rank) = tokenizer.bpe_ranks.get(&(left.clone(), right.clone())) {
                if *rank < best_rank {
                    best_rank = *rank;
                    best_pair = Some((left, right));
                }
            }
        }

        let Some((left, right)) = best_pair else {
            break;
        };

        let merged = TokenSymbol::concat(&left, &right);

        let mut merged_symbols = Vec::with_capacity(symbols.len());
        let mut index = 0usize;
        while index < symbols.len() {
            if index + 1 < symbols.len() && symbols[index] == left && symbols[index + 1] == right {
                merged_symbols.push(merged.clone());
                index += 2;
            } else {
                merged_symbols.push(symbols[index].clone());
                index += 1;
            }
        }
        symbols = merged_symbols;
        if symbols.len() == 1 {
            break;
        }
    }

    symbols
        .into_iter()
        .map(|symbol| {
            tokenizer
                .inverse_vocab
                .get(&symbol)
                .copied()
                .ok_or(TokenizerError::MissingToken(symbol))
        })
        .collect::<Result<Vec<_>>>()
        .map(Into::into)
}
