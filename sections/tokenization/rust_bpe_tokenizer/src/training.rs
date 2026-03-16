//! Training and artifact-loading helpers for the tokenizer.
//!
//! This module holds the deterministic steps behind `train` and OpenAI artifact
//! loading so the main public type stays readable.

use crate::tokenizer::{
    BPETokenizerSimple, MergePriority, Result, SpecialTokenSet, TokenId, TokenIdSequence,
    TokenSymbol, TokenizerError, DEFAULT_END_OF_TEXT, GPT2_CARRIAGE_RETURN_ID, GPT2_END_OF_TEXT_ID,
    GPT2_NEWLINE_ID, GPT2_NEWLINE_SYMBOL, GPT2_SPACE_MARKER,
};
use crate::util::{find_most_frequent_pair, replace_pair};

/// Add explicit special tokens to the base vocabulary.
///
/// Special tokens are appended with consecutive IDs so merges only operate on
/// real symbol vocabulary entries.
pub(crate) fn add_allowed_special_tokens(
    tokenizer: &mut BPETokenizerSimple,
    allowed_special: &SpecialTokenSet,
) {
    for token in allowed_special.iter() {
        if !tokenizer.inverse_vocab.contains_key(token) {
            let new_id = TokenId::new(tokenizer.vocab.len());
            tokenizer.vocab.insert(new_id, token.clone());
            tokenizer.inverse_vocab.insert(token.clone(), new_id);
        }
    }
}

/// Learn merge rules by repeatedly replacing the most frequent adjacent pair.
///
/// The loop stops when either:
/// - vocab size target is reached, or
/// - no adjacent pair remains.
pub(crate) fn learn_merge_rules(
    tokenizer: &mut BPETokenizerSimple,
    token_ids: &mut TokenIdSequence,
    target_vocab_size: usize,
) -> Result<()> {
    if target_vocab_size < tokenizer.vocab.len() {
        return Ok(());
    }

    let mut next_id = tokenizer.vocab.len();
    while next_id < target_vocab_size {
        let Some(pair) = find_most_frequent_pair(token_ids) else {
            break;
        };
        let replacement = TokenId::new(next_id);
        *token_ids = replace_pair(token_ids, pair, replacement);
        tokenizer.bpe_merges.insert(pair, replacement);
        tokenizer.merge_order.push((pair, replacement));
        next_id += 1;
    }

    Ok(())
}

/// Materialize merged symbols from pair rules into the forward vocabulary map.
///
/// This keeps `vocab` and `inverse_vocab` in sync with `merge_order`.
pub(crate) fn append_merged_symbols(tokenizer: &mut BPETokenizerSimple) -> Result<()> {
    for (pair, merged_id) in tokenizer.merge_order.iter().copied() {
        let left = tokenizer
            .vocab
            .get(&pair.left())
            .ok_or(TokenizerError::MissingTokenId(pair.left()))?;
        let right = tokenizer
            .vocab
            .get(&pair.right())
            .ok_or(TokenizerError::MissingTokenId(pair.right()))?;
        let merged = TokenSymbol::concat(left, right);
        tokenizer.vocab.insert(merged_id, merged.clone());
        tokenizer.inverse_vocab.insert(merged, merged_id);
    }

    Ok(())
}

/// Parse `vocab.bpe` lines into ranked symbol pairs.
///
/// Invalid lines and comments are skipped, matching the common GPT-2 artifact format.
pub(crate) fn parse_openai_merges(tokenizer: &mut BPETokenizerSimple, merge_text: &str) {
    let mut rank = 0usize;

    for line in merge_text.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() != 2 {
            continue;
        }
        let left = TokenSymbol::new(parts[0]);
        let right = TokenSymbol::new(parts[1]);
        if tokenizer.inverse_vocab.contains_key(&left)
            && tokenizer.inverse_vocab.contains_key(&right)
        {
            tokenizer
                .bpe_ranks
                .insert((left, right), MergePriority::new(rank));
            rank += 1;
        }
    }
}

/// Validate expected GPT-2 anchors are available before applying merge ranks.
///
/// Ensures token IDs used by downstream assumptions are present and stable.
pub(crate) fn validate_required_openai_ids(tokenizer: &BPETokenizerSimple) -> Result<()> {
    let newline_id = tokenizer
        .inverse_vocab
        .get(&TokenSymbol::new(GPT2_NEWLINE_SYMBOL))
        .ok_or_else(|| {
            TokenizerError::UnknownSpecialMissing(TokenSymbol::new(GPT2_NEWLINE_SYMBOL))
        })?;
    if newline_id.value() != GPT2_NEWLINE_ID {
        return Err(TokenizerError::InvalidArtifact(format!(
            "expected '{GPT2_NEWLINE_SYMBOL}' to have id {GPT2_NEWLINE_ID}"
        )));
    }

    let end_of_text_id = tokenizer
        .inverse_vocab
        .get(&TokenSymbol::new(DEFAULT_END_OF_TEXT))
        .ok_or_else(|| {
            TokenizerError::UnknownSpecialMissing(TokenSymbol::new(DEFAULT_END_OF_TEXT))
        })?;
    if end_of_text_id.value() != GPT2_END_OF_TEXT_ID {
        return Err(TokenizerError::InvalidArtifact(
            "expected <|endoftext|> to have id 50256".to_string(),
        ));
    }

    Ok(())
}

/// Keep compatibility with legacy newline handling used by the decode path.
///
/// The loader may expose newline and carriage-return placeholders through known
/// fallback IDs when those tokens are not explicit in the file.
pub(crate) fn ensure_control_symbol_mapping(tokenizer: &mut BPETokenizerSimple) -> Result<()> {
    // GPT-2 artifacts sometimes model newline as the printable unicode placeholder.
    // Keep a direct "\n" entry by reusing an already known token ID.
    let newline_id = tokenizer
        .inverse_vocab
        .get(&TokenSymbol::new(GPT2_NEWLINE_SYMBOL))
        .copied()
        .ok_or_else(|| {
            TokenizerError::UnknownSpecialMissing(TokenSymbol::new(GPT2_NEWLINE_SYMBOL))
        })?;

    tokenizer.vocab.insert(newline_id, TokenSymbol::new("\n"));
    tokenizer
        .inverse_vocab
        .insert(TokenSymbol::new("\n"), newline_id);

    let carriage_id = TokenId::new(GPT2_CARRIAGE_RETURN_ID);
    let fallback_candidates = [
        TokenSymbol::new(DEFAULT_END_OF_TEXT),
        TokenSymbol::new(GPT2_SPACE_MARKER),
        TokenSymbol::new(""),
    ];

    if !tokenizer
        .inverse_vocab
        .contains_key(&TokenSymbol::new("\r"))
    {
        let fallback_id = tokenizer
            .inverse_vocab
            .keys()
            .find(|token| fallback_candidates.contains(*token))
            .and_then(|token| tokenizer.inverse_vocab.get(token))
            .copied()
            .unwrap_or(carriage_id);

        tokenizer.vocab.insert(fallback_id, TokenSymbol::new("\r"));
        tokenizer
            .inverse_vocab
            .insert(TokenSymbol::new("\r"), fallback_id);
    }

    Ok(())
}
