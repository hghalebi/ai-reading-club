//! Public tokenizer API split into small, tutorial-oriented modules.
//!
//! This module is the curriculum-facing surface. It introduces explicit domain
//! types so each API call reads as domain language instead of raw numeric/string
//! plumbing.
//!
//! The high-level flow is:
//! 1. build or load a vocabulary model,
//! 2. encode and decode text with predictable special-token handling,
//! 3. persist or restore learned artifacts.

use crate::encoding::{encode_with_special_tokens, encode_without_special_tokens};
use crate::io::{read_json_value, read_text_file, save_json_file};
use crate::training::{
    add_allowed_special_tokens, append_merged_symbols, ensure_control_symbol_mapping,
    learn_merge_rules, parse_openai_merges, validate_required_openai_ids,
};
use crate::util::{
    find_most_frequent_pair, preprocess_spaces_as_leading_marker, unique_char_symbols,
};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt::{Display, Formatter};
use std::path::{Path, PathBuf};
use thiserror::Error;

/// GPT-2-style marker that prepends a leading space in tokenized symbols.
pub const GPT2_SPACE_MARKER: &str = "Ġ";
/// Default special token used in the notebook-compatible flow.
pub const DEFAULT_END_OF_TEXT: &str = "<|endoftext|>";
/// Minimum safe vocabulary size for training.
pub const MIN_VOCAB_SIZE: usize = 256;
/// Expected GPT-2 newline token id in `encoder.json`.
pub const GPT2_NEWLINE_ID: usize = 198;
/// Expected GPT-2 carriage-return token id in `encoder.json`.
pub const GPT2_CARRIAGE_RETURN_ID: usize = 201;
/// Expected GPT-2 end-of-text token id in `encoder.json`.
pub const GPT2_END_OF_TEXT_ID: usize = 50_256;
/// GPT-2 placeholder symbol that maps to newline.
pub const GPT2_NEWLINE_SYMBOL: &str = "\u{010A}";

/// Stable, typed identifier for token table rows.
///
/// This wrapper is deliberately small: it exists to prevent accidental confusion
/// between IDs, symbol text, and merged output values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TokenId(usize);

impl TokenId {
    /// Creates a typed token identifier from a numeric ID.
    pub const fn new(raw: usize) -> Self {
        Self(raw)
    }

    /// Returns the raw numeric ID.
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::tokenizer::TokenId;
    /// let id = TokenId::new(7);
    /// assert_eq!(id.value(), 7);
    /// ```
    pub const fn value(self) -> usize {
        self.0
    }
}

impl Display for TokenId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A mergeable symbol fragment that can be resolved in vocab tables.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TokenSymbol(String);

impl TokenSymbol {
    /// Creates a symbol from a string-like value.
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::tokenizer::TokenSymbol;
    /// let symbol = TokenSymbol::new("hello");
    /// assert_eq!(symbol.as_str(), "hello");
    /// ```
    pub fn new<S: Into<String>>(value: S) -> Self {
        Self(value.into())
    }

    /// Creates a one-character symbol.
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::tokenizer::TokenSymbol;
    /// let symbol = TokenSymbol::from_char('A');
    /// assert_eq!(symbol.as_str(), "A");
    /// ```
    pub fn from_char(ch: char) -> Self {
        Self(ch.to_string())
    }

    /// Returns a borrowed symbol string view.
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::tokenizer::TokenSymbol;
    /// let symbol = TokenSymbol::new("space");
    /// assert_eq!(symbol.as_str(), "space");
    /// ```
    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }

    /// Concatenates adjacent symbols when applying a merge rule.
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::tokenizer::TokenSymbol;
    ///
    /// let left = TokenSymbol::new("Hel");
    /// let right = TokenSymbol::new("lo");
    /// assert_eq!(TokenSymbol::concat(&left, &right).as_str(), "Hello");
    /// ```
    pub fn concat(left: &TokenSymbol, right: &TokenSymbol) -> Self {
        Self(format!("{}{}", left.as_str(), right.as_str()))
    }
}

impl Display for TokenSymbol {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Input text wrapper used by the public tokenizer API.
///
/// The wrapper keeps public method signatures consistent and avoids accidental
/// ad-hoc usage of raw `String` and `&str` across entry points.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Text(String);

impl Text {
    /// Creates `Text` from any UTF-8 owned input.
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::tokenizer::Text;
    /// let text = Text::new("hello");
    /// assert_eq!(text.as_str(), "hello");
    /// ```
    pub fn new(raw: impl Into<String>) -> Self {
        Self(raw.into())
    }

    /// Returns a borrowed view of the wrapped text.
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::tokenizer::Text;
    /// let text = Text::new("hello world");
    /// assert_eq!(text.as_str().contains(' '), true);
    /// ```
    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }

    /// Unwraps the owned text for caller-owned consumption.
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::tokenizer::Text;
    /// let original = String::from("abc");
    /// let text = Text::new(&original);
    /// assert_eq!(text.into_inner(), "abc".to_string());
    /// ```
    pub fn into_inner(self) -> String {
        self.0
    }
}

impl Display for Text {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Validated vocabulary target for the training loop.
///
/// The value is intentionally validated at construction time so downstream
/// functions can assume a minimum-sized symbol table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VocabularySize(usize);

impl VocabularySize {
    /// Validates and wraps a requested vocabulary size.
    ///
    /// # Errors
    ///
    /// Returns [`TokenizerError::InvalidVocabularySize`] if `raw` is below
    /// [`MIN_VOCAB_SIZE`].
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::tokenizer::{VocabularySize, MIN_VOCAB_SIZE};
    /// assert!(VocabularySize::try_new(MIN_VOCAB_SIZE).is_ok());
    /// assert!(VocabularySize::try_new(MIN_VOCAB_SIZE - 1).is_err());
    /// ```
    pub fn try_new(raw: usize) -> Result<Self> {
        if raw < MIN_VOCAB_SIZE {
            Err(TokenizerError::InvalidVocabularySize {
                requested: Self(raw),
                minimum: Self(MIN_VOCAB_SIZE),
            })
        } else {
            Ok(Self(raw))
        }
    }

    /// Returns the validated numeric vocab size.
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::tokenizer::VocabularySize;
    /// let size = VocabularySize::try_new(512).unwrap();
    /// assert_eq!(size.value(), 512);
    /// ```
    pub const fn value(self) -> usize {
        self.0
    }
}

impl Display for VocabularySize {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Pair of adjacent token IDs that form a merge candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TokenIdPair {
    left: TokenId,
    right: TokenId,
}

impl TokenIdPair {
    /// Creates an ordered pair from two IDs.
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::tokenizer::{TokenId, TokenIdPair};
    /// let pair = TokenIdPair::new(TokenId::new(1), TokenId::new(2));
    /// assert_eq!(pair.left(), TokenId::new(1));
    /// assert_eq!(pair.right(), TokenId::new(2));
    /// ```
    pub const fn new(left: TokenId, right: TokenId) -> Self {
        Self { left, right }
    }

    /// Returns the left side of the pair.
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::tokenizer::{TokenId, TokenIdPair};
    /// assert_eq!(TokenIdPair::new(TokenId::new(1), TokenId::new(2)).left(), TokenId::new(1));
    /// ```
    pub const fn left(self) -> TokenId {
        self.left
    }

    /// Returns the right side of the pair.
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::tokenizer::{TokenId, TokenIdPair};
    /// assert_eq!(TokenIdPair::new(TokenId::new(1), TokenId::new(2)).right(), TokenId::new(2));
    /// ```
    pub const fn right(self) -> TokenId {
        self.right
    }
}

/// Merge priority where lower values are preferred.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct MergePriority(usize);

impl MergePriority {
    /// Wraps a merge-rank value.
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::tokenizer::MergePriority;
    ///
    /// let highest_priority = MergePriority::new(0);
    /// assert!(highest_priority <= MergePriority::new(10));
    /// ```
    pub const fn new(value: usize) -> Self {
        Self(value)
    }
}

/// Collection of explicit special tokens allowed to bypass BPE.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SpecialTokenSet {
    values: BTreeSet<TokenSymbol>,
}

impl SpecialTokenSet {
    /// Builds an ordered set from token symbols.
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::tokenizer::{SpecialTokenSet, TokenSymbol};
    /// let allowed = SpecialTokenSet::new([TokenSymbol::new("<|endoftext|>")]);
    /// assert!(!allowed.is_empty());
    /// assert!(allowed.contains(&TokenSymbol::new("<|endoftext|>")));
    /// ```
    pub fn new(values: impl IntoIterator<Item = TokenSymbol>) -> Self {
        let mut set = Self {
            values: BTreeSet::new(),
        };
        set.values.extend(values);
        set
    }

    /// Returns `true` when no special tokens are configured.
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::tokenizer::SpecialTokenSet;
    /// let allowed = SpecialTokenSet::new([]);
    /// assert!(allowed.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Returns `true` if `token` is configured as allowed special token.
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::tokenizer::{SpecialTokenSet, TokenSymbol};
    /// let special = SpecialTokenSet::new([TokenSymbol::new("<|x|>")]);
    /// assert!(special.contains(&TokenSymbol::new("<|x|>")));
    /// assert!(!special.contains(&TokenSymbol::new("<|y|>")));
    /// ```
    pub fn contains(&self, token: &TokenSymbol) -> bool {
        self.values.contains(token)
    }

    /// Iterates in sorted order for deterministic behavior.
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::tokenizer::{SpecialTokenSet, TokenSymbol};
    /// let special = SpecialTokenSet::new([TokenSymbol::new("<|b|>"), TokenSymbol::new("<|a|>")]);
    /// let symbols: Vec<_> = special.iter().collect();
    /// assert_eq!(symbols[0], &TokenSymbol::new("<|a|>"));
    /// assert_eq!(symbols[1], &TokenSymbol::new("<|b|>"));
    /// ```
    pub fn iter(&self) -> impl Iterator<Item = &TokenSymbol> {
        self.values.iter()
    }
}

impl FromIterator<TokenSymbol> for SpecialTokenSet {
    fn from_iter<I: IntoIterator<Item = TokenSymbol>>(iter: I) -> Self {
        Self::new(iter)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenIdSequence(Vec<TokenId>);

impl TokenIdSequence {
    /// Builds an empty ID sequence.
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::tokenizer::TokenIdSequence;
    /// let sequence = TokenIdSequence::empty();
    /// assert!(sequence.is_empty());
    /// ```
    pub fn empty() -> Self {
        Self(Vec::new())
    }

    /// Builds an ID sequence from a `Vec`.
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::tokenizer::{TokenId, TokenIdSequence};
    /// let sequence = TokenIdSequence::from_vec(vec![TokenId::new(1), TokenId::new(2)]);
    /// assert_eq!(sequence.as_slice(), &[TokenId::new(1), TokenId::new(2)]);
    /// ```
    pub fn from_vec(values: Vec<TokenId>) -> Self {
        Self(values)
    }

    /// Exposes a borrowed slice for inspection and iteration.
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::tokenizer::{TokenId, TokenIdSequence};
    /// let sequence = TokenIdSequence::from_vec(vec![TokenId::new(10)]);
    /// assert_eq!(sequence.as_slice().len(), 1);
    /// ```
    pub fn as_slice(&self) -> &[TokenId] {
        self.0.as_slice()
    }

    /// Converts back into an owned vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::tokenizer::{TokenId, TokenIdSequence};
    /// let ids = vec![TokenId::new(3), TokenId::new(4)];
    /// let sequence = TokenIdSequence::from_vec(ids.clone());
    /// assert_eq!(sequence.into_inner(), ids);
    /// ```
    pub fn into_inner(self) -> Vec<TokenId> {
        self.0
    }

    /// Returns `true` when the sequence contains no IDs.
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::tokenizer::TokenIdSequence;
    /// let sequence = TokenIdSequence::from_vec(vec![]);
    /// assert!(sequence.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Appends one token id.
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::tokenizer::{TokenId, TokenIdSequence};
    /// let mut sequence = TokenIdSequence::empty();
    /// sequence.push(TokenId::new(7));
    /// assert_eq!(sequence.as_slice(), &[TokenId::new(7)]);
    /// ```
    pub fn push(&mut self, token_id: TokenId) {
        self.0.push(token_id);
    }

    /// Appends many token ids from an iterator.
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::tokenizer::{TokenId, TokenIdSequence};
    /// let mut sequence = TokenIdSequence::empty();
    /// sequence.extend([TokenId::new(1), TokenId::new(2)]);
    /// assert_eq!(sequence.as_slice().len(), 2);
    /// ```
    pub fn extend<I: IntoIterator<Item = TokenId>>(&mut self, token_ids: I) {
        self.0.extend(token_ids);
    }
}

impl From<Vec<TokenId>> for TokenIdSequence {
    fn from(value: Vec<TokenId>) -> Self {
        Self(value)
    }
}

impl From<TokenIdSequence> for Vec<TokenId> {
    fn from(value: TokenIdSequence) -> Self {
        value.into_inner()
    }
}

/// Filesystem path to vocab/merge artifact files.
#[derive(Debug, Clone)]
pub struct ArtifactPath(PathBuf);

impl ArtifactPath {
    /// Builds a typed artifact path.
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::tokenizer::ArtifactPath;
    /// let path = ArtifactPath::new("/tmp/vocab.json");
    /// assert_eq!(path.as_path().to_string_lossy(), "/tmp/vocab.json");
    /// ```
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self(path.into())
    }

    /// Returns a borrowed path reference.
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::tokenizer::ArtifactPath;
    /// let path = ArtifactPath::new("/tmp/vocab.json");
    /// assert!(path.as_path().is_absolute() == false);
    /// ```
    pub fn as_path(&self) -> &Path {
        &self.0
    }
}

impl AsRef<Path> for ArtifactPath {
    fn as_ref(&self) -> &Path {
        self.as_path()
    }
}

impl Display for ArtifactPath {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0.display())
    }
}

/// Localized result type for all tokenizer operations.
pub type Result<T> = std::result::Result<T, TokenizerError>;

/// Full tokenizer state and conversion tables.
#[derive(Debug, Default)]
pub struct BPETokenizerSimple {
    /// ID → symbol table.
    pub vocab: BTreeMap<TokenId, TokenSymbol>,
    /// Symbol → ID table.
    pub inverse_vocab: HashMap<TokenSymbol, TokenId>,
    /// Learn-time merges produced during training.
    pub bpe_merges: HashMap<TokenIdPair, TokenId>,
    /// Deterministic order used to materialize merged symbols.
    pub merge_order: Vec<(TokenIdPair, TokenId)>,
    /// GPT-2 rank table used by OpenAI-compatible merge path.
    pub bpe_ranks: HashMap<(TokenSymbol, TokenSymbol), MergePriority>,
}

impl BPETokenizerSimple {
    /// Creates a tokenizer in a clean, empty state.
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::tokenizer::BPETokenizerSimple;
    /// let tokenizer = BPETokenizerSimple::new();
    /// assert!(tokenizer.vocab.is_empty());
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    /// Train a tokenizer from raw text.
    ///
    /// This follows three explicit phases:
    /// 1. Build an initial symbol table from bytes + observed symbols.
    /// 2. Add allowed special tokens so the training loop can reserve IDs.
    /// 3. Learn merge pairs until target vocabulary size is reached.
    ///
    /// # Examples
    ///
    /// ```
    /// # use crate::tokenizer::{
    /// #   BPETokenizerSimple, SpecialTokenSet, Text, VocabularySize, DEFAULT_END_OF_TEXT, TokenSymbol,
    /// # };
    /// let mut tokenizer = BPETokenizerSimple::new();
    /// let text = Text::new("train me");
    /// let allowed = SpecialTokenSet::new([TokenSymbol::new(DEFAULT_END_OF_TEXT)]);
    /// tokenizer
    ///     .train(&text, VocabularySize::try_new(512).unwrap(), &allowed)
    ///     .unwrap();
    /// assert!(!tokenizer.vocab.is_empty());
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`TokenizerError::InvalidVocabularySize`] if `vocab_size` is below
    /// [`MIN_VOCAB_SIZE`], or [`TokenizerError::MissingToken`] if preprocessing
    /// yields a character with no initial vocabulary mapping.
    ///
    /// # Panics
    ///
    /// This method does not intentionally panic.
    pub fn train(
        &mut self,
        text: &Text,
        vocab_size: VocabularySize,
        allowed_special: &SpecialTokenSet,
    ) -> Result<()> {
        self.reset();

        let processed_text = preprocess_spaces_as_leading_marker(text).into_inner();
        let mut symbols: Vec<TokenSymbol> = (0u32..=255)
            .filter_map(std::char::from_u32)
            .map(TokenSymbol::from_char)
            .collect();
        for symbol in unique_char_symbols(&processed_text) {
            let token = TokenSymbol::from_char(symbol);
            if !symbols.contains(&token) {
                symbols.push(token);
            }
        }
        if !symbols
            .iter()
            .any(|symbol| symbol.as_str() == GPT2_SPACE_MARKER)
        {
            symbols.push(TokenSymbol::new(GPT2_SPACE_MARKER));
        }

        for (raw_id, symbol) in symbols.into_iter().enumerate() {
            let token_id = TokenId::new(raw_id);
            self.vocab.insert(token_id, symbol.clone());
            self.inverse_vocab.insert(symbol, token_id);
        }

        add_allowed_special_tokens(self, allowed_special);

        let mut token_ids: TokenIdSequence = processed_text
            .chars()
            .map(|character| {
                self.inverse_vocab
                    .get(&TokenSymbol::from_char(character))
                    .copied()
                    .ok_or_else(|| TokenizerError::MissingToken(TokenSymbol::from_char(character)))
            })
            .collect::<Result<Vec<_>>>()?
            .into();

        learn_merge_rules(self, &mut token_ids, vocab_size.value())?;
        append_merged_symbols(self)?;

        Ok(())
    }

    /// Loads OpenAI-compatible artifacts (`encoder.json` and `vocab.bpe`).
    ///
    /// The loader validates GPT-2 anchor tokens and builds merge priority table
    /// from the textual merge file.
    ///
    /// # Errors
    ///
    /// Propagates file or JSON failures via [`TokenizerError::FileRead`] and
    /// [`TokenizerError::JsonParse`], and anchor validation failures as
    /// [`TokenizerError::InvalidArtifact`] / [`TokenizerError::UnknownSpecialMissing`].
    pub fn load_vocab_and_merges_from_openai(
        &mut self,
        vocab_path: &ArtifactPath,
        bpe_merges_path: &ArtifactPath,
    ) -> Result<()> {
        self.reset();

        let raw_vocab = read_json_value(vocab_path)?;
        let loaded_vocab: BTreeMap<String, usize> =
            serde_json::from_value(raw_vocab).map_err(|source| TokenizerError::JsonParse {
                path: vocab_path.clone(),
                source,
            })?;

        for (symbol_text, id) in loaded_vocab {
            let token_id = TokenId::new(id);
            let symbol = TokenSymbol::new(symbol_text);
            self.vocab.insert(token_id, symbol.clone());
            self.inverse_vocab.insert(symbol, token_id);
        }

        validate_required_openai_ids(self)?;
        ensure_control_symbol_mapping(self)?;

        let merge_text = read_text_file(bpe_merges_path)?;
        parse_openai_merges(self, &merge_text);

        Ok(())
    }

    /// Persists the current vocabulary and merge rules to JSON files.
    ///
    /// The artifacts are intentionally simple key/value structures so they can be
    /// versioned and reloaded deterministically for teaching and debugging.
    ///
    /// # Errors
    ///
    /// Returns [`TokenizerError::JsonParse`] when values cannot be serialized.
    pub fn save_vocab_and_merges(
        &self,
        vocab_path: &ArtifactPath,
        bpe_merges_path: &ArtifactPath,
    ) -> Result<()> {
        let serialized_vocab: BTreeMap<usize, String> = self
            .vocab
            .iter()
            .map(|(id, symbol)| (id.value(), symbol.as_str().to_string()))
            .collect();

        let serialized_merges: Vec<serde_json::Value> = self
            .merge_order
            .iter()
            .map(|(pair, merged_id)| {
                serde_json::json!({
                    "pair": [pair.left().value(), pair.right().value()],
                    "new_id": merged_id.value(),
                })
            })
            .collect();

        save_json_file(
            vocab_path,
            &serde_json::to_value(serialized_vocab).map_err(|source| {
                TokenizerError::JsonParse {
                    path: vocab_path.clone(),
                    source,
                }
            })?,
        )?;
        save_json_file(
            bpe_merges_path,
            &serde_json::to_value(serialized_merges).map_err(|source| {
                TokenizerError::JsonParse {
                    path: bpe_merges_path.clone(),
                    source,
                }
            })?,
        )?;

        Ok(())
    }

    /// Loads artifacts created by [`Self::save_vocab_and_merges`].
    ///
    /// # Errors
    ///
    /// Returns:
    /// - [`TokenizerError::FileRead`] and [`TokenizerError::JsonParse`] for I/O/JSON issues.
    /// - [`TokenizerError::InvalidArtifact`] when the serialized shape is invalid.
    pub fn load_vocab_and_merges(
        &mut self,
        vocab_path: &ArtifactPath,
        bpe_merges_path: &ArtifactPath,
    ) -> Result<()> {
        self.reset();

        let raw_vocab = read_json_value(vocab_path)?;
        let loaded_vocab: BTreeMap<String, String> =
            serde_json::from_value(raw_vocab).map_err(|source| TokenizerError::JsonParse {
                path: vocab_path.clone(),
                source,
            })?;

        for (raw_id, symbol_text) in loaded_vocab {
            let id: usize = raw_id.parse().map_err(|_| {
                TokenizerError::InvalidArtifact(format!(
                    "token id '{raw_id}' in vocab is not numeric"
                ))
            })?;
            let token_id = TokenId::new(id);
            let symbol = TokenSymbol::new(symbol_text);
            self.vocab.insert(token_id, symbol.clone());
            self.inverse_vocab.insert(symbol, token_id);
        }

        let raw_merges = read_json_value(bpe_merges_path)?;
        let records: Vec<serde_json::Value> =
            serde_json::from_value(raw_merges).map_err(|source| TokenizerError::JsonParse {
                path: bpe_merges_path.clone(),
                source,
            })?;

        for record in records {
            let pair = record
                .get("pair")
                .and_then(|value| value.as_array())
                .ok_or_else(|| {
                    TokenizerError::InvalidArtifact(
                        "merge record must include an array field named 'pair'".to_string(),
                    )
                })?;
            if pair.len() != 2 {
                return Err(TokenizerError::InvalidArtifact(
                    "merge pair must contain two numeric token IDs".to_string(),
                ));
            }

            let Some(left_raw) = pair[0].as_u64() else {
                return Err(TokenizerError::InvalidArtifact(
                    "merge pair values must be integer token IDs".to_string(),
                ));
            };
            let Some(right_raw) = pair[1].as_u64() else {
                return Err(TokenizerError::InvalidArtifact(
                    "merge pair values must be integer token IDs".to_string(),
                ));
            };
            let Some(new_raw) = record.get("new_id").and_then(|id| id.as_u64()) else {
                return Err(TokenizerError::InvalidArtifact(
                    "merge record must include integer field 'new_id'".to_string(),
                ));
            };

            let pair = TokenIdPair::new(
                TokenId::new(left_raw as usize),
                TokenId::new(right_raw as usize),
            );
            let merged_id = TokenId::new(new_raw as usize);
            self.bpe_merges.insert(pair, merged_id);
            self.merge_order.push((pair, merged_id));
        }

        Ok(())
    }

    /// Encodes plain text into token IDs.
    ///
    /// When `allowed_special` is `Some`, tokens in that set pass through unchanged.
    /// All other special-token patterns are rejected.
    ///
    /// # Errors
    ///
    /// Returns [`TokenizerError::MissingToken`] for unmapped characters and
    /// [`TokenizerError::UnknownSpecialToken`] when disallowed special tokens are
    /// found in input text.
    ///
    /// # Examples
    ///
    /// ```
    /// # use crate::tokenizer::{
    /// #   BPETokenizerSimple, SpecialTokenSet, Text, VocabularySize, DEFAULT_END_OF_TEXT, TokenSymbol,
    /// # };
    /// let mut tokenizer = BPETokenizerSimple::new();
    /// let text = Text::new("hello world");
    /// let allowed = SpecialTokenSet::new([TokenSymbol::new(DEFAULT_END_OF_TEXT)]);
    /// tokenizer
    ///     .train(&text, VocabularySize::try_new(300).unwrap(), &allowed)
    ///     .unwrap();
    /// let encoded = tokenizer.encode(&text, Some(&allowed)).unwrap();
    /// assert!(!encoded.is_empty());
    /// ```
    pub fn encode(
        &self,
        text: &Text,
        allowed_special: Option<&SpecialTokenSet>,
    ) -> Result<TokenIdSequence> {
        match allowed_special.filter(|token_set| !token_set.is_empty()) {
            Some(token_set) => encode_with_special_tokens(self, text, token_set),
            None => encode_without_special_tokens(self, text),
        }
    }

    /// Decodes token IDs back into text.
    ///
    /// This mirrors the inverse of the training/encoding rules:
    /// - control-symbol markers are reconstructed,
    /// - GPT-2 space markers are converted back to `' '`.
    ///
    /// # Errors
    ///
    /// Returns [`TokenizerError::MissingTokenId`] when a token id has no
    /// vocabulary entry.
    ///
    /// # Examples
    ///
    /// ```
    /// # use crate::tokenizer::{
    /// #     BPETokenizerSimple, SpecialTokenSet, Text, VocabularySize, DEFAULT_END_OF_TEXT, TokenSymbol,
    /// # };
    /// let mut tokenizer = BPETokenizerSimple::new();
    /// let input = Text::new("decode");
    /// let allowed = SpecialTokenSet::new([TokenSymbol::new(DEFAULT_END_OF_TEXT)]);
    /// tokenizer
    ///     .train(&input, VocabularySize::try_new(300).unwrap(), &allowed)
    ///     .unwrap();
    /// let ids = tokenizer.encode(&input, Some(&allowed)).unwrap();
    /// let decoded = tokenizer.decode(&ids).unwrap();
    /// assert_eq!(decoded, input);
    /// ```
    pub fn decode(&self, token_ids: &TokenIdSequence) -> Result<Text> {
        let mut output = String::new();
        let newline_id = TokenId::new(GPT2_NEWLINE_ID);
        let carriage_id = TokenId::new(GPT2_CARRIAGE_RETURN_ID);

        for token_id in token_ids.as_slice() {
            let symbol = self
                .vocab
                .get(token_id)
                .ok_or(TokenizerError::MissingTokenId(*token_id))?;
            let token = symbol.as_str();

            if token == "\r" || *token_id == carriage_id {
                output.push('\r');
            } else if token_id == &newline_id || token == "\n" {
                if !output.ends_with(' ') {
                    output.push(' ');
                }
                output.push('\n');
            } else if let Some(rest) = token.strip_prefix(GPT2_SPACE_MARKER) {
                output.push(' ');
                output.push_str(rest);
            } else {
                output.push_str(token);
            }
        }

        Ok(Text::new(output))
    }

    /// Returns the ID for a user-facing symbol, if present.
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::tokenizer::{BPETokenizerSimple, TokenSymbol};
    /// let tokenizer = BPETokenizerSimple::new();
    /// assert_eq!(tokenizer.get_special_token_id(&TokenSymbol::new("<|missing|>")), None);
    /// ```
    pub fn get_special_token_id(&self, token: &TokenSymbol) -> Option<TokenId> {
        self.inverse_vocab.get(token).copied()
    }

    /// Finds the most frequent adjacent ID pair (for educational demonstration).
    ///
    /// If no adjacent pair exists, returns `None`.
    ///
    /// # Examples
    /// ```
    /// use crate::tokenizer::{BPETokenizerSimple, TokenId, TokenIdSequence};
    /// let ids = TokenIdSequence::from_vec(vec![TokenId::new(1), TokenId::new(2), TokenId::new(1)]);
    /// assert_eq!(
    ///     BPETokenizerSimple::find_freq_pair(&ids),
    ///     Some(crate::tokenizer::TokenIdPair::new(TokenId::new(1), TokenId::new(2)))
    /// );
    /// ```
    pub fn find_freq_pair(token_ids: &TokenIdSequence) -> Option<TokenIdPair> {
        find_most_frequent_pair(token_ids)
    }

    fn reset(&mut self) {
        self.vocab.clear();
        self.inverse_vocab.clear();
        self.bpe_merges.clear();
        self.merge_order.clear();
        self.bpe_ranks.clear();
    }
}

#[derive(Debug, Error)]
pub enum TokenizerError {
    /// Returned when `train()` receives a vocabulary target smaller than
    /// [`MIN_VOCAB_SIZE`].
    #[error("training vocabulary size {requested} is below minimum {minimum}")]
    InvalidVocabularySize {
        requested: VocabularySize,
        minimum: VocabularySize,
    },

    /// Returned when a source symbol is absent from the vocabulary map.
    #[error("token '{0}' is not present in the vocabulary")]
    MissingToken(TokenSymbol),

    /// Returned when an id does not resolve through the ID→symbol vocabulary table.
    #[error("token id '{0}' is not present in the vocabulary")]
    MissingTokenId(TokenId),

    /// Returned when an input contains a special token that is not in the allowed set.
    #[error("special token '{0}' is not allowed in this input")]
    UnknownSpecialToken(TokenSymbol),

    /// Returned when a persisted artifact has structurally invalid contents.
    #[error("invalid artifact file: {0}")]
    InvalidArtifact(String),

    /// Returned when reading artifact files fails.
    #[error("could not read artifact '{path}': {source}")]
    FileRead {
        path: ArtifactPath,
        source: std::io::Error,
    },

    /// Returned when JSON encoding/decoding fails.
    #[error("could not parse JSON artifact '{path}': {source}")]
    JsonParse {
        path: ArtifactPath,
        source: serde_json::Error,
    },

    /// Returned when a known special marker is required by a loader but absent.
    #[error("special token '{0}' is missing in this tokenizer")]
    UnknownSpecialMissing(TokenSymbol),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vocabulary_size_is_validated() {
        let error = VocabularySize::try_new(1);
        assert!(error.is_err());
        let valid = VocabularySize::try_new(MIN_VOCAB_SIZE);
        assert!(valid.is_ok());
    }

    #[test]
    fn token_sequence_and_pair_round_trip() {
        let sequence = TokenIdSequence::from_vec(vec![TokenId::new(1), TokenId::new(2)]);
        let pair = TokenIdPair::new(TokenId::new(1), TokenId::new(2));

        assert_eq!(sequence.as_slice(), &[TokenId::new(1), TokenId::new(2)]);
        assert_eq!(pair.left(), TokenId::new(1));
        assert_eq!(pair.right(), TokenId::new(2));
    }

    #[test]
    fn encode_roundtrip_keeps_text() {
        let mut tokenizer = BPETokenizerSimple::new();
        let text = Text::new("Byte Pair Encoding keeps frequent byte patterns together.");
        let allowed = SpecialTokenSet::new([TokenSymbol::new(DEFAULT_END_OF_TEXT)]);

        tokenizer
            .train(
                &text,
                VocabularySize::try_new(512).expect("valid vocabulary size"),
                &allowed,
            )
            .expect("training should work");
        let encoded = tokenizer
            .encode(&text, Some(&allowed))
            .expect("encoding should work");
        let decoded = tokenizer.decode(&encoded).expect("decoding should work");

        assert_eq!(decoded, text);
    }

    #[test]
    fn most_frequent_pair_prefers_leftmost_when_tied() {
        let ids = TokenIdSequence::from_vec(vec![
            TokenId::new(1),
            TokenId::new(2),
            TokenId::new(1),
            TokenId::new(2),
            TokenId::new(1),
        ]);
        assert_eq!(
            BPETokenizerSimple::find_freq_pair(&ids),
            Some(TokenIdPair::new(TokenId::new(1), TokenId::new(2)))
        );
    }

    #[test]
    fn encode_rejects_disallowed_special_tokens_in_plain_text() {
        let mut tokenizer = BPETokenizerSimple::new();
        let allowed = SpecialTokenSet::new([TokenSymbol::new(DEFAULT_END_OF_TEXT)]);
        let extra = SpecialTokenSet::new([
            TokenSymbol::new(DEFAULT_END_OF_TEXT),
            TokenSymbol::new("<|bad|>"),
        ]);
        let text = Text::new("Hello <|bad|> world");

        tokenizer
            .train(&text, VocabularySize::try_new(256).expect("valid"), &extra)
            .expect("training with explicit special token");
        let result = tokenizer.encode(&text, Some(&allowed));

        assert!(
            matches!(result, Err(TokenizerError::UnknownSpecialToken(token)) if token == TokenSymbol::new("<|bad|>"))
        );
    }
}
