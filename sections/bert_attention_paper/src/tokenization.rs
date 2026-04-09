//! Tokenizer helpers that preserve word-to-token alignment.

use crate::error::{PaperError, Result};
use std::path::Path;
use tokenizers::Tokenizer;

/// An encoded BERT input with explicit word-to-token alignment.
#[derive(Debug, Clone)]
pub struct EncodedInput {
    pub words: Vec<String>,
    pub tokens: Vec<String>,
    pub input_ids: Vec<u32>,
    pub token_type_ids: Vec<u32>,
    pub attention_mask: Vec<u32>,
    pub sequence_ids: Vec<Option<usize>>,
    pub word_spans: Vec<Vec<usize>>,
}

/// Thin wrapper around Hugging Face tokenizers for this project.
#[derive(Debug, Clone)]
pub struct BertTokenizer {
    inner: Tokenizer,
    cls_token_id: u32,
    sep_token_id: u32,
    mask_token_id: u32,
}

impl BertTokenizer {
    /// Loads a tokenizer from a `tokenizer.json` file.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let inner = Tokenizer::from_file(path.as_ref())?;
        let cls_token_id = inner
            .token_to_id("[CLS]")
            .ok_or(PaperError::MissingSpecialToken { token: "[CLS]" })?;
        let sep_token_id = inner
            .token_to_id("[SEP]")
            .ok_or(PaperError::MissingSpecialToken { token: "[SEP]" })?;
        let mask_token_id = inner
            .token_to_id("[MASK]")
            .ok_or(PaperError::MissingSpecialToken { token: "[MASK]" })?;

        Ok(Self {
            inner,
            cls_token_id,
            sep_token_id,
            mask_token_id,
        })
    }

    /// Returns the special `[CLS]` token id.
    pub fn cls_token_id(&self) -> u32 {
        self.cls_token_id
    }

    /// Returns the special `[SEP]` token id.
    pub fn sep_token_id(&self) -> u32 {
        self.sep_token_id
    }

    /// Returns the special `[MASK]` token id.
    pub fn mask_token_id(&self) -> u32 {
        self.mask_token_id
    }

    /// Encodes one already-tokenized word sequence.
    pub fn encode_words(&self, words: &[String], max_len: usize) -> Result<EncodedInput> {
        let refs = words.iter().map(String::as_str).collect::<Vec<_>>();
        let encoding = self.inner.encode(refs.as_slice(), true)?;
        Self::encoding_to_input(words.to_vec(), encoding, 0, max_len)
    }

    /// Encodes a BERT pair input `[CLS] first [SEP] second [SEP]`.
    pub fn encode_word_pair(
        &self,
        first: &[String],
        second: &[String],
        max_len: usize,
    ) -> Result<EncodedInput> {
        let first_refs = first.iter().map(String::as_str).collect::<Vec<_>>();
        let second_refs = second.iter().map(String::as_str).collect::<Vec<_>>();
        let encoding = self
            .inner
            .encode((first_refs.as_slice(), second_refs.as_slice()), true)?;

        let mut words = first.to_vec();
        words.extend(second.iter().cloned());
        Self::encoding_to_input(words, encoding, first.len(), max_len)
    }

    fn encoding_to_input(
        words: Vec<String>,
        encoding: tokenizers::Encoding,
        second_sequence_offset: usize,
        max_len: usize,
    ) -> Result<EncodedInput> {
        if encoding.len() > max_len {
            return Err(PaperError::SequenceTooLong {
                length: encoding.len(),
                maximum: max_len,
            });
        }

        let sequence_ids = encoding.get_sequence_ids();
        let mut word_spans = vec![Vec::new(); words.len()];
        for (token_index, (word_id, sequence_id)) in encoding
            .get_word_ids()
            .iter()
            .copied()
            .zip(sequence_ids.iter().copied())
            .enumerate()
        {
            let Some(word_id) = word_id else {
                continue;
            };
            let global_index = match sequence_id {
                Some(0) | None => word_id as usize,
                Some(1) => second_sequence_offset + word_id as usize,
                Some(other) => {
                    return Err(PaperError::InvalidAnalysis {
                        reason: format!("unexpected sequence id {other} in tokenizer output"),
                    });
                }
            };
            if let Some(span) = word_spans.get_mut(global_index) {
                span.push(token_index);
            }
        }

        Ok(EncodedInput {
            words,
            tokens: encoding.get_tokens().to_vec(),
            input_ids: encoding.get_ids().to_vec(),
            token_type_ids: encoding.get_type_ids().to_vec(),
            attention_mask: encoding.get_attention_mask().to_vec(),
            sequence_ids,
            word_spans,
        })
    }
}
