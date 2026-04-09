//! Shared runtime helpers used by the staged binaries.

use crate::attention::AttentionExample;
use crate::bert::{
    download_model_files, tensor_attentions_to_vec, BertForMaskedLm, DownloadedModelFiles,
};
use crate::error::Result;
use crate::tokenization::BertTokenizer;
use candle_core::{Device, Tensor};

/// Loaded model state reused across multiple analyses.
#[derive(Debug, Clone)]
pub struct ModelBundle {
    pub files: DownloadedModelFiles,
    pub tokenizer: BertTokenizer,
    pub model: BertForMaskedLm,
}

/// Loads the tokenizer and model on CPU.
pub fn load_model_bundle(model_repo: &str) -> Result<ModelBundle> {
    let files = download_model_files(model_repo)?;
    let tokenizer = BertTokenizer::from_file(&files.tokenizer_path)?;
    let device = Device::Cpu;
    let model = BertForMaskedLm::from_files(&files.config_path, &files.weights_path, &device)?;
    Ok(ModelBundle {
        files,
        tokenizer,
        model,
    })
}

/// Extracts token-level attention maps for a single word sequence.
pub fn extract_single_attention(
    bundle: &ModelBundle,
    words: &[String],
    max_sequence_length: usize,
) -> Result<AttentionExample> {
    let encoded = bundle.tokenizer.encode_words(words, max_sequence_length)?;
    let input_ids =
        Tensor::new(encoded.input_ids.as_slice(), bundle.model.device())?.unsqueeze(0)?;
    let token_type_ids =
        Tensor::new(encoded.token_type_ids.as_slice(), bundle.model.device())?.unsqueeze(0)?;
    let attention_mask =
        Tensor::new(encoded.attention_mask.as_slice(), bundle.model.device())?.unsqueeze(0)?;
    let output =
        bundle
            .model
            .forward_with_attentions(&input_ids, &token_type_ids, Some(&attention_mask))?;
    Ok(AttentionExample {
        encoded,
        token_attentions: tensor_attentions_to_vec(&output.attentions)?,
    })
}

/// Extracts token-level attention maps for a BERT pair input.
pub fn extract_pair_attention(
    bundle: &ModelBundle,
    first: &[String],
    second: &[String],
    max_sequence_length: usize,
) -> Result<AttentionExample> {
    let encoded = bundle
        .tokenizer
        .encode_word_pair(first, second, max_sequence_length)?;
    let input_ids =
        Tensor::new(encoded.input_ids.as_slice(), bundle.model.device())?.unsqueeze(0)?;
    let token_type_ids =
        Tensor::new(encoded.token_type_ids.as_slice(), bundle.model.device())?.unsqueeze(0)?;
    let attention_mask =
        Tensor::new(encoded.attention_mask.as_slice(), bundle.model.device())?.unsqueeze(0)?;
    let output =
        bundle
            .model
            .forward_with_attentions(&input_ids, &token_type_ids, Some(&attention_mask))?;
    Ok(AttentionExample {
        encoded,
        token_attentions: tensor_attentions_to_vec(&output.attentions)?,
    })
}

/// Extracts attentions for a corpus of single-sequence examples.
pub fn extract_single_corpus(
    bundle: &ModelBundle,
    examples: &[Vec<String>],
    max_sequence_length: usize,
) -> Result<Vec<AttentionExample>> {
    examples
        .iter()
        .map(|words| extract_single_attention(bundle, words, max_sequence_length))
        .collect()
}

/// Extracts attentions for a corpus of BERT pair inputs.
pub fn extract_pair_corpus(
    bundle: &ModelBundle,
    examples: &[(Vec<String>, Vec<String>)],
    max_sequence_length: usize,
) -> Result<Vec<AttentionExample>> {
    examples
        .iter()
        .map(|(first, second)| extract_pair_attention(bundle, first, second, max_sequence_length))
        .collect()
}
