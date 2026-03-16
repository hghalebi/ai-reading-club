//! File and JSON helper functions used by the tokenizer API.
//!
//! These helpers keep all disk interaction in one module so tokenizer behavior is
//! easier to test and easier to explain in a tutorial setting.

use crate::tokenizer::{ArtifactPath, Result, TokenizerError};
use std::fs::read_to_string;

/// Reads a UTF-8 text file and wraps any `io::Error` in
/// [`TokenizerError::FileRead`].
///
/// # Errors
///
/// Returns [`TokenizerError::FileRead`] when the file cannot be opened or read.
pub fn read_text_file(path: &ArtifactPath) -> Result<String> {
    read_to_string(path).map_err(|source| TokenizerError::FileRead {
        path: path.clone(),
        source,
    })
}

/// Reads JSON text from `path` and decodes it with `serde_json`.
///
/// # Errors
///
/// - [`TokenizerError::FileRead`] when the file cannot be read.
/// - [`TokenizerError::JsonParse`] when parsing fails.
pub fn read_json_value(path: &ArtifactPath) -> Result<serde_json::Value> {
    let raw = read_text_file(path)?;
    serde_json::from_str(&raw).map_err(|source| TokenizerError::JsonParse {
        path: path.clone(),
        source,
    })
}

/// Serializes and writes JSON to disk using pretty formatting.
///
/// # Errors
///
/// - [`TokenizerError::FileRead`] when writing fails.
/// - [`TokenizerError::JsonParse`] when serialization fails.
pub fn save_json_file(path: &ArtifactPath, value: &serde_json::Value) -> Result<()> {
    let serialized =
        serde_json::to_string_pretty(value).map_err(|source| TokenizerError::JsonParse {
            path: path.clone(),
            source,
        })?;
    std::fs::write(path, serialized).map_err(|source| TokenizerError::FileRead {
        path: path.clone(),
        source,
    })
}
