//! Error types for the BERT attention paper reimplementation.

use std::num::{ParseFloatError, ParseIntError};
use std::path::PathBuf;

/// The crate-wide result type.
pub type Result<T> = std::result::Result<T, PaperError>;

/// Strongly typed error variants used across loaders, model execution, and analysis.
#[derive(Debug, thiserror::Error)]
pub enum PaperError {
    #[error("I/O failure at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to parse JSON at {path}: {source}")]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },

    #[error("tokenizer failure: {0}")]
    Tokenizer(#[from] tokenizers::Error),

    #[error("tensor operation failed: {0}")]
    Candle(#[from] candle_core::Error),

    #[error("failed to access Hugging Face Hub: {0}")]
    Hub(#[from] hf_hub::api::sync::ApiError),

    #[error("invalid dependency line {line_number} in {path}: {line}")]
    InvalidDependencyLine {
        path: PathBuf,
        line_number: usize,
        line: String,
    },

    #[error("invalid coreference document in {path}: {reason}")]
    InvalidCoreferenceDocument { path: PathBuf, reason: String },

    #[error("failed to parse integer '{value}' in {path}: {source}")]
    ParseInt {
        path: PathBuf,
        value: String,
        #[source]
        source: ParseIntError,
    },

    #[error("failed to parse float '{value}' in {path}: {source}")]
    ParseFloat {
        path: PathBuf,
        value: String,
        #[source]
        source: ParseFloatError,
    },

    #[error("sequence length {length} exceeds the configured limit {maximum}")]
    SequenceTooLong { length: usize, maximum: usize },

    #[error("required special token '{token}' is missing from the tokenizer")]
    MissingSpecialToken { token: &'static str },

    #[error("dataset at {path} was empty")]
    EmptyDataset { path: PathBuf },

    #[error("analysis precondition failed: {reason}")]
    InvalidAnalysis { reason: String },
}

impl PaperError {
    /// Creates an I/O error variant with the provided path context.
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }

    /// Creates a JSON error variant with the provided path context.
    pub fn json(path: impl Into<PathBuf>, source: serde_json::Error) -> Self {
        Self::Json {
            path: path.into(),
            source,
        }
    }

    /// Creates a parse-int error variant with the provided path context.
    pub fn parse_int(
        path: impl Into<PathBuf>,
        value: impl Into<String>,
        source: ParseIntError,
    ) -> Self {
        Self::ParseInt {
            path: path.into(),
            value: value.into(),
            source,
        }
    }

    /// Creates a parse-float error variant with the provided path context.
    pub fn parse_float(
        path: impl Into<PathBuf>,
        value: impl Into<String>,
        source: ParseFloatError,
    ) -> Self {
        Self::ParseFloat {
            path: path.into(),
            value: value.into(),
            source,
        }
    }
}
