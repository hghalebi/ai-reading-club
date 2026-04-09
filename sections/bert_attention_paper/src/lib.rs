//! Pedagogical Rust reimplementation of the analysis pipeline from
//! "What Does BERT Look At? An Analysis of BERT's Attention".
//!
//! The crate is organized around small, composable modules:
//! - [`data`] loads the tutorial datasets.
//! - [`tokenization`] turns words into BERT tokens while preserving alignment.
//! - [`bert`] runs a minimal BERT encoder and exposes per-layer attention maps.
//! - [`attention`] contains reusable transformations on attention maps.
//! - [`analysis`] implements the paper's analyses on top of those primitives.
//! - [`probe`] trains the paper's lightweight attention probes.

pub mod analysis;
pub mod attention;
pub mod bert;
pub mod data;
pub mod error;
pub mod probe;
pub mod runtime;
pub mod tokenization;

pub use error::{PaperError, Result};

use std::path::PathBuf;

/// Default Hugging Face repository for the paper-aligned model.
pub const DEFAULT_MODEL_REPO: &str = "google-bert/bert-base-uncased";
/// Default maximum input length used throughout the paper.
pub const DEFAULT_MAX_SEQUENCE_LENGTH: usize = 128;

/// Returns the crate-local `data/` directory.
pub fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data")
}

/// Initializes the local tracing subscriber once.
pub fn init_tracing() {
    use tracing_subscriber::EnvFilter;

    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("bert_attention_paper=info"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .compact()
        .try_init();
}
