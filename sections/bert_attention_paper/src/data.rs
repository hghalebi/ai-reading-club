//! Dataset loaders for the tutorial corpora.

use crate::error::{PaperError, Result};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

/// A document represented as a list of already segmented sentences.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnlabeledDocument {
    pub sentences: Vec<Vec<String>>,
}

/// A dependency parsing example.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DependencySentence {
    pub words: Vec<String>,
    pub heads: Vec<usize>,
    pub relations: Vec<String>,
}

/// A coreference mention category.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MentionType {
    Pronoun,
    Proper,
    Nominal,
}

/// Optional agreement-style attributes used by the rule-based coreference baseline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MorphValue {
    Unknown,
    Singular,
    Plural,
    Masculine,
    Feminine,
    Neuter,
    First,
    Second,
    Third,
}

/// A mention annotation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoreferenceMention {
    pub start: usize,
    pub end: usize,
    pub head: usize,
    pub cluster_id: usize,
    pub mention_type: MentionType,
    #[serde(default = "unknown_morph")]
    pub number: MorphValue,
    #[serde(default = "unknown_morph")]
    pub gender: MorphValue,
    #[serde(default = "unknown_morph")]
    pub person: MorphValue,
}

/// A document with explicit coreference mentions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoreferenceDocument {
    pub words: Vec<String>,
    pub mentions: Vec<CoreferenceMention>,
}

fn unknown_morph() -> MorphValue {
    MorphValue::Unknown
}

/// Reads an unlabeled corpus where each non-empty line is a sentence and empty lines separate documents.
pub fn read_unlabeled_documents(path: impl AsRef<Path>) -> Result<Vec<UnlabeledDocument>> {
    let path = path.as_ref();
    let file = File::open(path).map_err(|source| PaperError::io(path, source))?;
    let reader = BufReader::new(file);

    let mut documents = Vec::new();
    let mut current_document = Vec::new();
    for line in reader.lines() {
        let line = line.map_err(|source| PaperError::io(path, source))?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            if !current_document.is_empty() {
                documents.push(UnlabeledDocument {
                    sentences: current_document,
                });
                current_document = Vec::new();
            }
            continue;
        }

        let words = trimmed
            .split_whitespace()
            .map(ToOwned::to_owned)
            .collect::<Vec<_>>();
        if !words.is_empty() {
            current_document.push(words);
        }
    }

    if !current_document.is_empty() {
        documents.push(UnlabeledDocument {
            sentences: current_document,
        });
    }

    if documents.is_empty() {
        return Err(PaperError::EmptyDataset {
            path: path.to_path_buf(),
        });
    }
    Ok(documents)
}

/// Reads a simple dependency dataset.
///
/// Each non-empty line is:
/// `WORD<TAB>HEAD<TAB>RELATION`
///
/// Empty lines separate sentences. `HEAD` is 1-based and `0` denotes root.
pub fn read_dependency_sentences(path: impl AsRef<Path>) -> Result<Vec<DependencySentence>> {
    let path = path.as_ref();
    let file = File::open(path).map_err(|source| PaperError::io(path, source))?;
    let reader = BufReader::new(file);

    let mut sentences = Vec::new();
    let mut words = Vec::new();
    let mut heads = Vec::new();
    let mut relations = Vec::new();

    for (index, line) in reader.lines().enumerate() {
        let line_number = index + 1;
        let line = line.map_err(|source| PaperError::io(path, source))?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            if !words.is_empty() {
                sentences.push(DependencySentence {
                    words: std::mem::take(&mut words),
                    heads: std::mem::take(&mut heads),
                    relations: std::mem::take(&mut relations),
                });
            }
            continue;
        }

        let parts = trimmed.split('\t').collect::<Vec<_>>();
        if parts.len() != 3 {
            return Err(PaperError::InvalidDependencyLine {
                path: path.to_path_buf(),
                line_number,
                line,
            });
        }

        words.push(parts[0].to_owned());
        let head = parts[1]
            .parse::<usize>()
            .map_err(|source| PaperError::parse_int(path, parts[1], source))?;
        heads.push(head);
        relations.push(parts[2].to_owned());
    }

    if !words.is_empty() {
        sentences.push(DependencySentence {
            words,
            heads,
            relations,
        });
    }

    if sentences.is_empty() {
        return Err(PaperError::EmptyDataset {
            path: path.to_path_buf(),
        });
    }
    Ok(sentences)
}

/// Reads a JSONL coreference dataset.
pub fn read_coreference_documents(path: impl AsRef<Path>) -> Result<Vec<CoreferenceDocument>> {
    let path = path.as_ref();
    let file = File::open(path).map_err(|source| PaperError::io(path, source))?;
    let reader = BufReader::new(file);

    let mut docs = Vec::new();
    for line in reader.lines() {
        let line = line.map_err(|source| PaperError::io(path, source))?;
        if line.trim().is_empty() {
            continue;
        }

        let doc = serde_json::from_str::<CoreferenceDocument>(&line)
            .map_err(|source| PaperError::json(path, source))?;
        validate_coreference_document(path, &doc)?;
        docs.push(doc);
    }

    if docs.is_empty() {
        return Err(PaperError::EmptyDataset {
            path: path.to_path_buf(),
        });
    }
    Ok(docs)
}

/// Pairs consecutive sentences to mimic the paper's pretraining-style two-segment input.
pub fn pair_consecutive_sentences(
    documents: &[UnlabeledDocument],
) -> Vec<(Vec<String>, Vec<String>)> {
    let mut pairs = Vec::new();
    for document in documents {
        for chunk in document.sentences.chunks(2) {
            let left = chunk.first().cloned().unwrap_or_default();
            let right = chunk.get(1).cloned().unwrap_or_default();
            pairs.push((left, right));
        }
    }
    pairs
}

fn validate_coreference_document(path: &Path, doc: &CoreferenceDocument) -> Result<()> {
    if doc.words.is_empty() {
        return Err(PaperError::InvalidCoreferenceDocument {
            path: path.to_path_buf(),
            reason: "documents must contain at least one word".to_owned(),
        });
    }

    for mention in &doc.mentions {
        if mention.start >= mention.end || mention.end > doc.words.len() {
            return Err(PaperError::InvalidCoreferenceDocument {
                path: path.to_path_buf(),
                reason: format!(
                    "mention span [{}, {}) is out of bounds for {} words",
                    mention.start,
                    mention.end,
                    doc.words.len()
                ),
            });
        }
        if mention.head < mention.start || mention.head >= mention.end {
            return Err(PaperError::InvalidCoreferenceDocument {
                path: path.to_path_buf(),
                reason: format!(
                    "mention head {} must lie inside span [{}, {})",
                    mention.head, mention.start, mention.end
                ),
            });
        }
    }
    Ok(())
}

/// Returns a crate-relative path inside the tutorial `data/` directory.
pub fn dataset_path(relative: impl AsRef<Path>) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("data")
        .join(relative)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sentence_pairing_keeps_odd_trailing_sentence() {
        let docs = vec![UnlabeledDocument {
            sentences: vec![
                vec!["A".to_owned()],
                vec!["B".to_owned()],
                vec!["C".to_owned()],
            ],
        }];

        let pairs = pair_consecutive_sentences(&docs);
        assert_eq!(
            pairs,
            vec![
                (vec!["A".to_owned()], vec!["B".to_owned()]),
                (vec!["C".to_owned()], Vec::new())
            ]
        );
    }
}
