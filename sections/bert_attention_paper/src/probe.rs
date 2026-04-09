//! Lightweight attention probes inspired by section 5 of the paper.

use crate::attention::{AttentionTensor, WordAttentionExample};
use crate::data::DependencySentence;
use crate::error::{PaperError, Result};
use serde::Serialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Comparison table for the probe experiments.
#[derive(Debug, Clone, Serialize)]
pub struct ProbeSummary {
    pub right_branching: f64,
    pub distance_and_glove: f64,
    pub attention_only: f64,
    pub attention_and_glove: f64,
}

/// Fixed word embeddings loaded from a GloVe-style file.
#[derive(Debug, Clone)]
pub struct GloveStore {
    dimension: usize,
    vectors: HashMap<String, Vec<f32>>,
    unknown: Vec<f32>,
}

impl GloveStore {
    /// Loads a GloVe-format text file.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path).map_err(|source| PaperError::io(path, source))?;
        let reader = BufReader::new(file);

        let mut dimension = None;
        let mut vectors = HashMap::new();
        for line in reader.lines() {
            let line = line.map_err(|source| PaperError::io(path, source))?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let parts = trimmed.split_whitespace().collect::<Vec<_>>();
            if parts.len() < 2 {
                return Err(PaperError::InvalidAnalysis {
                    reason: format!("invalid GloVe line in {}", path.display()),
                });
            }
            let word = parts[0].to_owned();
            let values = parts[1..]
                .iter()
                .map(|value| {
                    value
                        .parse::<f32>()
                        .map_err(|source| PaperError::parse_float(path, *value, source))
                })
                .collect::<Result<Vec<_>>>()?;
            match dimension {
                Some(existing) if existing != values.len() => {
                    return Err(PaperError::InvalidAnalysis {
                        reason: format!(
                            "GloVe dimension mismatch in {}: expected {}, found {}",
                            path.display(),
                            existing,
                            values.len()
                        ),
                    });
                }
                None => dimension = Some(values.len()),
                _ => {}
            }
            vectors.insert(word, values);
        }

        let dimension = dimension.ok_or_else(|| PaperError::EmptyDataset {
            path: path.to_path_buf(),
        })?;
        Ok(Self {
            dimension,
            vectors,
            unknown: vec![0.0; dimension],
        })
    }

    /// Returns the embedding dimension.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    fn embedding(&self, word: &str) -> &[f32] {
        self.vectors
            .get(&word.to_lowercase())
            .map(|vector| vector.as_slice())
            .unwrap_or(self.unknown.as_slice())
    }
}

/// Section-5 attention-only probe.
#[derive(Debug, Clone)]
pub struct AttentionOnlyProbe {
    forward_weights: Vec<f32>,
    backward_weights: Vec<f32>,
}

impl AttentionOnlyProbe {
    /// Creates a zero-initialized probe.
    pub fn new(head_count: usize) -> Self {
        Self {
            forward_weights: vec![0.0; head_count],
            backward_weights: vec![0.0; head_count],
        }
    }

    /// Trains the probe with simple sentence-level SGD.
    pub fn train(
        &mut self,
        sentences: &[DependencySentence],
        attentions: &[WordAttentionExample],
        epochs: usize,
        learning_rate: f32,
    ) -> Result<()> {
        ensure_alignment(sentences, attentions)?;
        let head_count = flattened_head_count(&attentions[0].word_attentions)?;

        if self.forward_weights.len() != head_count || self.backward_weights.len() != head_count {
            return Err(PaperError::InvalidAnalysis {
                reason: format!(
                    "probe expected {} heads but model exposes {}",
                    self.forward_weights.len(),
                    head_count
                ),
            });
        }

        for _ in 0..epochs {
            for (sentence, example) in sentences.iter().zip(attentions.iter()) {
                for dependent in 0..sentence.words.len() {
                    let gold_head = sentence.heads[dependent];
                    if gold_head == 0 {
                        continue;
                    }

                    let candidates = (0..sentence.words.len())
                        .filter(|candidate| *candidate != dependent)
                        .collect::<Vec<_>>();
                    let mut scores = Vec::with_capacity(candidates.len());
                    let mut features = Vec::with_capacity(candidates.len());
                    for &candidate in &candidates {
                        let (forward, backward) = attention_pair_features(
                            &example.word_attentions,
                            candidate,
                            dependent,
                        )?;
                        let score = dot(&self.forward_weights, &forward)
                            + dot(&self.backward_weights, &backward);
                        scores.push(score);
                        features.push((forward, backward));
                    }
                    let probabilities = softmax(&scores);
                    for (candidate_index, &candidate) in candidates.iter().enumerate() {
                        let target = if candidate == gold_head - 1 { 1.0 } else { 0.0 };
                        let error = probabilities[candidate_index] - target;
                        let (ref forward, ref backward) = features[candidate_index];
                        for head_index in 0..head_count {
                            self.forward_weights[head_index] -=
                                learning_rate * error * forward[head_index];
                            self.backward_weights[head_index] -=
                                learning_rate * error * backward[head_index];
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Evaluates unlabeled attachment score on the given split.
    pub fn evaluate(
        &self,
        sentences: &[DependencySentence],
        attentions: &[WordAttentionExample],
    ) -> Result<f64> {
        ensure_alignment(sentences, attentions)?;
        let mut correct = 0_usize;
        let mut total = 0_usize;

        for (sentence, example) in sentences.iter().zip(attentions.iter()) {
            for dependent in 0..sentence.words.len() {
                let gold_head = sentence.heads[dependent];
                if gold_head == 0 {
                    continue;
                }
                total += 1;
                let predicted = self.predict_head(example, dependent)?;
                if predicted == Some(gold_head - 1) {
                    correct += 1;
                }
            }
        }

        Ok(ratio(correct, total))
    }

    fn predict_head(
        &self,
        example: &WordAttentionExample,
        dependent: usize,
    ) -> Result<Option<usize>> {
        let mut best_candidate = None;
        let mut best_score = f32::NEG_INFINITY;
        for candidate in 0..example.words.len() {
            if candidate == dependent {
                continue;
            }
            let (forward, backward) =
                attention_pair_features(&example.word_attentions, candidate, dependent)?;
            let score =
                dot(&self.forward_weights, &forward) + dot(&self.backward_weights, &backward);
            if score > best_score {
                best_score = score;
                best_candidate = Some(candidate);
            }
        }
        Ok(best_candidate)
    }
}

/// Section-5 attention-and-words probe.
#[derive(Debug, Clone)]
pub struct AttentionAndWordProbe {
    forward_weights: Vec<Vec<f32>>,
    backward_weights: Vec<Vec<f32>>,
}

impl AttentionAndWordProbe {
    /// Creates a zero-initialized probe.
    pub fn new(head_count: usize, feature_dimension: usize) -> Self {
        Self {
            forward_weights: vec![vec![0.0; feature_dimension]; head_count],
            backward_weights: vec![vec![0.0; feature_dimension]; head_count],
        }
    }

    /// Trains the probe with sentence-level SGD.
    pub fn train(
        &mut self,
        sentences: &[DependencySentence],
        attentions: &[WordAttentionExample],
        glove: &GloveStore,
        epochs: usize,
        learning_rate: f32,
    ) -> Result<()> {
        ensure_alignment(sentences, attentions)?;
        let head_count = flattened_head_count(&attentions[0].word_attentions)?;
        let feature_dimension = glove.dimension() * 2;
        if self.forward_weights.len() != head_count
            || self.forward_weights[0].len() != feature_dimension
        {
            return Err(PaperError::InvalidAnalysis {
                reason: "attention-and-word probe dimension mismatch".to_owned(),
            });
        }

        for _ in 0..epochs {
            for (sentence, example) in sentences.iter().zip(attentions.iter()) {
                for dependent in 0..sentence.words.len() {
                    let gold_head = sentence.heads[dependent];
                    if gold_head == 0 {
                        continue;
                    }

                    let candidates = (0..sentence.words.len())
                        .filter(|candidate| *candidate != dependent)
                        .collect::<Vec<_>>();
                    let mut features = Vec::with_capacity(candidates.len());
                    let mut scores = Vec::with_capacity(candidates.len());

                    for &candidate in &candidates {
                        let pair_features = pair_embedding(
                            glove,
                            &sentence.words[dependent],
                            &sentence.words[candidate],
                        );
                        let (forward, backward) = attention_pair_features(
                            &example.word_attentions,
                            candidate,
                            dependent,
                        )?;
                        let mut score = 0.0_f32;
                        for head_index in 0..head_count {
                            score += dot(&self.forward_weights[head_index], &pair_features)
                                * forward[head_index];
                            score += dot(&self.backward_weights[head_index], &pair_features)
                                * backward[head_index];
                        }
                        features.push((pair_features, forward, backward));
                        scores.push(score);
                    }

                    let probabilities = softmax(&scores);
                    for (candidate_index, &candidate) in candidates.iter().enumerate() {
                        let target = if candidate == gold_head - 1 { 1.0 } else { 0.0 };
                        let error = probabilities[candidate_index] - target;
                        let (ref pair_features, ref forward, ref backward) =
                            features[candidate_index];
                        for head_index in 0..head_count {
                            for (feature_index, feature_value) in pair_features
                                .iter()
                                .copied()
                                .enumerate()
                                .take(feature_dimension)
                            {
                                self.forward_weights[head_index][feature_index] -=
                                    learning_rate * error * forward[head_index] * feature_value;
                                self.backward_weights[head_index][feature_index] -=
                                    learning_rate * error * backward[head_index] * feature_value;
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Evaluates unlabeled attachment score.
    pub fn evaluate(
        &self,
        sentences: &[DependencySentence],
        attentions: &[WordAttentionExample],
        glove: &GloveStore,
    ) -> Result<f64> {
        ensure_alignment(sentences, attentions)?;
        let mut correct = 0_usize;
        let mut total = 0_usize;

        for (sentence, example) in sentences.iter().zip(attentions.iter()) {
            for dependent in 0..sentence.words.len() {
                let gold_head = sentence.heads[dependent];
                if gold_head == 0 {
                    continue;
                }
                total += 1;
                let predicted = self.predict_head(example, glove, &sentence.words, dependent)?;
                if predicted == Some(gold_head - 1) {
                    correct += 1;
                }
            }
        }
        Ok(ratio(correct, total))
    }

    fn predict_head(
        &self,
        example: &WordAttentionExample,
        glove: &GloveStore,
        words: &[String],
        dependent: usize,
    ) -> Result<Option<usize>> {
        let mut best_candidate = None;
        let mut best_score = f32::NEG_INFINITY;
        for candidate in 0..words.len() {
            if candidate == dependent {
                continue;
            }
            let pair_features = pair_embedding(glove, &words[dependent], &words[candidate]);
            let (forward, backward) =
                attention_pair_features(&example.word_attentions, candidate, dependent)?;
            let mut score = 0.0_f32;
            for head_index in 0..self.forward_weights.len() {
                score +=
                    dot(&self.forward_weights[head_index], &pair_features) * forward[head_index];
                score +=
                    dot(&self.backward_weights[head_index], &pair_features) * backward[head_index];
            }
            if score > best_score {
                best_score = score;
                best_candidate = Some(candidate);
            }
        }
        Ok(best_candidate)
    }
}

/// Section-5 style comparison using compact tutorial-friendly baselines.
pub fn run_probe_comparison(
    train_sentences: &[DependencySentence],
    train_attentions: &[WordAttentionExample],
    dev_sentences: &[DependencySentence],
    dev_attentions: &[WordAttentionExample],
    glove: &GloveStore,
) -> Result<ProbeSummary> {
    ensure_alignment(train_sentences, train_attentions)?;
    ensure_alignment(dev_sentences, dev_attentions)?;

    let head_count = flattened_head_count(&train_attentions[0].word_attentions)?;
    let right_branching = right_branching_baseline(dev_sentences);
    let distance_and_glove = distance_and_glove_baseline(train_sentences, dev_sentences, glove)?;

    let mut attention_only = AttentionOnlyProbe::new(head_count);
    attention_only.train(train_sentences, train_attentions, 40, 0.05)?;
    let attention_only_score = attention_only.evaluate(dev_sentences, dev_attentions)?;

    let mut attention_and_glove = AttentionAndWordProbe::new(head_count, glove.dimension() * 2);
    attention_and_glove.train(train_sentences, train_attentions, glove, 40, 0.02)?;
    let attention_and_glove_score =
        attention_and_glove.evaluate(dev_sentences, dev_attentions, glove)?;

    Ok(ProbeSummary {
        right_branching,
        distance_and_glove,
        attention_only: attention_only_score,
        attention_and_glove: attention_and_glove_score,
    })
}

fn right_branching_baseline(sentences: &[DependencySentence]) -> f64 {
    let mut correct = 0_usize;
    let mut total = 0_usize;
    for sentence in sentences {
        for dependent in 0..sentence.words.len() {
            let gold_head = sentence.heads[dependent];
            if gold_head == 0 || dependent + 1 >= sentence.words.len() {
                continue;
            }
            total += 1;
            if dependent + 1 == gold_head - 1 {
                correct += 1;
            }
        }
    }
    ratio(correct, total)
}

fn distance_and_glove_baseline(
    train_sentences: &[DependencySentence],
    dev_sentences: &[DependencySentence],
    glove: &GloveStore,
) -> Result<f64> {
    let feature_dim = glove.dimension() * 2 + 5;
    let mut weights = vec![0.0_f32; feature_dim];
    let learning_rate = 0.05_f32;

    for _ in 0..50 {
        for sentence in train_sentences {
            for dependent in 0..sentence.words.len() {
                let gold_head = sentence.heads[dependent];
                if gold_head == 0 {
                    continue;
                }
                let candidates = (0..sentence.words.len())
                    .filter(|candidate| *candidate != dependent)
                    .collect::<Vec<_>>();
                let features = candidates
                    .iter()
                    .map(|&candidate| {
                        baseline_features(glove, &sentence.words, dependent, candidate)
                    })
                    .collect::<Vec<_>>();
                let scores = features
                    .iter()
                    .map(|feature| dot(&weights, feature))
                    .collect::<Vec<_>>();
                let probabilities = softmax(&scores);
                for (candidate_index, &candidate) in candidates.iter().enumerate() {
                    let target = if candidate == gold_head - 1 { 1.0 } else { 0.0 };
                    let error = probabilities[candidate_index] - target;
                    for feature_index in 0..feature_dim {
                        weights[feature_index] -=
                            learning_rate * error * features[candidate_index][feature_index];
                    }
                }
            }
        }
    }

    let mut correct = 0_usize;
    let mut total = 0_usize;
    for sentence in dev_sentences {
        for dependent in 0..sentence.words.len() {
            let gold_head = sentence.heads[dependent];
            if gold_head == 0 {
                continue;
            }
            total += 1;
            let mut best_candidate = None;
            let mut best_score = f32::NEG_INFINITY;
            for candidate in 0..sentence.words.len() {
                if candidate == dependent {
                    continue;
                }
                let feature = baseline_features(glove, &sentence.words, dependent, candidate);
                let score = dot(&weights, &feature);
                if score > best_score {
                    best_score = score;
                    best_candidate = Some(candidate);
                }
            }
            if best_candidate == Some(gold_head - 1) {
                correct += 1;
            }
        }
    }
    Ok(ratio(correct, total))
}

fn baseline_features(
    glove: &GloveStore,
    words: &[String],
    dependent: usize,
    candidate: usize,
) -> Vec<f32> {
    let mut features = pair_embedding(glove, &words[dependent], &words[candidate]);
    let distance = candidate as isize - dependent as isize;
    features.push(distance as f32);
    features.push((distance > 0) as u8 as f32);
    features.push((distance < 0) as u8 as f32);
    features.push((distance.abs() == 1) as u8 as f32);
    features.push((distance.abs() == 2) as u8 as f32);
    features
}

fn pair_embedding(glove: &GloveStore, dependent: &str, candidate: &str) -> Vec<f32> {
    let mut features = glove.embedding(dependent).to_vec();
    features.extend_from_slice(glove.embedding(candidate));
    features
}

fn ensure_alignment(
    sentences: &[DependencySentence],
    attentions: &[WordAttentionExample],
) -> Result<()> {
    if sentences.len() != attentions.len() {
        return Err(PaperError::InvalidAnalysis {
            reason: format!(
                "dependency corpus and attention tensors differ in length ({} vs {})",
                sentences.len(),
                attentions.len()
            ),
        });
    }
    for (sentence, attention) in sentences.iter().zip(attentions.iter()) {
        if sentence.words != attention.words {
            return Err(PaperError::InvalidAnalysis {
                reason: format!(
                    "word mismatch between dependency example {:?} and attention example {:?}",
                    sentence.words, attention.words
                ),
            });
        }
    }
    Ok(())
}

fn flattened_head_count(attentions: &AttentionTensor) -> Result<usize> {
    let layers = attentions.len();
    let heads = attentions
        .first()
        .ok_or_else(|| PaperError::InvalidAnalysis {
            reason: "attention tensor had zero layers".to_owned(),
        })?
        .len();
    Ok(layers * heads)
}

fn attention_pair_features(
    attentions: &AttentionTensor,
    candidate: usize,
    dependent: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let head_count = flattened_head_count(attentions)?;
    let mut forward = vec![0.0_f32; head_count];
    let mut backward = vec![0.0_f32; head_count];

    let heads_per_layer = attentions[0].len();
    for (layer_index, layer) in attentions.iter().enumerate() {
        for (head_index, head) in layer.iter().enumerate() {
            let flat_index = layer_index * heads_per_layer + head_index;
            forward[flat_index] = head[candidate][dependent];
            backward[flat_index] = head[dependent][candidate];
        }
    }
    Ok((forward, backward))
}

fn softmax(scores: &[f32]) -> Vec<f32> {
    let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps = scores
        .iter()
        .map(|score| (*score - max_score).exp())
        .collect::<Vec<_>>();
    let total = exps.iter().sum::<f32>().max(f32::EPSILON);
    exps.into_iter().map(|value| value / total).collect()
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .copied()
        .zip(right.iter().copied())
        .map(|(lhs, rhs)| lhs * rhs)
        .sum::<f32>()
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}
