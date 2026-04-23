//! Paper-specific analyses on top of extracted attention maps.

use crate::attention::{
    argmax_excluding, average_entropy, jensen_shannon_divergence, AttentionExample,
    AttentionTensor, WordAttentionExample,
};
use crate::data::{CoreferenceDocument, DependencySentence, MentionType, MorphValue};
use crate::error::{PaperError, Result};
use nalgebra::{DMatrix, SymmetricEigen};
use serde::Serialize;
use std::collections::{BTreeMap, HashMap};

/// Direction for interpreting an attention head in syntax analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AttentionDirection {
    CandidateToDependent,
    DependentToCandidate,
}

/// Identifies a particular attention head.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct HeadLocator {
    pub layer: usize,
    pub head: usize,
}

impl HeadLocator {
    /// Returns a human-readable `Lx-Hy` label.
    pub fn label(self) -> String {
        format!("L{}-H{}", self.layer + 1, self.head + 1)
    }
}

/// Per-head surface-level statistics.
#[derive(Debug, Clone, Serialize)]
pub struct SurfaceHeadStats {
    pub head: HeadLocator,
    pub previous_token_attention: f64,
    pub self_attention: f64,
    pub next_token_attention: f64,
    pub cls_attention: f64,
    pub sep_attention: f64,
    pub punctuation_attention: f64,
    pub other_attention: f64,
    pub sep_to_sep_attention: f64,
    pub other_to_sep_attention: f64,
    pub average_entropy: f64,
}

/// Summary of the paper's section-3 style analyses.
#[derive(Debug, Clone, Serialize)]
pub struct SurfaceSummary {
    pub heads: Vec<SurfaceHeadStats>,
}

/// Best-performing head for a dependency relation.
#[derive(Debug, Clone, Serialize)]
pub struct RelationSummary {
    pub relation: String,
    pub head: HeadLocator,
    pub direction: AttentionDirection,
    pub accuracy: f64,
    pub baseline_offset: isize,
    pub baseline_accuracy: f64,
    pub instances: usize,
}

/// Dependency parsing summary.
#[derive(Debug, Clone, Serialize)]
pub struct SyntaxSummary {
    pub best_overall: RelationSummary,
    pub relations: Vec<RelationSummary>,
}

/// Accuracy by mention type.
#[derive(Debug, Clone, Serialize)]
pub struct MentionTypeScore {
    pub mention_type: MentionType,
    pub accuracy: f64,
    pub instances: usize,
}

/// Coreference summary.
#[derive(Debug, Clone, Serialize)]
pub struct CoreferenceSummary {
    pub best_head: RelationSummary,
    pub nearest_baseline: f64,
    pub head_match_baseline: f64,
    pub rule_based_baseline: f64,
    pub by_type: Vec<MentionTypeScore>,
}

/// Coordinates for a clustered attention head.
#[derive(Debug, Clone, Serialize)]
pub struct HeadClusterPoint {
    pub head: HeadLocator,
    pub x: f64,
    pub y: f64,
}

/// Result of the section-6 clustering analysis.
#[derive(Debug, Clone, Serialize)]
pub struct ClusteringSummary {
    pub distance_matrix: Vec<Vec<f64>>,
    pub points: Vec<HeadClusterPoint>,
}

/// Reproduces the paper's surface-level attention statistics.
pub fn analyze_surface_patterns(examples: &[AttentionExample]) -> Result<SurfaceSummary> {
    let Some(first) = examples.first() else {
        return Err(PaperError::InvalidAnalysis {
            reason: "surface analysis requires at least one attention example".to_owned(),
        });
    };

    let layer_count = first.token_attentions.len();
    let head_count = first.token_attentions[0].len();
    let mut totals = vec![SurfaceAccumulator::default(); layer_count * head_count];

    for example in examples {
        let tokens = &example.encoded.tokens;
        for layer_index in 0..layer_count {
            for head_index in 0..head_count {
                let accumulator =
                    &mut totals[flatten_head_index(head_count, layer_index, head_index)];
                let head = &example.token_attentions[layer_index][head_index];
                accumulator.record(tokens, head);
            }
        }
    }

    let heads = totals
        .into_iter()
        .enumerate()
        .map(|(index, accumulator)| {
            let layer = index / head_count;
            let head = index % head_count;
            SurfaceHeadStats {
                head: HeadLocator { layer, head },
                previous_token_attention: accumulator.previous_attention.average(),
                self_attention: accumulator.self_attention.average(),
                next_token_attention: accumulator.next_attention.average(),
                cls_attention: accumulator.cls_attention.average(),
                sep_attention: accumulator.sep_attention.average(),
                punctuation_attention: accumulator.punctuation_attention.average(),
                other_attention: accumulator.other_attention.average(),
                sep_to_sep_attention: accumulator.sep_to_sep.average(),
                other_to_sep_attention: accumulator.other_to_sep.average(),
                average_entropy: accumulator.entropy.average(),
            }
        })
        .collect();

    Ok(SurfaceSummary { heads })
}

/// Evaluates all attention heads against dependency annotations.
pub fn analyze_dependency_syntax(
    sentences: &[DependencySentence],
    examples: &[WordAttentionExample],
) -> Result<SyntaxSummary> {
    ensure_dependency_alignment(sentences, examples)?;
    let head_shape = attention_shape(&examples[0].word_attentions)?;

    let mut relation_names = BTreeMap::<String, usize>::new();
    for sentence in sentences {
        for relation in &sentence.relations {
            *relation_names.entry(relation.clone()).or_default() += 1;
        }
    }

    let mut relation_summaries = Vec::new();
    for relation in relation_names.keys() {
        relation_summaries.push(best_relation_summary(
            relation, sentences, examples, head_shape, false,
        ));
    }

    let overall_summary = best_relation_summary("__all__", sentences, examples, head_shape, true);

    Ok(SyntaxSummary {
        best_overall: overall_summary,
        relations: relation_summaries,
    })
}

/// Evaluates the paper's coreference antecedent-selection task.
pub fn analyze_coreference(
    documents: &[CoreferenceDocument],
    examples: &[WordAttentionExample],
) -> Result<CoreferenceSummary> {
    ensure_coreference_alignment(documents, examples)?;
    let head_shape = attention_shape(&examples[0].word_attentions)?;

    let mut best_head = RelationSummary {
        relation: "coreference".to_owned(),
        head: HeadLocator { layer: 0, head: 0 },
        direction: AttentionDirection::DependentToCandidate,
        accuracy: f64::MIN,
        baseline_offset: 0,
        baseline_accuracy: 0.0,
        instances: 0,
    };

    let mut type_hits: HashMap<MentionType, usize> = HashMap::new();
    let mut type_totals: HashMap<MentionType, usize> = HashMap::new();

    for layer in 0..head_shape.0 {
        for head in 0..head_shape.1 {
            let (accuracy, hits, totals, instances) =
                coreference_head_accuracy(documents, examples, layer, head);
            if accuracy > best_head.accuracy {
                best_head = RelationSummary {
                    relation: "coreference".to_owned(),
                    head: HeadLocator { layer, head },
                    direction: AttentionDirection::DependentToCandidate,
                    accuracy,
                    baseline_offset: 0,
                    baseline_accuracy: 0.0,
                    instances,
                };
                type_hits = hits;
                type_totals = totals;
            }
        }
    }

    let nearest = evaluate_coref_baseline(documents, CorefBaseline::NearestMention);
    let head_match = evaluate_coref_baseline(documents, CorefBaseline::HeadMatch);
    let rule_based = evaluate_coref_baseline(documents, CorefBaseline::RuleBased);
    let by_type = [
        MentionType::Pronoun,
        MentionType::Proper,
        MentionType::Nominal,
    ]
    .into_iter()
    .map(|mention_type| {
        let hits = *type_hits.get(&mention_type).unwrap_or(&0);
        let total = *type_totals.get(&mention_type).unwrap_or(&0);
        MentionTypeScore {
            mention_type,
            accuracy: ratio(hits, total),
            instances: total,
        }
    })
    .collect::<Vec<_>>();

    Ok(CoreferenceSummary {
        best_head,
        nearest_baseline: nearest,
        head_match_baseline: head_match,
        rule_based_baseline: rule_based,
        by_type,
    })
}

/// Computes average Jensen-Shannon distances between all attention heads and projects them with classical MDS.
pub fn cluster_attention_heads(examples: &[AttentionExample]) -> Result<ClusteringSummary> {
    let Some(first) = examples.first() else {
        return Err(PaperError::InvalidAnalysis {
            reason: "clustering requires at least one attention example".to_owned(),
        });
    };
    let (layer_count, head_count, _, _) = attention_shape(&first.token_attentions)?;
    let flattened_heads = layer_count * head_count;

    let mut distance_matrix = vec![vec![0.0_f64; flattened_heads]; flattened_heads];
    let mut counts = vec![vec![0_usize; flattened_heads]; flattened_heads];

    for example in examples {
        for layer_a in 0..layer_count {
            for head_a in 0..head_count {
                let head_index_a = flatten_head_index(head_count, layer_a, head_a);
                let head_a_matrix = &example.token_attentions[layer_a][head_a];
                for layer_b in layer_a..layer_count {
                    for head_b in 0..head_count {
                        let head_index_b = flatten_head_index(head_count, layer_b, head_b);
                        if head_index_b < head_index_a {
                            continue;
                        }
                        let head_b_matrix = &example.token_attentions[layer_b][head_b];
                        let per_query = head_a_matrix
                            .iter()
                            .zip(head_b_matrix.iter())
                            .map(|(left, right)| jensen_shannon_divergence(left, right))
                            .sum::<f64>();
                        distance_matrix[head_index_a][head_index_b] += per_query;
                        counts[head_index_a][head_index_b] += head_a_matrix.len();
                    }
                }
            }
        }
    }

    for row in 0..flattened_heads {
        for col in row..flattened_heads {
            let average = if counts[row][col] == 0 {
                0.0
            } else {
                distance_matrix[row][col] / counts[row][col] as f64
            };
            distance_matrix[row][col] = average;
            distance_matrix[col][row] = average;
        }
    }

    let points = classical_mds(&distance_matrix)
        .into_iter()
        .enumerate()
        .map(|(index, (x, y))| HeadClusterPoint {
            head: HeadLocator {
                layer: index / head_count,
                head: index % head_count,
            },
            x,
            y,
        })
        .collect::<Vec<_>>();

    Ok(ClusteringSummary {
        distance_matrix,
        points,
    })
}

fn best_relation_summary(
    relation: &str,
    sentences: &[DependencySentence],
    examples: &[WordAttentionExample],
    head_shape: (usize, usize, usize, usize),
    include_all_relations: bool,
) -> RelationSummary {
    let mut best = RelationSummary {
        relation: if include_all_relations {
            "all".to_owned()
        } else {
            relation.to_owned()
        },
        head: HeadLocator { layer: 0, head: 0 },
        direction: AttentionDirection::DependentToCandidate,
        accuracy: f64::MIN,
        baseline_offset: 0,
        baseline_accuracy: 0.0,
        instances: 0,
    };
    let (baseline_offset, baseline_accuracy, baseline_instances) =
        best_fixed_offset(sentences, relation, include_all_relations);

    for layer in 0..head_shape.0 {
        for head in 0..head_shape.1 {
            for direction in [
                AttentionDirection::CandidateToDependent,
                AttentionDirection::DependentToCandidate,
            ] {
                let (accuracy, instances) = syntax_head_accuracy(
                    sentences,
                    examples,
                    layer,
                    head,
                    direction,
                    relation,
                    include_all_relations,
                );
                if accuracy > best.accuracy {
                    best = RelationSummary {
                        relation: if include_all_relations {
                            "all".to_owned()
                        } else {
                            relation.to_owned()
                        },
                        head: HeadLocator { layer, head },
                        direction,
                        accuracy,
                        baseline_offset,
                        baseline_accuracy,
                        instances,
                    };
                }
            }
        }
    }
    if best.instances == 0 {
        best.instances = baseline_instances;
        best.accuracy = 0.0;
    }
    best
}

fn syntax_head_accuracy(
    sentences: &[DependencySentence],
    examples: &[WordAttentionExample],
    layer: usize,
    head: usize,
    direction: AttentionDirection,
    relation: &str,
    include_all_relations: bool,
) -> (f64, usize) {
    let mut correct = 0_usize;
    let mut total = 0_usize;

    for (sentence, example) in sentences.iter().zip(examples.iter()) {
        let head_matrix = &example.word_attentions[layer][head];
        for dependent_index in 0..sentence.words.len() {
            let gold_head = sentence.heads[dependent_index];
            if gold_head == 0 {
                continue;
            }
            if !include_all_relations && sentence.relations[dependent_index] != relation {
                continue;
            }

            let predicted = predict_head_index(head_matrix, dependent_index, direction);
            total += 1;
            if predicted == Some(gold_head - 1) {
                correct += 1;
            }
        }
    }

    (ratio(correct, total), total)
}

fn predict_head_index(
    head_matrix: &[Vec<f32>],
    dependent_index: usize,
    direction: AttentionDirection,
) -> Option<usize> {
    match direction {
        AttentionDirection::DependentToCandidate => {
            argmax_excluding(&head_matrix[dependent_index], dependent_index).map(|(index, _)| index)
        }
        AttentionDirection::CandidateToDependent => {
            let scores = (0..head_matrix.len())
                .map(|candidate| {
                    if candidate == dependent_index {
                        f32::NEG_INFINITY
                    } else {
                        head_matrix[candidate][dependent_index]
                    }
                })
                .collect::<Vec<_>>();
            argmax_excluding(&scores, dependent_index).map(|(index, _)| index)
        }
    }
}

fn best_fixed_offset(
    sentences: &[DependencySentence],
    relation: &str,
    include_all_relations: bool,
) -> (isize, f64, usize) {
    let max_len = sentences
        .iter()
        .map(|sentence| sentence.words.len())
        .max()
        .unwrap_or(1);
    let mut best_offset = 1_isize;
    let mut best_accuracy = f64::MIN;
    let mut total_instances = 0_usize;

    for offset in -((max_len as isize) - 1)..=((max_len as isize) - 1) {
        if offset == 0 {
            continue;
        }
        let mut correct = 0_usize;
        let mut total = 0_usize;
        for sentence in sentences {
            for dependent_index in 0..sentence.words.len() {
                let gold_head = sentence.heads[dependent_index];
                if gold_head == 0 {
                    continue;
                }
                if !include_all_relations && sentence.relations[dependent_index] != relation {
                    continue;
                }

                let predicted = dependent_index as isize + offset;
                total += 1;
                if predicted >= 0
                    && predicted < sentence.words.len() as isize
                    && predicted as usize == gold_head - 1
                {
                    correct += 1;
                }
            }
        }
        if ratio(correct, total) > best_accuracy {
            best_accuracy = ratio(correct, total);
            best_offset = offset;
            total_instances = total;
        }
    }

    (best_offset, best_accuracy.max(0.0), total_instances)
}

fn coreference_head_accuracy(
    documents: &[CoreferenceDocument],
    examples: &[WordAttentionExample],
    layer: usize,
    head: usize,
) -> (
    f64,
    HashMap<MentionType, usize>,
    HashMap<MentionType, usize>,
    usize,
) {
    let mut correct = 0_usize;
    let mut total = 0_usize;
    let mut type_hits = HashMap::new();
    let mut type_totals = HashMap::new();

    for (document, example) in documents.iter().zip(examples.iter()) {
        let head_matrix = &example.word_attentions[layer][head];
        for (mention_index, mention) in document.mentions.iter().enumerate() {
            let antecedents = antecedent_mentions(&document.mentions, mention_index);
            if antecedents.is_empty() {
                continue;
            }
            let mention_head = mention.head;
            let predicted = head_matrix[mention_head]
                .iter()
                .copied()
                .enumerate()
                .take(mention_head)
                .max_by(|(_, left), (_, right)| left.total_cmp(right))
                .map(|(index, _)| index);

            total += 1;
            *type_totals.entry(mention.mention_type).or_insert(0) += 1;
            if let Some(predicted) = predicted {
                if antecedents
                    .iter()
                    .any(|candidate| candidate.head == predicted)
                {
                    correct += 1;
                    *type_hits.entry(mention.mention_type).or_insert(0) += 1;
                }
            }
        }
    }

    (ratio(correct, total), type_hits, type_totals, total)
}

#[derive(Debug, Clone, Copy)]
enum CorefBaseline {
    NearestMention,
    HeadMatch,
    RuleBased,
}

fn evaluate_coref_baseline(documents: &[CoreferenceDocument], baseline: CorefBaseline) -> f64 {
    let mut correct = 0_usize;
    let mut total = 0_usize;

    for document in documents {
        for mention_index in 0..document.mentions.len() {
            let antecedents = antecedent_mentions(&document.mentions, mention_index);
            if antecedents.is_empty() {
                continue;
            }

            let predicted = match baseline {
                CorefBaseline::NearestMention => {
                    nearest_previous_mention(&document.mentions, mention_index)
                }
                CorefBaseline::HeadMatch => nearest_head_match(document, mention_index),
                CorefBaseline::RuleBased => rule_based_antecedent(document, mention_index),
            };
            total += 1;
            if let Some(predicted) = predicted {
                if antecedents
                    .iter()
                    .any(|candidate| candidate.head == predicted.head)
                {
                    correct += 1;
                }
            }
        }
    }

    ratio(correct, total)
}

fn nearest_previous_mention(
    mentions: &[crate::data::CoreferenceMention],
    mention_index: usize,
) -> Option<&crate::data::CoreferenceMention> {
    mention_index
        .checked_sub(1)
        .and_then(|index| mentions.get(index))
}

fn nearest_head_match(
    document: &CoreferenceDocument,
    mention_index: usize,
) -> Option<&crate::data::CoreferenceMention> {
    let current = &document.mentions[mention_index];
    let current_head_word = document.words[current.head].to_lowercase();
    document.mentions[..mention_index]
        .iter()
        .rev()
        .find(|candidate| document.words[candidate.head].eq_ignore_ascii_case(&current_head_word))
}

fn rule_based_antecedent(
    document: &CoreferenceDocument,
    mention_index: usize,
) -> Option<&crate::data::CoreferenceMention> {
    let current = &document.mentions[mention_index];
    let previous = &document.mentions[..mention_index];
    let current_text = mention_text(document, current).to_lowercase();
    let current_head_word = document.words[current.head].to_lowercase();

    previous
        .iter()
        .rev()
        .find(|candidate| mention_text(document, candidate).eq_ignore_ascii_case(&current_text))
        .or_else(|| {
            previous.iter().rev().find(|candidate| {
                document.words[candidate.head].eq_ignore_ascii_case(&current_head_word)
            })
        })
        .or_else(|| {
            previous
                .iter()
                .rev()
                .find(|candidate| morph_compatible(current, candidate))
        })
        .or_else(|| previous.last())
}

fn morph_compatible(
    left: &crate::data::CoreferenceMention,
    right: &crate::data::CoreferenceMention,
) -> bool {
    matches_or_unknown(left.number, right.number)
        && matches_or_unknown(left.gender, right.gender)
        && matches_or_unknown(left.person, right.person)
}

fn matches_or_unknown(left: MorphValue, right: MorphValue) -> bool {
    left == MorphValue::Unknown || right == MorphValue::Unknown || left == right
}

fn mention_text(
    document: &CoreferenceDocument,
    mention: &crate::data::CoreferenceMention,
) -> String {
    document.words[mention.start..mention.end].join(" ")
}

fn antecedent_mentions(
    mentions: &[crate::data::CoreferenceMention],
    mention_index: usize,
) -> Vec<&crate::data::CoreferenceMention> {
    let current = &mentions[mention_index];
    mentions[..mention_index]
        .iter()
        .filter(|candidate| candidate.cluster_id == current.cluster_id)
        .collect::<Vec<_>>()
}

fn classical_mds(distance_matrix: &[Vec<f64>]) -> Vec<(f64, f64)> {
    let n = distance_matrix.len();
    let distances_squared = DMatrix::from_fn(n, n, |row, col| {
        let value = distance_matrix[row][col];
        value * value
    });
    let identity = DMatrix::<f64>::identity(n, n);
    let centering = identity - DMatrix::<f64>::from_element(n, n, 1.0 / n as f64);
    let kernel = -0.5 * &centering * distances_squared * centering;
    let decomposition = SymmetricEigen::new(kernel);

    let mut eigenpairs = decomposition
        .eigenvalues
        .iter()
        .copied()
        .enumerate()
        .collect::<Vec<_>>();
    eigenpairs.sort_by(|(_, left), (_, right)| right.total_cmp(left));

    let first = eigenpairs.first().copied().unwrap_or((0, 0.0));
    let second = eigenpairs.get(1).copied().unwrap_or((0, 0.0));
    let first_scale = first.1.max(0.0).sqrt();
    let second_scale = second.1.max(0.0).sqrt();

    (0..n)
        .map(|row| {
            let x = decomposition.eigenvectors[(row, first.0)] * first_scale;
            let y = decomposition.eigenvectors[(row, second.0)] * second_scale;
            (x, y)
        })
        .collect()
}

fn ensure_dependency_alignment(
    sentences: &[DependencySentence],
    examples: &[WordAttentionExample],
) -> Result<()> {
    if sentences.len() != examples.len() {
        return Err(PaperError::InvalidAnalysis {
            reason: format!(
                "dependency corpus and attention examples differ in length ({} vs {})",
                sentences.len(),
                examples.len()
            ),
        });
    }
    for (sentence, example) in sentences.iter().zip(examples.iter()) {
        if sentence.words != example.words {
            return Err(PaperError::InvalidAnalysis {
                reason: format!(
                    "word mismatch between dependency sentence {:?} and attention example {:?}",
                    sentence.words, example.words
                ),
            });
        }
    }
    Ok(())
}

fn ensure_coreference_alignment(
    documents: &[CoreferenceDocument],
    examples: &[WordAttentionExample],
) -> Result<()> {
    if documents.len() != examples.len() {
        return Err(PaperError::InvalidAnalysis {
            reason: format!(
                "coreference corpus and attention examples differ in length ({} vs {})",
                documents.len(),
                examples.len()
            ),
        });
    }
    for (document, example) in documents.iter().zip(examples.iter()) {
        if document.words != example.words {
            return Err(PaperError::InvalidAnalysis {
                reason: format!(
                    "word mismatch between coreference document {:?} and attention example {:?}",
                    document.words, example.words
                ),
            });
        }
    }
    Ok(())
}

fn attention_shape(attentions: &AttentionTensor) -> Result<(usize, usize, usize, usize)> {
    let layers = attentions.len();
    let heads = attentions
        .first()
        .ok_or_else(|| PaperError::InvalidAnalysis {
            reason: "attention tensor had zero layers".to_owned(),
        })?
        .len();
    let query = attentions[0]
        .first()
        .ok_or_else(|| PaperError::InvalidAnalysis {
            reason: "attention tensor had zero heads".to_owned(),
        })?
        .len();
    let key = attentions[0][0]
        .first()
        .ok_or_else(|| PaperError::InvalidAnalysis {
            reason: "attention tensor had zero query rows".to_owned(),
        })?
        .len();
    Ok((layers, heads, query, key))
}

fn flatten_head_index(heads_per_layer: usize, layer: usize, head: usize) -> usize {
    layer * heads_per_layer + head
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

#[derive(Debug, Clone, Default)]
struct RunningAverage {
    total: f64,
    count: usize,
}

impl RunningAverage {
    fn push(&mut self, value: f64) {
        self.total += value;
        self.count += 1;
    }

    fn average(&self) -> f64 {
        ratio_float(self.total, self.count)
    }
}

fn ratio_float(total: f64, count: usize) -> f64 {
    if count == 0 {
        0.0
    } else {
        total / count as f64
    }
}

#[derive(Debug, Clone, Default)]
struct SurfaceAccumulator {
    previous_attention: RunningAverage,
    self_attention: RunningAverage,
    next_attention: RunningAverage,
    cls_attention: RunningAverage,
    sep_attention: RunningAverage,
    punctuation_attention: RunningAverage,
    other_attention: RunningAverage,
    sep_to_sep: RunningAverage,
    other_to_sep: RunningAverage,
    entropy: RunningAverage,
}

impl SurfaceAccumulator {
    fn record(&mut self, tokens: &[String], head: &[Vec<f32>]) {
        self.entropy.push(average_entropy(head));

        let cls_indices = token_indices(tokens, |token| token == "[CLS]");
        let sep_indices = token_indices(tokens, |token| token == "[SEP]");
        let punctuation_indices = token_indices(tokens, |token| token == "." || token == ",");

        for (query_index, row) in head.iter().enumerate() {
            if query_index > 0 {
                self.previous_attention.push(row[query_index - 1] as f64);
            }
            self.self_attention.push(row[query_index] as f64);
            if query_index + 1 < row.len() {
                self.next_attention.push(row[query_index + 1] as f64);
            }

            self.cls_attention.push(sum_indices(row, &cls_indices));
            self.sep_attention.push(sum_indices(row, &sep_indices));
            self.punctuation_attention
                .push(sum_indices(row, &punctuation_indices));
            self.other_attention.push(
                1.0 - sum_indices(row, &cls_indices)
                    - sum_indices(row, &sep_indices)
                    - sum_indices(row, &punctuation_indices),
            );

            if tokens[query_index] == "[SEP]" {
                self.sep_to_sep.push(sum_indices(row, &sep_indices));
            } else {
                self.other_to_sep.push(sum_indices(row, &sep_indices));
            }
        }
    }
}

fn token_indices<F>(tokens: &[String], predicate: F) -> Vec<usize>
where
    F: Fn(&str) -> bool,
{
    tokens
        .iter()
        .enumerate()
        .filter_map(|(index, token)| predicate(token).then_some(index))
        .collect()
}

fn sum_indices(row: &[f32], indices: &[usize]) -> f64 {
    indices.iter().map(|&index| row[index] as f64).sum::<f64>()
}
