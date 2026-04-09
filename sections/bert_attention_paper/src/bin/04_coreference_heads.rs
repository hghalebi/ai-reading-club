use bert_attention_paper::analysis::analyze_coreference;
use bert_attention_paper::data::{dataset_path, read_coreference_documents};
use bert_attention_paper::runtime::{extract_single_corpus, load_model_bundle};
use bert_attention_paper::{init_tracing, DEFAULT_MAX_SEQUENCE_LENGTH, DEFAULT_MODEL_REPO};
use clap::Parser;

#[derive(Debug, Parser)]
struct Args {
    #[arg(long, default_value = DEFAULT_MODEL_REPO)]
    model_repo: String,
    #[arg(long, default_value_t = DEFAULT_MAX_SEQUENCE_LENGTH)]
    max_sequence_length: usize,
    #[arg(long, default_value_os_t = dataset_path("coreference_demo.jsonl"))]
    input: std::path::PathBuf,
}

fn main() -> bert_attention_paper::Result<()> {
    init_tracing();
    let args = Args::parse();

    let documents = read_coreference_documents(&args.input)?;
    let words = documents
        .iter()
        .map(|document| document.words.clone())
        .collect::<Vec<_>>();
    let bundle = load_model_bundle(&args.model_repo)?;
    let token_examples = extract_single_corpus(&bundle, &words, args.max_sequence_length)?;
    let word_examples = token_examples
        .iter()
        .map(|example| example.to_word_level())
        .collect::<Vec<_>>();
    let summary = analyze_coreference(&documents, &word_examples)?;

    println!("coreference analysis on {} documents", documents.len());
    println!(
        "\nBest head: {} accuracy={:.3}",
        summary.best_head.head.label(),
        summary.best_head.accuracy
    );
    println!(
        "Baselines: nearest={:.3} head-match={:.3} rule-based={:.3}",
        summary.nearest_baseline, summary.head_match_baseline, summary.rule_based_baseline
    );
    println!("\nAccuracy by mention type:");
    for score in &summary.by_type {
        println!(
            "  {:?}: acc={:.3} n={}",
            score.mention_type, score.accuracy, score.instances
        );
    }

    Ok(())
}
