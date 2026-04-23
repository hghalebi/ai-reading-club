use bert_attention_paper::analysis::analyze_dependency_syntax;
use bert_attention_paper::data::{dataset_path, read_dependency_sentences};
use bert_attention_paper::runtime::{extract_single_corpus, load_model_bundle};
use bert_attention_paper::{init_tracing, DEFAULT_MAX_SEQUENCE_LENGTH, DEFAULT_MODEL_REPO};
use clap::Parser;

#[derive(Debug, Parser)]
struct Args {
    #[arg(long, default_value = DEFAULT_MODEL_REPO)]
    model_repo: String,
    #[arg(long, default_value_t = DEFAULT_MAX_SEQUENCE_LENGTH)]
    max_sequence_length: usize,
    #[arg(long, default_value_os_t = dataset_path("dependency_dev.tsv"))]
    input: std::path::PathBuf,
}

fn main() -> bert_attention_paper::Result<()> {
    init_tracing();
    let args = Args::parse();

    let sentences = read_dependency_sentences(&args.input)?;
    let words = sentences
        .iter()
        .map(|sentence| sentence.words.clone())
        .collect::<Vec<_>>();
    let bundle = load_model_bundle(&args.model_repo)?;
    let token_examples = extract_single_corpus(&bundle, &words, args.max_sequence_length)?;
    let word_examples = token_examples
        .iter()
        .map(|example| example.to_word_level())
        .collect::<Vec<_>>();
    let summary = analyze_dependency_syntax(&sentences, &word_examples)?;

    println!(
        "dependency syntax analysis on {} sentences",
        sentences.len()
    );
    println!(
        "\nBest overall head: {} {:?} accuracy={:.3} baseline(offset {})={:.3}",
        summary.best_overall.head.label(),
        summary.best_overall.direction,
        summary.best_overall.accuracy,
        summary.best_overall.baseline_offset,
        summary.best_overall.baseline_accuracy
    );

    println!("\nBest head per relation:");
    for relation in &summary.relations {
        println!(
            "  {:<10} {} {:?} acc={:.3} baseline({})={:.3} n={}",
            relation.relation,
            relation.head.label(),
            relation.direction,
            relation.accuracy,
            relation.baseline_offset,
            relation.baseline_accuracy,
            relation.instances
        );
    }

    Ok(())
}
