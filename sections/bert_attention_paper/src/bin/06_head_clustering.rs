use bert_attention_paper::analysis::cluster_attention_heads;
use bert_attention_paper::data::{
    dataset_path, pair_consecutive_sentences, read_unlabeled_documents,
};
use bert_attention_paper::runtime::{extract_pair_corpus, load_model_bundle};
use bert_attention_paper::{init_tracing, DEFAULT_MAX_SEQUENCE_LENGTH, DEFAULT_MODEL_REPO};
use clap::Parser;

#[derive(Debug, Parser)]
struct Args {
    #[arg(long, default_value = DEFAULT_MODEL_REPO)]
    model_repo: String,
    #[arg(long, default_value_t = DEFAULT_MAX_SEQUENCE_LENGTH)]
    max_sequence_length: usize,
    #[arg(long, default_value_os_t = dataset_path("unlabeled_demo.txt"))]
    input: std::path::PathBuf,
}

fn main() -> bert_attention_paper::Result<()> {
    init_tracing();
    let args = Args::parse();

    let documents = read_unlabeled_documents(&args.input)?;
    let pairs = pair_consecutive_sentences(&documents);
    let bundle = load_model_bundle(&args.model_repo)?;
    let examples = extract_pair_corpus(&bundle, &pairs, args.max_sequence_length)?;
    let summary = cluster_attention_heads(&examples)?;

    println!(
        "computed {}x{} head distance matrix",
        summary.distance_matrix.len(),
        summary.distance_matrix.len()
    );
    println!("\nFirst 12 MDS coordinates:");
    for point in summary.points.iter().take(12) {
        println!(
            "  {} -> ({:.3}, {:.3})",
            point.head.label(),
            point.x,
            point.y
        );
    }
    Ok(())
}
