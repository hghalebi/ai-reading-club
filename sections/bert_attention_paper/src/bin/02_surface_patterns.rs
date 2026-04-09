use bert_attention_paper::analysis::analyze_surface_patterns;
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
    let summary = analyze_surface_patterns(&examples)?;

    let mut sep_sorted = summary.heads.clone();
    sep_sorted.sort_by(|left, right| right.sep_attention.total_cmp(&left.sep_attention));
    let mut next_sorted = summary.heads.clone();
    next_sorted.sort_by(|left, right| {
        right
            .next_token_attention
            .total_cmp(&left.next_token_attention)
    });
    let mut low_entropy = summary.heads.clone();
    low_entropy.sort_by(|left, right| left.average_entropy.total_cmp(&right.average_entropy));

    println!("surface analysis on {} paired segments", examples.len());
    println!("\nTop heads by [SEP] attention:");
    for head in sep_sorted.iter().take(5) {
        println!(
            "  {} sep={:.3} cls={:.3} punct={:.3} entropy={:.3}",
            head.head.label(),
            head.sep_attention,
            head.cls_attention,
            head.punctuation_attention,
            head.average_entropy
        );
    }

    println!("\nTop heads by next-token attention:");
    for head in next_sorted.iter().take(5) {
        println!(
            "  {} next={:.3} prev={:.3} self={:.3}",
            head.head.label(),
            head.next_token_attention,
            head.previous_token_attention,
            head.self_attention
        );
    }

    println!("\nLowest-entropy heads:");
    for head in low_entropy.iter().take(5) {
        println!(
            "  {} entropy={:.3} sep->sep={:.3} other->sep={:.3}",
            head.head.label(),
            head.average_entropy,
            head.sep_to_sep_attention,
            head.other_to_sep_attention
        );
    }

    Ok(())
}
