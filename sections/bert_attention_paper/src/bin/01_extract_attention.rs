use bert_attention_paper::runtime::{extract_single_attention, load_model_bundle};
use bert_attention_paper::{init_tracing, DEFAULT_MAX_SEQUENCE_LENGTH, DEFAULT_MODEL_REPO};
use clap::Parser;

#[derive(Debug, Parser)]
struct Args {
    #[arg(long, default_value = DEFAULT_MODEL_REPO)]
    model_repo: String,
    #[arg(long, default_value_t = DEFAULT_MAX_SEQUENCE_LENGTH)]
    max_sequence_length: usize,
    #[arg(long, default_value = "The student read the book .")]
    text: String,
}

fn main() -> bert_attention_paper::Result<()> {
    init_tracing();
    let args = Args::parse();
    let words = args
        .text
        .split_whitespace()
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();

    let bundle = load_model_bundle(&args.model_repo)?;
    let example = extract_single_attention(&bundle, &words, args.max_sequence_length)?;

    println!("model repo: {}", args.model_repo);
    println!("tokens: {:?}", example.encoded.tokens);
    println!("word spans: {:?}", example.encoded.word_spans);
    println!(
        "attention shape: layers={}, heads/layer={}, seq={}, seq={}",
        example.token_attentions.len(),
        example.token_attentions[0].len(),
        example.token_attentions[0][0].len(),
        example.token_attentions[0][0][0].len()
    );
    println!(
        "example head L1-H1 attention from token 1: {:?}",
        example.token_attentions[0][0][1]
    );
    Ok(())
}
