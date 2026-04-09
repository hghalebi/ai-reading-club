use bert_attention_paper::data::{dataset_path, read_dependency_sentences};
use bert_attention_paper::probe::{run_probe_comparison, GloveStore};
use bert_attention_paper::runtime::{extract_single_corpus, load_model_bundle};
use bert_attention_paper::{init_tracing, DEFAULT_MAX_SEQUENCE_LENGTH, DEFAULT_MODEL_REPO};
use clap::Parser;

#[derive(Debug, Parser)]
struct Args {
    #[arg(long, default_value = DEFAULT_MODEL_REPO)]
    model_repo: String,
    #[arg(long, default_value_t = DEFAULT_MAX_SEQUENCE_LENGTH)]
    max_sequence_length: usize,
    #[arg(long, default_value_os_t = dataset_path("dependency_train.tsv"))]
    train: std::path::PathBuf,
    #[arg(long, default_value_os_t = dataset_path("dependency_dev.tsv"))]
    dev: std::path::PathBuf,
    #[arg(long, default_value_os_t = dataset_path("glove_demo.txt"))]
    glove: std::path::PathBuf,
}

fn main() -> bert_attention_paper::Result<()> {
    init_tracing();
    let args = Args::parse();

    let train_sentences = read_dependency_sentences(&args.train)?;
    let dev_sentences = read_dependency_sentences(&args.dev)?;
    let glove = GloveStore::from_file(&args.glove)?;

    let bundle = load_model_bundle(&args.model_repo)?;
    let train_words = train_sentences
        .iter()
        .map(|sentence| sentence.words.clone())
        .collect::<Vec<_>>();
    let dev_words = dev_sentences
        .iter()
        .map(|sentence| sentence.words.clone())
        .collect::<Vec<_>>();
    let train_attn = extract_single_corpus(&bundle, &train_words, args.max_sequence_length)?
        .into_iter()
        .map(|example| example.to_word_level())
        .collect::<Vec<_>>();
    let dev_attn = extract_single_corpus(&bundle, &dev_words, args.max_sequence_length)?
        .into_iter()
        .map(|example| example.to_word_level())
        .collect::<Vec<_>>();

    let summary = run_probe_comparison(
        &train_sentences,
        &train_attn,
        &dev_sentences,
        &dev_attn,
        &glove,
    )?;

    println!("probe comparison on demo dependency data");
    println!("  right-branching  : {:.3}", summary.right_branching);
    println!("  distance + glove : {:.3}", summary.distance_and_glove);
    println!("  attn only        : {:.3}", summary.attention_only);
    println!("  attn + glove     : {:.3}", summary.attention_and_glove);
    Ok(())
}
