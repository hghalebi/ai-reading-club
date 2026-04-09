use bert_attention_paper::analysis::{
    analyze_coreference, analyze_dependency_syntax, analyze_surface_patterns,
    cluster_attention_heads,
};
use bert_attention_paper::data::{
    dataset_path, pair_consecutive_sentences, read_coreference_documents,
    read_dependency_sentences, read_unlabeled_documents,
};
use bert_attention_paper::probe::{run_probe_comparison, GloveStore};
use bert_attention_paper::runtime::{
    extract_pair_corpus, extract_single_corpus, load_model_bundle,
};
use bert_attention_paper::{
    init_tracing, PaperError, DEFAULT_MAX_SEQUENCE_LENGTH, DEFAULT_MODEL_REPO,
};
use clap::Parser;

#[derive(Debug, Parser)]
struct Args {
    #[arg(long, default_value = DEFAULT_MODEL_REPO)]
    model_repo: String,
    #[arg(long, default_value_t = DEFAULT_MAX_SEQUENCE_LENGTH)]
    max_sequence_length: usize,
}

fn main() -> bert_attention_paper::Result<()> {
    init_tracing();
    let args = Args::parse();

    let bundle = load_model_bundle(&args.model_repo)?;

    let unlabeled = read_unlabeled_documents(dataset_path("unlabeled_demo.txt"))?;
    let pairs = pair_consecutive_sentences(&unlabeled);
    let pair_examples = extract_pair_corpus(&bundle, &pairs, args.max_sequence_length)?;
    let surface = analyze_surface_patterns(&pair_examples)?;
    let clustering = cluster_attention_heads(&pair_examples)?;

    let train = read_dependency_sentences(dataset_path("dependency_train.tsv"))?;
    let dev = read_dependency_sentences(dataset_path("dependency_dev.tsv"))?;
    let train_words = train
        .iter()
        .map(|sentence| sentence.words.clone())
        .collect::<Vec<_>>();
    let dev_words = dev
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
    let syntax = analyze_dependency_syntax(&dev, &dev_attn)?;
    let glove = GloveStore::from_file(dataset_path("glove_demo.txt"))?;
    let probes = run_probe_comparison(&train, &train_attn, &dev, &dev_attn, &glove)?;

    let coref_docs = read_coreference_documents(dataset_path("coreference_demo.jsonl"))?;
    let coref_words = coref_docs
        .iter()
        .map(|document| document.words.clone())
        .collect::<Vec<_>>();
    let coref_attn = extract_single_corpus(&bundle, &coref_words, args.max_sequence_length)?
        .into_iter()
        .map(|example| example.to_word_level())
        .collect::<Vec<_>>();
    let coref = analyze_coreference(&coref_docs, &coref_attn)?;

    println!("# What Does BERT Look At? Rust walkthrough");
    println!("\n## Surface patterns");
    let strongest_sep = surface
        .heads
        .iter()
        .max_by(|left, right| left.sep_attention.total_cmp(&right.sep_attention))
        .ok_or_else(|| PaperError::InvalidAnalysis {
            reason: "surface analysis did not produce any heads".to_owned(),
        })?;
    println!(
        "- strongest [SEP] head: {} (sep={:.3}, entropy={:.3})",
        strongest_sep.head.label(),
        strongest_sep.sep_attention,
        strongest_sep.average_entropy
    );

    println!("\n## Dependency syntax");
    println!(
        "- best overall head: {} {:?} with accuracy {:.3}",
        syntax.best_overall.head.label(),
        syntax.best_overall.direction,
        syntax.best_overall.accuracy
    );
    for relation in syntax.relations.iter().take(6) {
        println!(
            "- {:<10} {} {:?} acc={:.3}",
            relation.relation,
            relation.head.label(),
            relation.direction,
            relation.accuracy
        );
    }

    println!("\n## Coreference");
    println!(
        "- best head: {} acc={:.3} (nearest={:.3}, head-match={:.3}, rule-based={:.3})",
        coref.best_head.head.label(),
        coref.best_head.accuracy,
        coref.nearest_baseline,
        coref.head_match_baseline,
        coref.rule_based_baseline
    );

    println!("\n## Probes");
    println!(
        "- right-branching={:.3}, distance+glove={:.3}, attn={:.3}, attn+glove={:.3}",
        probes.right_branching,
        probes.distance_and_glove,
        probes.attention_only,
        probes.attention_and_glove
    );

    println!("\n## Head clustering");
    for point in clustering.points.iter().take(8) {
        println!(
            "- {} -> ({:.3}, {:.3})",
            point.head.label(),
            point.x,
            point.y
        );
    }

    Ok(())
}
