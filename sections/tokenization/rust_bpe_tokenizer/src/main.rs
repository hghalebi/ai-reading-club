mod encoding;
mod io;
mod tokenizer;
mod training;
mod util;

use std::env;
use std::path::Path;

use tokenizer::{
    ArtifactPath, BPETokenizerSimple, Result, SpecialTokenSet, Text, TokenId, TokenSymbol,
    TokenizerError, VocabularySize, DEFAULT_END_OF_TEXT,
};

const DEMO_TEXT: &str = "Byte Pair Encoding keeps frequent byte patterns together.";
const DEMO_DIR: &str = "rust-bpe-tutorial";
const DEMO_VOCAB_SIZE: usize = 512;
const ENV_OPENAI_ENCODER: &str = "OPENAI_ENCODER_JSON";
const ENV_OPENAI_MERGES: &str = "OPENAI_VOCAB_BPE";

fn main() -> Result<()> {
    let mut tokenizer = BPETokenizerSimple::new();
    let allowed_special = SpecialTokenSet::new([TokenSymbol::new(DEFAULT_END_OF_TEXT)]);

    run_training_demo(&mut tokenizer, DEMO_TEXT, &allowed_special)?;
    run_artifact_roundtrip_demo(&mut tokenizer, DEMO_TEXT, &allowed_special)?;
    run_frequency_demo();
    run_openai_demo()?;
    Ok(())
}

fn run_training_demo(
    tokenizer: &mut BPETokenizerSimple,
    input: &str,
    allowed_special: &SpecialTokenSet,
) -> Result<()> {
    let text = Text::new(input);
    let vocab_size = VocabularySize::try_new(DEMO_VOCAB_SIZE)?;

    tokenizer.train(&text, vocab_size, allowed_special)?;
    let encoded = tokenizer.encode(&text, Some(allowed_special))?;
    let decoded = tokenizer.decode(&encoded)?;
    let end_of_text_id = tokenizer
        .get_special_token_id(&TokenSymbol::new(DEFAULT_END_OF_TEXT))
        .ok_or_else(|| TokenizerError::MissingToken(TokenSymbol::new(DEFAULT_END_OF_TEXT)))?;

    println!("input: {input}");
    println!("encoded ids: {encoded:?}");
    println!("decoded: {decoded}");
    println!("end-of-text id: {end_of_text_id}");
    Ok(())
}

fn run_artifact_roundtrip_demo(
    tokenizer: &mut BPETokenizerSimple,
    input: &str,
    allowed_special: &SpecialTokenSet,
) -> Result<()> {
    let text = Text::new(input);

    let output_dir = env::temp_dir().join(DEMO_DIR);
    std::fs::create_dir_all(&output_dir).map_err(|source| TokenizerError::FileRead {
        path: ArtifactPath::new(&output_dir),
        source,
    })?;
    let vocab_path = ArtifactPath::new(output_dir.join("vocab.json"));
    let merges_path = ArtifactPath::new(output_dir.join("bpe_merges.json"));

    tokenizer.save_vocab_and_merges(&vocab_path, &merges_path)?;

    let mut restored = BPETokenizerSimple::new();
    restored.load_vocab_and_merges(&vocab_path, &merges_path)?;
    let restored_ids = restored.encode(&text, Some(allowed_special))?;
    let restored_text = restored.decode(&restored_ids)?;

    println!("artifact roundtrip decoded: {restored_text}");
    Ok(())
}

fn run_frequency_demo() {
    let sequence = tokenizer::TokenIdSequence::from_vec(vec![
        TokenId::new(1),
        TokenId::new(2),
        TokenId::new(1),
        TokenId::new(2),
        TokenId::new(1),
    ]);
    let pair = BPETokenizerSimple::find_freq_pair(&sequence);
    println!("most frequent pair in [1,2,1,2,1] = {pair:?}");
}

fn run_openai_demo() -> Result<()> {
    let maybe_vocab = env::var(ENV_OPENAI_ENCODER).ok();
    let maybe_merges = env::var(ENV_OPENAI_MERGES).ok();

    let (Some(vocab_path), Some(merges_path)) = (maybe_vocab, maybe_merges) else {
        return Ok(());
    };
    if !Path::new(&vocab_path).exists() || !Path::new(&merges_path).exists() {
        return Ok(());
    }

    let mut openai = BPETokenizerSimple::new();
    openai.load_vocab_and_merges_from_openai(
        &ArtifactPath::new(vocab_path),
        &ArtifactPath::new(merges_path),
    )?;
    println!("loaded OpenAI-compatible tokenizer example");
    Ok(())
}
