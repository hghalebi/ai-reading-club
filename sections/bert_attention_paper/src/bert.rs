//! Minimal BERT implementation that exposes attention weights as first-class outputs.
//!
//! The code is intentionally compact and inference-focused. It loads a standard
//! Hugging Face BERT checkpoint, runs the encoder, and optionally produces
//! masked-language-model logits so we can reproduce the paper's gradient-based
//! attention diagnostics later in the tutorial flow.

use crate::attention::AttentionTensor;
use crate::error::{PaperError, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{embedding, linear, Embedding, Init, LayerNorm, Linear, Module, VarBuilder};
use hf_hub::api::sync::ApiBuilder;
use serde::Deserialize;
use std::fs;
use std::fs::File;
use std::path::{Path, PathBuf};

const MODEL_DTYPE: DType = DType::F32;

/// Local paths to the model assets required by the tutorial.
#[derive(Debug, Clone)]
pub struct DownloadedModelFiles {
    pub config_path: PathBuf,
    pub tokenizer_path: PathBuf,
    pub weights_path: PathBuf,
}

fn hugging_face_cache_dir() -> PathBuf {
    if let Ok(hf_home) = std::env::var("HF_HOME") {
        return PathBuf::from(hf_home).join("hub");
    }

    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(".cache")
        .join("huggingface")
        .join("hub")
}

/// Loads the model assets from the Hugging Face Hub.
pub fn download_model_files(model_repo: &str) -> Result<DownloadedModelFiles> {
    let cache_dir = hugging_face_cache_dir();
    fs::create_dir_all(&cache_dir).map_err(|source| PaperError::io(&cache_dir, source))?;
    let api = ApiBuilder::new()
        .with_cache_dir(cache_dir)
        .with_progress(false)
        .build()?;
    let repo = api.model(model_repo.to_owned());
    let config_path = repo.get("config.json")?;
    let tokenizer_path = repo.get("tokenizer.json")?;
    let weights_path = repo.get("model.safetensors")?;

    Ok(DownloadedModelFiles {
        config_path,
        tokenizer_path,
        weights_path,
    })
}

/// BERT configuration subset used by this project.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct BertConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub hidden_act: HiddenAct,
    pub hidden_dropout_prob: f64,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub initializer_range: f64,
    pub layer_norm_eps: f64,
    pub pad_token_id: usize,
    #[serde(default)]
    pub model_type: Option<String>,
}

/// Activation functions used inside BERT.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HiddenAct {
    Gelu,
    GeluApproximate,
    Relu,
}

impl HiddenAct {
    fn forward(self, tensor: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::Gelu => tensor.gelu_erf(),
            Self::GeluApproximate => tensor.gelu(),
            Self::Relu => tensor.relu(),
        }
    }
}

#[derive(Debug, Clone)]
struct Dropout {
    probability: f64,
}

impl Dropout {
    fn new(probability: f64) -> Self {
        Self { probability }
    }
}

impl Module for Dropout {
    fn forward(&self, tensor: &Tensor) -> candle_core::Result<Tensor> {
        let _ = self.probability;
        Ok(tensor.clone())
    }
}

fn load_layer_norm(vb: VarBuilder, size: usize, eps: f64) -> candle_core::Result<LayerNorm> {
    let weight = vb
        .get_with_hints(size, "weight", Init::Const(1.0))
        .or_else(|_| vb.get_with_hints(size, "gamma", Init::Const(1.0)))?;
    let bias = vb
        .get_with_hints(size, "bias", Init::Const(0.0))
        .or_else(|_| vb.get_with_hints(size, "beta", Init::Const(0.0)))?;
    Ok(LayerNorm::new(weight, bias, eps))
}

#[derive(Debug, Clone)]
struct BertEmbeddings {
    word_embeddings: Embedding,
    position_embeddings: Embedding,
    token_type_embeddings: Embedding,
    layer_norm: LayerNorm,
    dropout: Dropout,
}

impl BertEmbeddings {
    fn load(vb: VarBuilder, config: &BertConfig) -> candle_core::Result<Self> {
        let word_embeddings = embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("word_embeddings"),
        )?;
        let position_embeddings = embedding(
            config.max_position_embeddings,
            config.hidden_size,
            vb.pp("position_embeddings"),
        )?;
        let token_type_embeddings = embedding(
            config.type_vocab_size,
            config.hidden_size,
            vb.pp("token_type_embeddings"),
        )?;
        let layer_norm = load_layer_norm(
            vb.pp("LayerNorm"),
            config.hidden_size,
            config.layer_norm_eps,
        )?;

        Ok(Self {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            layer_norm,
            dropout: Dropout::new(config.hidden_dropout_prob),
        })
    }

    fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor) -> candle_core::Result<Tensor> {
        let (_, sequence_length) = input_ids.dims2()?;
        let input_embeddings = self.word_embeddings.forward(input_ids)?;
        let token_type_embeddings = self.token_type_embeddings.forward(token_type_ids)?;
        let mut embeddings = (&input_embeddings + token_type_embeddings)?;

        let positions = (0..sequence_length as u32).collect::<Vec<_>>();
        let position_ids = Tensor::new(positions.as_slice(), input_ids.device())?;
        let position_embeddings = self.position_embeddings.forward(&position_ids)?;
        embeddings = embeddings.broadcast_add(&position_embeddings)?;
        embeddings = self.layer_norm.forward(&embeddings)?;
        self.dropout.forward(&embeddings)
    }
}

#[derive(Debug, Clone)]
struct BertSelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    dropout: Dropout,
    num_attention_heads: usize,
    attention_head_size: usize,
}

impl BertSelfAttention {
    fn load(vb: VarBuilder, config: &BertConfig) -> candle_core::Result<Self> {
        let attention_head_size = config.hidden_size / config.num_attention_heads;
        let all_head_size = config.num_attention_heads * attention_head_size;
        Ok(Self {
            query: linear(config.hidden_size, all_head_size, vb.pp("query"))?,
            key: linear(config.hidden_size, all_head_size, vb.pp("key"))?,
            value: linear(config.hidden_size, all_head_size, vb.pp("value"))?,
            dropout: Dropout::new(config.hidden_dropout_prob),
            num_attention_heads: config.num_attention_heads,
            attention_head_size,
        })
    }

    fn transpose_for_scores(&self, tensor: &Tensor) -> candle_core::Result<Tensor> {
        let mut new_shape = tensor.dims().to_vec();
        new_shape.pop();
        new_shape.push(self.num_attention_heads);
        new_shape.push(self.attention_head_size);
        tensor
            .reshape(new_shape.as_slice())?
            .transpose(1, 2)?
            .contiguous()
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let query = self.transpose_for_scores(&self.query.forward(hidden_states)?)?;
        let key = self.transpose_for_scores(&self.key.forward(hidden_states)?)?;
        let value = self.transpose_for_scores(&self.value.forward(hidden_states)?)?;

        let scores = query.matmul(&key.t()?)?;
        let scores = (scores / (self.attention_head_size as f64).sqrt())?;
        let scores = scores.broadcast_add(attention_mask)?;
        let probabilities = candle_nn::ops::softmax(&scores, candle_core::D::Minus1)?;
        let probabilities = self.dropout.forward(&probabilities)?;

        let context = probabilities.matmul(&value)?;
        let context = context.transpose(1, 2)?.contiguous()?;
        let context = context.flatten_from(candle_core::D::Minus2)?;
        Ok((context, probabilities))
    }
}

#[derive(Debug, Clone)]
struct BertSelfOutput {
    dense: Linear,
    layer_norm: LayerNorm,
    dropout: Dropout,
}

impl BertSelfOutput {
    fn load(vb: VarBuilder, config: &BertConfig) -> candle_core::Result<Self> {
        Ok(Self {
            dense: linear(config.hidden_size, config.hidden_size, vb.pp("dense"))?,
            layer_norm: load_layer_norm(
                vb.pp("LayerNorm"),
                config.hidden_size,
                config.layer_norm_eps,
            )?,
            dropout: Dropout::new(config.hidden_dropout_prob),
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        input_tensor: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        let hidden_states = self.dropout.forward(&hidden_states)?;
        self.layer_norm.forward(&(hidden_states + input_tensor)?)
    }
}

#[derive(Debug, Clone)]
struct BertAttention {
    self_attention: BertSelfAttention,
    self_output: BertSelfOutput,
}

impl BertAttention {
    fn load(vb: VarBuilder, config: &BertConfig) -> candle_core::Result<Self> {
        Ok(Self {
            self_attention: BertSelfAttention::load(vb.pp("self"), config)?,
            self_output: BertSelfOutput::load(vb.pp("output"), config)?,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let (self_output, probabilities) =
            self.self_attention.forward(hidden_states, attention_mask)?;
        let attention_output = self.self_output.forward(&self_output, hidden_states)?;
        Ok((attention_output, probabilities))
    }
}

#[derive(Debug, Clone)]
struct BertIntermediate {
    dense: Linear,
    activation: HiddenAct,
}

impl BertIntermediate {
    fn load(vb: VarBuilder, config: &BertConfig) -> candle_core::Result<Self> {
        Ok(Self {
            dense: linear(config.hidden_size, config.intermediate_size, vb.pp("dense"))?,
            activation: config.hidden_act,
        })
    }
}

impl Module for BertIntermediate {
    fn forward(&self, tensor: &Tensor) -> candle_core::Result<Tensor> {
        let tensor = self.dense.forward(tensor)?;
        self.activation.forward(&tensor)
    }
}

#[derive(Debug, Clone)]
struct BertOutput {
    dense: Linear,
    layer_norm: LayerNorm,
    dropout: Dropout,
}

impl BertOutput {
    fn load(vb: VarBuilder, config: &BertConfig) -> candle_core::Result<Self> {
        Ok(Self {
            dense: linear(config.intermediate_size, config.hidden_size, vb.pp("dense"))?,
            layer_norm: load_layer_norm(
                vb.pp("LayerNorm"),
                config.hidden_size,
                config.layer_norm_eps,
            )?,
            dropout: Dropout::new(config.hidden_dropout_prob),
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        input_tensor: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        let hidden_states = self.dropout.forward(&hidden_states)?;
        self.layer_norm.forward(&(hidden_states + input_tensor)?)
    }
}

#[derive(Debug, Clone)]
struct BertLayer {
    attention: BertAttention,
    intermediate: BertIntermediate,
    output: BertOutput,
}

impl BertLayer {
    fn load(vb: VarBuilder, config: &BertConfig) -> candle_core::Result<Self> {
        Ok(Self {
            attention: BertAttention::load(vb.pp("attention"), config)?,
            intermediate: BertIntermediate::load(vb.pp("intermediate"), config)?,
            output: BertOutput::load(vb.pp("output"), config)?,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let (attention_output, probabilities) =
            self.attention.forward(hidden_states, attention_mask)?;
        let intermediate_output = self.intermediate.forward(&attention_output)?;
        let layer_output = self
            .output
            .forward(&intermediate_output, &attention_output)?;
        Ok((layer_output, probabilities))
    }
}

#[derive(Debug, Clone)]
struct BertEncoder {
    layers: Vec<BertLayer>,
}

impl BertEncoder {
    fn load(vb: VarBuilder, config: &BertConfig) -> candle_core::Result<Self> {
        let layers = (0..config.num_hidden_layers)
            .map(|index| BertLayer::load(vb.pp(format!("layer.{index}")), config))
            .collect::<candle_core::Result<Vec<_>>>()?;
        Ok(Self { layers })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
    ) -> candle_core::Result<(Tensor, Vec<Tensor>)> {
        let mut states = hidden_states.clone();
        let mut attentions = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let (next_states, probabilities) = layer.forward(&states, attention_mask)?;
            attentions.push(probabilities);
            states = next_states;
        }
        Ok((states, attentions))
    }
}

#[derive(Debug, Clone)]
struct BertPredictionTransform {
    dense: Linear,
    layer_norm: LayerNorm,
    activation: HiddenAct,
}

impl BertPredictionTransform {
    fn load(vb: VarBuilder, config: &BertConfig) -> candle_core::Result<Self> {
        Ok(Self {
            dense: linear(config.hidden_size, config.hidden_size, vb.pp("dense"))?,
            layer_norm: load_layer_norm(
                vb.pp("LayerNorm"),
                config.hidden_size,
                config.layer_norm_eps,
            )?,
            activation: config.hidden_act,
        })
    }

    fn forward(&self, tensor: &Tensor) -> candle_core::Result<Tensor> {
        let tensor = self.dense.forward(tensor)?;
        let tensor = self.activation.forward(&tensor)?;
        self.layer_norm.forward(&tensor)
    }
}

#[derive(Debug, Clone)]
struct BertLMPredictionHead {
    transform: BertPredictionTransform,
    decoder: Linear,
}

impl BertLMPredictionHead {
    fn load(
        vb: VarBuilder,
        config: &BertConfig,
        tied_embeddings: &Embedding,
    ) -> candle_core::Result<Self> {
        let predictions = vb.pp("cls").pp("predictions");
        let decoder_weight = predictions
            .pp("decoder")
            .get((config.vocab_size, config.hidden_size), "weight")
            .unwrap_or_else(|_| tied_embeddings.embeddings().clone());
        let decoder_bias = predictions
            .pp("decoder")
            .get(config.vocab_size, "bias")
            .or_else(|_| predictions.get(config.vocab_size, "bias"))?;
        Ok(Self {
            transform: BertPredictionTransform::load(predictions.pp("transform"), config)?,
            decoder: Linear::new(decoder_weight, Some(decoder_bias)),
        })
    }

    fn forward(&self, tensor: &Tensor) -> candle_core::Result<Tensor> {
        let tensor = self.transform.forward(tensor)?;
        self.decoder.forward(&tensor)
    }
}

/// Encoder output with all attention maps.
#[derive(Debug, Clone)]
pub struct BertForwardOutput {
    pub sequence_output: Tensor,
    pub attentions: Vec<Tensor>,
}

/// Masked-LM output with all attention maps.
#[derive(Debug, Clone)]
pub struct BertMaskedLmOutput {
    pub logits: Tensor,
    pub attentions: Vec<Tensor>,
}

/// Minimal BERT encoder plus MLM head.
#[derive(Debug, Clone)]
pub struct BertForMaskedLm {
    config: BertConfig,
    embeddings: BertEmbeddings,
    encoder: BertEncoder,
    prediction_head: BertLMPredictionHead,
    device: Device,
}

impl BertForMaskedLm {
    /// Loads the model from local files.
    pub fn from_files(
        config_path: impl AsRef<Path>,
        weights_path: impl AsRef<Path>,
        device: &Device,
    ) -> Result<Self> {
        let config_path = config_path.as_ref();
        let weights_path = weights_path.as_ref();
        let reader =
            File::open(config_path).map_err(|source| PaperError::io(config_path, source))?;
        let config = serde_json::from_reader::<_, BertConfig>(reader)
            .map_err(|source| PaperError::json(config_path, source))?;

        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], MODEL_DTYPE, device)? };
        let root = if let Some(model_type) = &config.model_type {
            if model_type == "bert" {
                vb.clone().pp(model_type)
            } else {
                vb.clone()
            }
        } else {
            vb.clone()
        };

        let embeddings = BertEmbeddings::load(root.pp("embeddings"), &config)
            .or_else(|_| BertEmbeddings::load(vb.pp("bert").pp("embeddings"), &config))?;
        let encoder = BertEncoder::load(root.pp("encoder"), &config)
            .or_else(|_| BertEncoder::load(vb.pp("bert").pp("encoder"), &config))?;
        let prediction_head =
            BertLMPredictionHead::load(vb.clone(), &config, &embeddings.word_embeddings)?;

        Ok(Self {
            config,
            embeddings,
            encoder,
            prediction_head,
            device: device.clone(),
        })
    }

    /// Returns the loaded configuration.
    pub fn config(&self) -> &BertConfig {
        &self.config
    }

    /// Returns the device used by the model.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Runs the encoder and returns the hidden states plus attention probabilities.
    pub fn forward_with_attentions(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<BertForwardOutput> {
        let embeddings = self.embeddings.forward(input_ids, token_type_ids)?;
        let attention_mask = match attention_mask {
            Some(mask) => mask.clone(),
            None => input_ids.ones_like()?,
        };
        let extended_mask = get_extended_attention_mask(&attention_mask, embeddings.dtype())?;
        let (sequence_output, attentions) = self.encoder.forward(&embeddings, &extended_mask)?;
        Ok(BertForwardOutput {
            sequence_output,
            attentions,
        })
    }

    /// Runs masked-LM inference and returns logits plus attention probabilities.
    pub fn forward_masked_lm_with_attentions(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<BertMaskedLmOutput> {
        let output = self.forward_with_attentions(input_ids, token_type_ids, attention_mask)?;
        let logits = self.prediction_head.forward(&output.sequence_output)?;
        Ok(BertMaskedLmOutput {
            logits,
            attentions: output.attentions,
        })
    }
}

fn get_extended_attention_mask(
    attention_mask: &Tensor,
    dtype: DType,
) -> candle_core::Result<Tensor> {
    let attention_mask = match attention_mask.rank() {
        2 => attention_mask.unsqueeze(1)?.unsqueeze(1)?,
        3 => attention_mask.unsqueeze(1)?,
        _ => candle_core::bail!("expected rank-2 or rank-3 attention mask"),
    };
    let attention_mask = attention_mask.to_dtype(dtype)?;
    let ones = attention_mask.ones_like()?;
    (&ones - &attention_mask)? * -10_000.0
}

/// Converts attention tensors returned by the model into a nested Rust structure.
pub fn tensor_attentions_to_vec(attentions: &[Tensor]) -> Result<AttentionTensor> {
    attentions
        .iter()
        .map(|layer| {
            let layer = layer.squeeze(0)?;
            let layer_heads = layer.dims3()?;
            let mut heads = Vec::with_capacity(layer_heads.0);
            for head_index in 0..layer_heads.0 {
                let head = layer.get(head_index)?;
                heads.push(head.to_vec2::<f32>()?);
            }
            Ok(heads)
        })
        .collect::<candle_core::Result<Vec<_>>>()
        .map_err(Into::into)
}
