use anyhow::{Error as E, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::xlm_roberta::{Config, XLMRobertaForSequenceClassification};
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::fs::File;
use std::path::Path;
use tokenizers::Tokenizer;

fn load_config<P: AsRef<Path>>(config_filename: P) -> Result<Config> {
    let config = File::open(config_filename)?;
    Ok(serde_json::from_reader(config)?)
}

fn load_from_local(
    model_name: &str,
    device: &Device,
) -> Result<(Tokenizer, XLMRobertaForSequenceClassification)> {
    let default_model = model_name.to_owned();
    let default_revision = "main".to_owned();
    let repo = Repo::with_revision(default_model, RepoType::Model, default_revision);

    let (config_filename, tokenizer_filename, model_filename) = {
        let api = Api::new()?;
        let api = api.repo(repo);
        let config = api.get("config.json")?;
        let tokenizer = api.get("tokenizer.json")?;
        //let model = api.get("fix-bert-base-chinese.pth")?;
        let model = api.get("model.safetensors")?;
        (config, tokenizer, model)
    };

    let mut tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let params = tokenizers::PaddingParams::default();
    //println!("padding: {:?}", params);
    let truncation = tokenizers::TruncationParams::default();
    //println!("truncate: {:?}", truncation);
    tokenizer
        .with_padding(Some(params))
        .with_truncation(Some(truncation))
        .map_err(anyhow::Error::msg)?;

    let config = load_config(config_filename)?;
    //let vb = VarBuilder::from_pth(model_filename, DType::F32, &device)?;
    let vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[model_filename], DType::F32, &device)? };
    let bert = XLMRobertaForSequenceClassification::new(1, &config, vb)?;
    Ok((tokenizer, bert))
}

fn main() -> Result<()> {
    const MODLE_NAME: &str = "BAAI/bge-reranker-base";

    let device = macross::device(false)?;

    let (tokenizer, model) = load_from_local(MODLE_NAME, &device)?;

    let input = ("what is panda?", "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.");
    let encoded_input = tokenizer.encode(input, true).map_err(E::msg)?;
    let ids = encoded_input.get_ids();
    println!("ids: {:?}", ids);
    // let ids = encoded_input
    //     .iter()
    //     .map(|e| Tensor::new(e.get_ids(), &device))
    //     .collect::<candle_core::Result<Vec<_>>>()?;
    // let ids = Tensor::stack(&ids, 0)?;
    // // println!("ids: {:?}", ids.to_vec2::<u32>()?);

    // let type_ids = encoded_input
    //     .iter()
    //     .map(|e| Tensor::new(e.get_type_ids(), &device))
    //     .collect::<candle_core::Result<Vec<_>>>()?;
    // let type_ids = Tensor::stack(&type_ids, 0)?;
    // // println!("type_ids: {:?}", type_ids.to_vec2::<u32>()?);

    // let attention_mask = encoded_input
    //     .iter()
    //     .map(|e| Tensor::new(e.get_attention_mask(), &device))
    //     .collect::<candle_core::Result<Vec<_>>>()?;
    // let attention_mask = Tensor::stack(&attention_mask, 0)?;

    Ok(())
}
