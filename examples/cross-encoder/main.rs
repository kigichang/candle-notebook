use anyhow::{Error as E, Result};
use candle_core::{DType, Tensor};
use candle_nn::VarBuilder;
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::fs::File;
use std::path::Path;

mod bert;
use bert::{BertForSequenceClassification, Config};

fn main() -> Result<()> {
    let device = candle_notebook::device(false)?;

    // 從 HF Hub 下載 Config 和 Model。
    let repo = Repo::with_revision(
        "cross-encoder/ms-marco-MiniLM-L-6-v2".to_owned(),
        RepoType::Model,
        "main".to_owned(),
    );
    let api = Api::new()?;
    let api = api.repo(repo);
    let config = api.get("config.json")?;
    let bert = api.get("pytorch_model.bin")?;
    let tokenizer = api.get("tokenizer.json")?;

    let config = load_config(config)?;
    let vb = VarBuilder::from_pth(bert, DType::F32, &device)?;
    let bert = BertForSequenceClassification::load(vb, &config)?;

    // 從本地端讀取匯出的 tokenizer.json。
    let mut tokenizer = tokenizers::Tokenizer::from_file(tokenizer).map_err(E::msg)?;
    // 加入預設的 padding 設定，讓所有的 token 長度對齊一致。
    let params = tokenizers::PaddingParams::default();
    let tokenizer = tokenizer.with_padding(Some(params));

    let encoded = tokenizer.encode_batch(vec![
        (
            "How many people live in Berlin?",
            "Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.",

        ),
        (
            "How many people live in Berlin?",
            "New York City is famous for the Metropolitan Museum of Art.",
        ),
    ], true).map_err(E::msg)?;

    let ids = encoded
        .iter()
        .map(|e| Tensor::new(e.get_ids(), &device).unwrap())
        .collect::<Vec<_>>();
    let ids = Tensor::stack(&ids, 0)?;
    let type_ids = encoded
        .iter()
        .map(|e| Tensor::new(e.get_type_ids(), &device).unwrap())
        .collect::<Vec<_>>();
    let type_ids = Tensor::stack(&type_ids, 0)?;
    let attention_mask = encoded
        .iter()
        .map(|e| Tensor::new(e.get_attention_mask(), &device).unwrap())
        .collect::<Vec<_>>();
    let attention_mask = Tensor::stack(&attention_mask, 0)?;

    // println!("ids: {:?}", ids);
    // println!("type_ids: {:?}", type_ids);
    // println!("attention_mask: {:?}", attention_mask);
    let result = bert.forward(&ids, &type_ids, Some(&attention_mask))?;

    // println!("{:?}", result);
    println!("{:?}", result.to_vec2::<f32>());
    Ok(())
}

fn load_config<P: AsRef<Path>>(config_filename: P) -> Result<Config> {
    let config = File::open(config_filename)?;
    Ok(serde_json::from_reader(config)?)
}
