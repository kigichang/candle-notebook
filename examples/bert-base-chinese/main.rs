use anyhow::{Error as E, Result};
use candle_core::{DType, IndexOp, Tensor};
use candle_nn::{ops::softmax, VarBuilder};
use candle_transformers::models::bert::{BertForMaskedLM, Config};
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::{fs::File, path::Path};
use tokenizers::Tokenizer;
fn main() -> Result<()> {
    let test_strs = vec!["巴黎是[MASK]国的首都。", "生活的真谛是[MASK]。"];

    let device = candle_notebook::device(false)?;

    let default_model = "kigichang/fix-bert-base-chinese".to_string();
    let default_revision = "main".to_string();
    let repo = Repo::with_revision(default_model, RepoType::Model, default_revision);

    let (config_filename, tokenizer_filename, model_filename) = {
        let api = Api::new()?;
        let api = api.repo(repo);
        let config = api.get("config.json")?;
        let tokenizer = api.get("tokenizer.json")?;
        //let model = api.get("fix-bert-base-chinese.pth")?;
        let model = api.get("fix-bert-base-chinese.safetensors")?;
        (config, tokenizer, model)
    };

    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
    let config = load_config(config_filename)?;
    //let vb = VarBuilder::from_pth(model_filename, DType::F32, &device)?;
    let vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[model_filename], DType::F32, &device)? };
    let bert = BertForMaskedLM::load(vb, &config)?;

    let mask_id: u32 = tokenizer.token_to_id("[MASK]").unwrap();

    for test_str in test_strs {
        let ids = tokenizer.encode(test_str, true).map_err(E::msg)?;
        let input_ids = Tensor::stack(&[Tensor::new(ids.get_ids(), &device)?], 0)?;
        let token_type_ids = Tensor::stack(&[Tensor::new(ids.get_type_ids(), &device)?], 0)?;
        let attention_mask = Tensor::stack(&[Tensor::new(ids.get_attention_mask(), &device)?], 0)?;
        let result = bert.forward(&input_ids, &token_type_ids, Some(&attention_mask))?;

        let mask_idx = ids.get_ids().iter().position(|&x| x == mask_id).unwrap();
        let mask_token_logits = result.i((0, mask_idx, ..))?;
        let mask_token_probs = softmax(&mask_token_logits, 0)?;
        let mut top5_tokens: Vec<(usize, f32)> = mask_token_probs
            .to_vec1::<f32>()?
            .into_iter()
            .enumerate()
            .collect();
        top5_tokens.sort_by(|a, b| b.1.total_cmp(&a.1));
        let top5_tokens = top5_tokens.into_iter().take(5).collect::<Vec<_>>();

        println!("Input: {}", test_str);
        for (idx, prob) in top5_tokens {
            println!(
                "{:?}: {:.3}",
                tokenizer.id_to_token(idx as u32).unwrap(),
                prob
            );
        }
    }

    Ok(())
}

fn load_config<P: AsRef<Path>>(config_filename: P) -> Result<Config> {
    let config = File::open(config_filename)?;
    Ok(serde_json::from_reader(config)?)
}
