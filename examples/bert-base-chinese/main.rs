use anyhow::Result;
fn main() -> Result<()> {
    // let device = candle_notebook::device(false)?;

    // let default_model = "google-bert/bert-base-chinese".to_string();
    // let default_revision = "main".to_string();
    // let repo = Repo::with_revision(default_model, RepoType::Model, default_revision);

    // let (config_filename, tokenizer_filename, weights_filename) = {
    //     let api = Api::new()?;
    //     let api = api.repo(repo);
    //     let config = api.get("config.json")?;
    //     let tokenizer = api.get("tokenizer.json")?;
    //     let weights = api.get("model.safetensors")?;
    //     (config, tokenizer, weights)
    // };

    // println!("{config_filename:?} {tokenizer_filename:?} {weights_filename:?}");

    // let config = fs::read_to_string(config_filename)?;
    // let mut config: Config = serde_json::from_str(&config)?;
    // let tokenizers = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    // let vb =
    //     unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DType::F32, &device)? };

    // let vb = vb.rename_f(|name| {
    //     println!("renaming {name}");
    //     name.to_string()
    // });
    // //let model = BertModel::load(vb.clone(), &config).unwrap();

    Ok(())
}
