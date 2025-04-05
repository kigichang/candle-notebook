use anyhow::Result;
use hf_hub::{api::sync::Api, Repo, RepoType};

fn main() -> Result<()> {
    let default_model = "google-bert/bert-base-chinese".to_owned(); // 指定模型名稱。
    let default_revision = "main".to_owned(); // 指定版本。
    let repo = Repo::with_revision(default_model, RepoType::Model, default_revision);

    let (config_filename, tokenizer_filename, model_filename) = {
        let api = Api::new()?;
        let api = api.repo(repo);
        let config = api.get("config.json")?; // 下載 config.json
        let tokenizer = api.get("tokenizer.json")?; // 下載 tokenizer.json
        let model = api.get("model.safetensors")?; // 下載 model.safetensors
        (config, tokenizer, model)
    };

    // 顯示存放的路徑。
    println!("tokenizer.json: {tokenizer_filename:?}");
    println!("config.json: {config_filename:?}");
    println!("model.safetensors: {model_filename:?}");

    Ok(())
}
