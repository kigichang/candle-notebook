use anyhow::Result;
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};

fn main() -> Result<()> {
    let default_model = "google-bert/bert-base-chinese".to_owned(); // 指定模型名稱。
    let default_revision = "main".to_owned(); // 指定版本。
    let repo = Repo::with_revision(default_model, RepoType::Model, default_revision);

    let (tokenizer_filename, config_filename, model_filename) = {
        let api = ApiBuilder::new()
            .with_cache_dir("hf_cache".into()) // 修改暫存路徑
            .build()?;

        // 使用環境變數
        // 需設定 HF_HOME 環境變數
        // export HF_HOME=hf_cache
        // 則最終的暫存路徑為 hf_cache/hub
        // let api = ApiBuilder::from_env().build()?;

        //let api = Api::new()?; // 使用預設的暫存路徑
        let api = api.repo(repo);
        let config = api.get("config.json")?; // 下載 config.json
        let tokenizer = api.get("tokenizer.json")?; // 下載 tokenizer.json
        let model = api.get("model.safetensors")?; // 下載 model.safetensors
        (tokenizer, config, model)
    };

    // 顯示存放的路徑。
    println!("tokenizer.json: {tokenizer_filename:?}");
    println!("config.json: {config_filename:?}");
    println!("model.safetensors: {model_filename:?}");

    Ok(())
}
