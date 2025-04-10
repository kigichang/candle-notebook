use anyhow::Result;
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use tokenizers::Tokenizer;

fn encode_with_tokenizer(tokenizer: &Tokenizer, sentences: &Vec<&str>) -> Result<()> {
    // 顯示目前的 Padding 與 Truncation 參數
    println!("padding: {:?}", tokenizer.get_padding());
    println!("truncation: {:?}", tokenizer.get_truncation());

    // 一句一句編碼
    println!("encoding each sentence");
    for s in sentences {
        let encoded = tokenizer.encode(*s, true).map_err(anyhow::Error::msg)?;
        let ids = encoded.get_ids();
        let tokens = encoded.get_tokens();
        let attention_mask = encoded.get_attention_mask();
        println!("sentence: {:?}", s);
        println!("tokens: {:?}", tokens);
        println!("token length: {}", tokens.len()); // 留意 token 的長度
        println!("ids: {:?}", ids);
        println!("attention mask: {:?}", attention_mask);
        println!();
    }

    // 批次編碼
    println!("encoding batch");
    let encoded = tokenizer
        .encode_batch(sentences.clone(), true)
        .map_err(anyhow::Error::msg)?;
    for (idx, encoded) in encoded.iter().enumerate() {
        let ids = encoded.get_ids();
        let tokens = encoded.get_tokens();
        let attention_mask = encoded.get_attention_mask();
        println!("sentence: {:?}", sentences[idx]);
        println!("tokens: {:?}", tokens);
        println!("token length: {}", tokens.len()); // 留意 token 的長度
        println!("ids: {:?}", ids);
        println!("attention mask: {:?}", attention_mask);
        println!();
    }

    Ok(())
}

fn enocde(model: &str, reversion: &str, sentences: Vec<&str>) -> Result<()> {
    println!("model: {}, reversion: {}", model, reversion);

    let tokenizer_filename = {
        let api = ApiBuilder::new()
            .with_cache_dir("hf_cache".into()) // 修改暫存路徑
            .build()?;

        let api = api.repo(Repo::with_revision(
            model.to_owned(),
            RepoType::Model,
            reversion.to_owned(),
        ));
        api.get("tokenizer.json")?
    };

    // 使用預設設定
    let tokenizer =
        tokenizers::Tokenizer::from_file(&tokenizer_filename).map_err(anyhow::Error::msg)?;

    encode_with_tokenizer(&tokenizer, &sentences)?;

    // 修改 Padding 與 Truncation 參數
    let tokenizer = {
        let mut tokenizer =
            tokenizers::Tokenizer::from_file(tokenizer_filename).map_err(anyhow::Error::msg)?;
        let params = tokenizers::PaddingParams::default();
        let truncation = tokenizers::TruncationParams::default();
        tokenizer
            .with_padding(Some(params))
            .with_truncation(Some(truncation))
            .map_err(anyhow::Error::msg)?;
        tokenizer
    };

    encode_with_tokenizer(&tokenizer, &sentences)
}

fn main() -> Result<()> {
    enocde(
        "sentence-transformers/all-MiniLM-L6-v2",
        "main",
        vec!["This is an example sentence", "Each sentence is converted"],
    )?;

    enocde(
        "google-bert/bert-base-chinese",
        "main",
        vec!["巴黎是[MASK]国的首都。", "生活的真谛是[MASK]。"],
    )
}
