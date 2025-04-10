  
# Huggingface Candle Tokenizer and Model
  
本章節主要介紹如何使用 Huggingface 的 `Tokenizer` 以及 Candle 如果操作模型。
  
## Load and Set Tokenizer
  
```rust
use anyhow::Result;
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
  
fn main() -> Result<()> {
    let sentences = vec!["This is an example sentence", "Each sentence is converted"];
    // let sentences = vec!["巴黎是[MASK]国的首都。"];
  
    let default_model = "sentence-transformers/all-MiniLM-L6-v2".to_owned();
    // let default_model = "google-bert/bert-base-chinese".to_owned();
  
    let default_revision = "main".to_owned(); // 指定版本。
    let repo = Repo::with_revision(default_model, RepoType::Model, default_revision);
  
    let tokenizer_filename = {
        let api = ApiBuilder::new()
            .with_cache_dir("hf_cache".into()) // 修改暫存路徑
            .build()?;
  
        let api = api.repo(repo);
        api.get("tokenizer.json")?
    };
  
    let tokenizer =
        tokenizers::Tokenizer::from_file(&tokenizer_filename).map_err(anyhow::Error::msg)?;
  
    println!("{:?}", tokenizer.get_padding());
    println!("{:?}", tokenizer.get_truncation());
  
    for s in &sentences {
        let encoding = tokenizer.encode(*s, true).map_err(anyhow::Error::msg)?;
        let ids = encoding.get_ids();
        let tokens = encoding.get_tokens();
        let attention_mask = encoding.get_attention_mask();
        println!("Sentence: {s}");
        println!("Tokens: {:?}", tokens);
        println!("IDs: {:?}", ids);
        println!("Attention Mask: {:?}", attention_mask);
        println!();
    }
  
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
  
    let encoded_inputs = tokenizer
        .encode_batch(sentences.clone(), true)
        .map_err(anyhow::Error::msg)?;
  
    for (idx, encoded) in encoded_inputs.iter().enumerate() {
        let ids = encoded.get_ids();
        let tokens = encoded.get_tokens();
        let attention_mask = encoded.get_attention_mask();
        println!("Sentence: {:?}", sentences[idx]);
        println!("Tokens: {:?}", tokens);
        println!("IDs: {:?}", ids);
        println!("Attention Mask: {:?}", attention_mask);
        println!();
    }
  
    Ok(())
}
  
```  
  
### Load Pretrained Tokenizer
  
```rust
tokenizers::Tokenizer::from_file("tokenizer.json")
```
  
## Create a Empty Model and Save It
  
## Load Pretrained Model
  