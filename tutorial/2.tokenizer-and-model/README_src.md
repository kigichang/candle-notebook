---
markdown:
  image_dir: assets
  path: README.md
  ignore_from_front_matter: true
  absolute_image_path: false

export_on_save:
  markdown: true
---
# Huggingface Candle Tokenizer and Model

本章節主要介紹如何使用 Huggingface 的 `Tokenizer` 以及 Candle 如果操作模型。

## Load and Set Tokenizer

@import "tokenizer.rs" {as=rust}

1. 使用 `Tokenizer::from_file` 來載入預訓練好的 `tokenizer.json` 檔。

    ```rust
    tokenizers::Tokenizer::from_file("tokenizer.json")
    ```

1. 如果修改 tokenizer 的設定，可以用 `with_*` 來修改。

    ```rust
    let mut tokenizer =
        tokenizers::Tokenizer::from_file(tokenizer_filename).map_err(anyhow::Error::msg)?;
    let params = tokenizers::PaddingParams::default();
    let truncation = tokenizers::TruncationParams::default();
    tokenizer
        .with_padding(Some(params))
        .with_truncation(Some(truncation))
        .map_err(anyhow::Error::msg)?;
    ```

### Ecnode Sentences

將句子編碼的方式有：

1. `encode`: 單句編碼，最常用方式。
1. `encode_batch`: 批次編碼，將多個句子一起編碼。常用 reranking，同時比較多個句子的相關性。

`encode` 與 `encode_batch` 的第二個參數是 `add_special_tokens`，建議都設定成 `true`，會在句字的開頭與結尾加上特殊 token。



## Create a Empty Model and Save It

## Load Pretrained Model
