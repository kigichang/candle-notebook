---
export_on_save:
  markdown: true
markdown:
  image_dir: assets
  path: README.md
  ignore_from_front_matter: true
  absolute_image_path: false
---
# Generation with GGUF

本章節主要介紹如何使用 Huggingface Candle 與 [Qwen2.5-3B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF) 進行文字生成。本範例是修改自 [https://github.com/huggingface/candle/tree/main/candle-examples/examples/quantized-qwen2-instruct](https://github.com/huggingface/candle/tree/main/candle-examples/examples/quantized-qwen2-instruct)。在使用預訓練模型時，可以先到 [https://github.com/huggingface/candle/tree/main/candle-examples/examples](https://github.com/huggingface/candle/tree/main/candle-examples/examples) 的尋找有沒有範例可以參考。

## 程式碼流程與說明

### 1. 自 Huggingface 下載相關檔案

使用的 `tokenizer_config.json`，`tokenizer.json`，與`generation_config.json` 需要從 [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) 下載取得。

### 2. 載入 Tokenizer 與 Chat Template

與 [4.generation-llama](../4.generation-llama/README.md) 範例類似。

### 3. 下載 GGUF 檔案並載入模型

本範例使用的模型是 __qwen2.5-3b-instruct-q4_0.gguf__，下載自 [Qwen/Qwen2.5-3B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF)。依範例中的程式碼，載入模型：

```rust
let model = {
    let model =
        gguf_file::Content::read(&mut model_file).map_err(|e| e.with_path(model_path))?;
    let mut total_size_in_bytes = 0;
    for (_, tensor) in model.tensor_infos.iter() {
        let elem_count = tensor.shape.elem_count();
        total_size_in_bytes +=
            elem_count * tensor.ggml_dtype.type_size() / tensor.ggml_dtype.block_size();
    }
    println!(
        "loaded {:?} tensors ({}) in {:.2}s",
        model.tensor_infos.len(),
        &format_size(total_size_in_bytes),
        start.elapsed().as_secs_f32(),
    );

    ModelWeights::from_gguf(model, &mut model_file, &device)?
};
```

1. 使用 `candle_core::quantized::gguf_file::Content` 來載入 GGUF 檔案。
1. 使用 `candle_transformers::models::quantized_qwen2::ModelWeights` 載入模型權重。

### 4. 生成文字

與 [4.generation-llama](../4.generation-llama/README.md) 範例類似。

## 完整的程式碼

@import "src/main.rs" {as=rust}
