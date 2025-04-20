---
export_on_save:
  markdown: true
markdown:
  image_dir: assets
  path: README.md
  ignore_from_front_matter: true
  absolute_image_path: false
---

# Generation

本章節主要介紹如何使用 Huggingface Candle 與 [Qwen2.5-1.5-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) 進行文字生成。本範例是修改自 [https://github.com/huggingface/candle/tree/main/candle-examples/examples/qwen](https://github.com/huggingface/candle/tree/main/candle-examples/examples/qwen)。

## 程式碼流程與說明

### 自 Huggingface 下載相關檔案

以先前範例雷同，但

1. 多下載了 `tokenizer_config.json`，讀取模型的 chat template。
1. 多下載了 `generation_config.json`，讀取 `eos_token_id`，用來判斷生成的結束條件。
1. 先下載 `model.safetensors`，如果沒有，代表該模型檔案太大，有切割成多個檔案，這時候就要下載 `model.safetensors.index.json`，然後再下載切割的模型檔案。
    - 使用 `candle_examples::hub_load_safetensors` 來處理最後存放的路徑。

### 載入 Tokenizer 與建立模型

與先前範例雷同，使用 `tokenizers` 載入 `tokenizer.json`，產生 `Tokenizer`。使用 `VarBuilder::from_mmaped_safetensors` 載入模型檔案。仿照官方 Sample Code。用 Device 是否為 CUDA 來決定是否使用 `BF16`。

```rust
let device = candle_examples::device(args.cpu)?;
// 參考 sample code.
let dtype = if device.is_cuda() {
    DType::BF16
} else {
    DType::F32
};

let tokenizer =
    tokenizers::Tokenizer::from_file(repo_files.tokenizer).map_err(anyhow::Error::msg)?;
let vb =
    unsafe { VarBuilder::from_mmaped_safetensors(&repo_files.model_files, dtype, &device)? };
```

使用 __Qwen2.5-1.5-Instruct__ 會使用 `candle_transformers::models::qwen2::Config` 與 `candle_transformers::models::qwen2::ModelForCausalLM` 自 `VarBuilder` 來載入預訓練模型。

使用 `candle_transformers::models::qwen2::Config` 載入 [https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/config.json](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/config.json) 預訓練模型的設定檔。至於使用那種預訓練模型結構，可以參考

```json
"architectures": [
  "Qwen2ForCausalLM"
],
```

比如 __Qwen2.5-1.5B-Instruct__ 是 `Qwen2ForCausalLM`，所以可以使用 `candle_transformers::models::qwen2::ModelForCausalLM` 來載入預訓練模型。

```rust
let config: Config = serde_json::from_reader(std::fs::File::open(repo_files.config)?)?;
let model = ModelForCausalLM::new(&config, vb)?;
```

### 使用 Chat Template

## 生成文字

## 完整程式碼

@import "src/main.rs" {as=rust}
