  
# Huggingface Candle Introduction
  
## Huggingface Candle 簡介
  
- 2023/09 月很低調的公開。
- 以 Rust 開發 Pytorch 上的功能。
- 以 Rust 開發 Python Transformers 支援 Model。
- 以 Rust 支援 Onnx <sub>(我還沒試過)</sub>。
- 支援 Cuda, Apple [Accelerate](https://developer.apple.com/documentation/accelerate )/[MLX](https://ml-explore.github.io/mlx/build/html/index.html ), [Intel MKL](https://github.com/rust-math/intel-mkl-src ) <sub>(似乎被放棄了)</sub>。
- 因為用 Rust 開發，支援跨平台，包含 WASM。
  - WASM 目前只支援 CPU SIMD，不支援 WebNN 與 WebGPU。
  - WebGPU 有分支: [https://github.com/KimHenrikOtte/candle/tree/wgpu_cleanup](https://github.com/KimHenrikOtte/candle/tree/wgpu_cleanup )。
  - Huggingface 另一個 WebGPU 框架: [https://github.com/huggingface/ratchet](https://github.com/huggingface/ratchet )。
  
## Why should I use Candle?
  
> Candle's core goal is to __make serverless inference possible__. Full machine learning frameworks like PyTorch are very large, which makes creating instances on a cluster slow. Candle allows deployment of lightweight binaries.
>
>Secondly, Candle lets you __remove Python from production workloads__. Python overhead can seriously hurt performance, and the GIL is a notorious source of headaches.
>
>Finally, ___Rust is cool!___ A lot of the HF ecosystem already has Rust crates, like [safetensors](https://github.com/huggingface/safetensors ) and [tokenizers](https://github.com/huggingface/tokenizers ).
>
  
FROM: [https://github.com/huggingface/candle?tab=readme-ov-file#why-should-i-use-candle](https://github.com/huggingface/candle?tab=readme-ov-file#why-should-i-use-candle )
  
## Structures of Huggingface Candle
  
- __candle-core: Core ops, devices, and Tensor struct definition__
- __candle-nn: Tools to build real models__
- candle-examples: Examples of using the library in realistic settings
- candle-kernels: CUDA custom kernels
- candle-datasets: Datasets and data loaders.
- __candle-transformers: transformers-related utilities.__
- candle-flash-attn: Flash attention v2 layer.
- candle-onnx: ONNX model evaluation.
  
1. __candle-transformers__ 對標 Huggingface Transformers，支援多種模型，但還是有缺漏。
1. 如果只實作 Inference 的話，查看 __candle-transformers__ 是否有支援；或者找 __candle-examples__ 是否有範例。直接改範例會比較快。
1. 如果要實作 Training 或模型沒有支援，需實作 Inference，就會用到 __candle-core__ 與 __candle-nn__。
1. 要實作 Inference 的話，可參考 [https://github.com/ToluClassics/candle-tutorial](https://github.com/ToluClassics/candle-tutorial )，有手把手實作 roberta 的範例。通常這個範例就涵蓋了大部分實作推論的細節。
  
## 與 Candle 高度相關的套件
  
- [Huggingface Hub](https://github.com/huggingface/hf-hub ): 相容 Python 版本的 Huggingface Hub，用來下載 Huggingface 上的模型。
- [tokenizers](https://github.com/huggingface/tokenizers ):
  - Huggingface 使用 Rust 實作 BPE, WordPiece and Unigram 等演算法的分詞器。
  - Python 版的 tokenizer 底層就是這個。
  - 對標: [Open AI tiktoken](https://github.com/openai/tiktoken ) (也是用 Rust 實作)
  - 在 hub 上的檔案是 tokenizer.json.
    - ex: [https://huggingface.co/google-bert/bert-base-chinese/blob/main/tokenizer.json](https://huggingface.co/google-bert/bert-base-chinese/blob/main/tokenizer.json )
    - 如果沒有這個檔，可以利用 Python 程式，來匯出。
  
## 自 Huggingface Hub 下載模型
  
以下範例是展示如何從 Huggingface Hub 下載模型，如果已經下載過，就會直接從本地載入，不會再下載。
  
```rust
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
  
```  
  
執行:
  
```bash
cargo run --bin candle-ex1
```
  
### 檔案名稱慣例
  
目前 Huggingface 官方推薦使用 `.safetensors`，如果是原本 Pytorch 的格式，副檔名會是 `.bin`。檔案名稱慣例，如果是 Pytorch 格式，則是：`pytorch_model.bin`，如果是 Safetensors 格式，則是：`model.safetensors`。
  
### 設定模型下載路徑
  
模型預設下載路徑會是 `~/.cache/huggingface/hub`，可以使用 `ApiBuilder` 來修改下載路徑。
  
1. `with_cache_dir` 來設定模型下載路徑。
  
    ```rust
    let api = ApiBuilder::new()
      .with_cache_dir("hf_cache".into()) // 修改暫存路徑
      .build()?;
    //let api = Api::new()?; // 使用預設的暫存路徑
    ```
  
1. `from_env()` 從環境變數取得下載路徑，需設定 __HF_HOME__ 環境變數。如: `export HF_HOME=hf_cache`，則最終的暫存路徑為 __hf_cache/hub__。
  
    ```rust
    let api = ApiBuilder::new()
      .from_env() // 從環境變數取得下載路徑
      .build()?;
    ```
  
## 解決沒有 tokenizer.json 方式
  
直接寫一個 Python 程式，來匯出 tokenizer.json 檔案。以 [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2 )。
  
```py
from transformers import AutoTokenizer
import torch
  
tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')
print(tokenizer)
# 存 tokenizer
tokenizer.save_pretrained("tmp") # 匯出檔案至 tmp 目錄下。
```  
  
利用 `save_pretrained` 來匯出 tokenizer.json 檔案。
  
## 顯示模型結構與修正舊版 weight 名稱
  
之前我練習發生過載入 LSTM 預訓練模型，發生 weight 值錯誤，以及舊版的 weight 名稱 (gamma/beta) 問題，可以寫一個 Python 程式，將模型重新匯出一次就可以解決。
  
以 [google-bert/bert-base-chinese](https://huggingface.co/google-bert/bert-base-chinese ) 為例，模型結構是舊版，裏面會有 `gamma` 與 `beta`。
  
```py
from transformers import AutoModelForMaskedLM
import torch
  
  
model = AutoModelForMaskedLM.from_pretrained("bert-base-chinese")
print(model) # 印出模型結構
  
# fix old format of the model for candle start
model.eval()
output_stat_dict = model.state_dict() # 取得模型所有 weight
for key in output_stat_dict:
    if "beta" in key or "gamma" in key:
        print("warning: old format name:", key)
torch.save(output_stat_dict, "fix-bert-base-chinese.pth")
# fix old format of the model for candle end
```  
  