---
markdown:
  image_dir: assets
  path: README.md
  ignore_from_front_matter: true
  absolute_image_path: false

export_on_save:
  markdown: true
---
# Huggingface Candle Introduction

## Huggingface Candle 簡介

- 2023/09 月很低調的公開。
- 以 Rust 開發 Pytorch 上的功能。
- 以 Rust 開發 Python Transformers 支援 Model。
- 以 Rust 支援 Onnx <sub>(我還沒試過)</sub>。
- 支援 Cuda, Apple [Accelerate](https://developer.apple.com/documentation/accelerate)/[MLX](https://ml-explore.github.io/mlx/build/html/index.html), [Intel MKL](https://github.com/rust-math/intel-mkl-src) <sub>(似乎被放棄了)</sub>。
- 因為用 Rust 開發，支援跨平台，包含 WASM。
  - WASM 目前只支援 CPU SIMD，不支援 WebNN 與 WebGPU。
  - WebGPU 有分支: [https://github.com/KimHenrikOtte/candle/tree/wgpu_cleanup](https://github.com/KimHenrikOtte/candle/tree/wgpu_cleanup)。
  - Huggingface 另一個 WebGPU 框架: [https://github.com/huggingface/ratchet](https://github.com/huggingface/ratchet)。

## Why should I use Candle?

> Candle's core goal is to __make serverless inference possible__. Full machine learning frameworks like PyTorch are very large, which makes creating instances on a cluster slow. Candle allows deployment of lightweight binaries.
>
>Secondly, Candle lets you __remove Python from production workloads__. Python overhead can seriously hurt performance, and the GIL is a notorious source of headaches.
>
>Finally, ___Rust is cool!___ A lot of the HF ecosystem already has Rust crates, like [safetensors](https://github.com/huggingface/safetensors) and [tokenizers](https://github.com/huggingface/tokenizers).
>

FROM: [https://github.com/huggingface/candle?tab=readme-ov-file#why-should-i-use-candle](https://github.com/huggingface/candle?tab=readme-ov-file#why-should-i-use-candle)

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
1. 要實作 Inference 的話，可參考 [https://github.com/ToluClassics/candle-tutorial](https://github.com/ToluClassics/candle-tutorial)，有手把手實作 roberta 的範例。通常這個範例就涵蓋了大部分實作推論的細節。

## 與 Candle 高度相關的套件

- [Huggingface Hub](https://github.com/huggingface/hf-hub): 相容 Python 版本的 Huggingface Hub，用來下載 Huggingface 上的模型。
- [tokenizers](https://github.com/huggingface/tokenizers):
  - Huggingface 使用 Rust 實作 BPE, WordPiece and Unigram 等演算法的分詞器。
  - Python 版的 tokenizer 底層就是這個。
  - 對標: [Open AI tiktoken](https://github.com/openai/tiktoken) (也是用 Rust 實作)
  - 在 hub 上的檔案是 tokenizer.json.
    - ex: [https://huggingface.co/google-bert/bert-base-chinese/blob/main/tokenizer.json](https://huggingface.co/google-bert/bert-base-chinese/blob/main/tokenizer.json)
    - 如果沒有這個檔，可以利用 Python 程式，來匯出。

## 自 Huggingface Hub 下載模型

以下範例是展示如何從 Huggingface Hub 下載模型，如果已經下載過，就會直接從本地載入，不會再下載。

@import "main.rs" {as="rust"}

執行:

```bash
cargo run --bin candle-ex1
```

### 檔案名稱慣例

目前 Huggingface 官方推薦使用 `.safetensors`，如果是原本 Pytorch 的格式，副檔名會是 `.bin`。檔案名稱慣例，如果是 Pytorch 格式，則是：`pytorch_model.bin`，如果是 Safetensors 格式，則是：`model.safetensors`。

## 解決沒有 tokenizer.json 方式

直接寫一個 Python 程式，來匯出 tokenizer.json 檔案。以 [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)。

@import "dump_tokenizer.py" {highlight=7}

利用 `save_pretrained` 來匯出 tokenizer.json 檔案。

## 顯示模型結構與修正舊版 weights 名稱

之前我練習發生過載入 LSTM 預訓練模型，發生 Weight 值錯誤，以及舊版的 Weight 名稱 (gamma/beta) 問題，可以寫一個 Python 程式，將模型重新匯出一次就可以解決。

以 [google-bert/bert-base-chinese](https://huggingface.co/google-bert/bert-base-chinese) 為例，模型結構是舊版，裏面會有 `gamma` 與 `beta`。

@import "dump_model.py" {highlight=[10, 14]}
