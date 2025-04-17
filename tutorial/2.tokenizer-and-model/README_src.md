---
export_on_save:
  markdown: true
markdown:
  image_dir: assets
  path: README.md
  ignore_from_front_matter: true
  absolute_image_path: false
---

# Huggingface Candle Tokenizer and Model

本章節主要介紹如何使用 Huggingface 的 `Tokenizer` 以及如何使用 Candle 讀寫模型。

## Tokenizer

範例程式如下：

@import "tokenizer.rs" {as=rust}

### Load Tokenizer

使用 `Tokenizer::from_file` 來載入預訓練好的 `tokenizer.json` 檔。如果沒有 `tokenizer.json` 檔， :eyes: [1.Introduction](1.Introduction/README.md) 會有說明如何產生。

```rust
tokenizers::Tokenizer::from_file("tokenizer.json")
```

### 編碼

將句子編碼的方式有：

1. `encode`: 單句編碼，最常用方式。
1. `encode_batch`: 批次編碼，將多個句子一起編碼。常用 reranking, 同時比較多個句子的相關性。

`encode` 與 `encode_batch` 的第二個參數是 `add_special_tokens`，建議都設定成 `true`，會在句字的開頭與結尾加上特殊 token，例如 `[CLS]` 與 `[SEP]`。

### 取得編碼後的 ids, type ids, 與 attention mask

通常模型常會用到這三個資料：

1. ids: `let ids = encoded.get_ids();`
1. type ids: `let type_ids = encoded.get_type_ids();`
1. attention mask: `let attention_mask = encoded.get_attention_mask();`

### 修改分詞器設定

如果修改 tokenizer 的設定，可以用 `with_*` 來修改。

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

通常用在 reranking 的時候，同時要比多個句字的相關性，使用 `encode_batch` 時，讓每個句字編號後的長度都一致，且又不是取固定最大長度。在範例中，修正 `padding` 與 `truncation` 的參數後，編碼後的長度會以最長的句子為準，其他的句子會補上 `[PAD]` 來讓長度一致。

## Model

操作模型主要會用到兩種結構:

1. `VarMap`: 類似 Pytorch 的 state_dict，存放模型的參數。儲存格式只支援 __safetensors__.
1. `VarBuilder`: 用來建立或讀取模型神經網路的結構。

在建立 `VarBuilder` 時，需指定 `Tensor` (張量) 的資料型態與裝置，這兩個參數會影響到模型的運算速度與記憶體的使用量。

1. Candle 支援的 Tensor 資料型態有：

    ```rust
    pub enum DType {
        // Unsigned 8 bits integer.
        U8,
        // Unsigned 32 bits integer.
        U32,
        // Signed 64 bits integer.
        I64,
        // Brain floating-point using half precision (16 bits).
        BF16,
        // Floating-point using half precision (16 bits).
        F16,
        // Floating-point using single precision (32 bits).
        F32,
        // Floating-point using double precision (64 bits).
        F64,
    }
    ```

    對應到 Rust 的資料型別如下：

    | DType | 對應 Rust 型別 |
    |:-----:| ------------- |
    | U8    | u8            |
    | U32   | u32           |
    | I64   | i64           |
    | BF16  | half::bf16    |
    | F16   | half::f16     |
    | F32   | f32           |
    | F64   | f64           |

1. 指定裝置，在範例中，使用 CPU，在後面的章節會介紹如何使用 CPU 加速與 GPU。

    ```rust
    let device = Device::Cpu;
    ```

    目前 Candle 支援 Intel® CPU 與 Intel® Math Kernel Library (MKL) 加速，但 Candle 使用的 [intel-mkl-src](https://github.com/rust-math/intel-mkl-src) 已很久沒更新了，我自己的測試，MKL 開發套件不要安裝最新 24.x.x。目前我使用的版本是 23.1.0。

    Candle 支援 CUDA 與 Apple M 系列的 CPU 與 GPU 加速。在 Mac 上開發 Candle，基本上不需要額外安裝任何套件，就可以使用硬體加速。

完整範例：

@import "model.rs" {as=rust}

### 建立空白的神經網路結構

以下範例會建立兩層 Linear 與一層 Embedding 的神經網路結構，並存檔至 __candle-ex2-2.safetensors__，可以使用 Netron 來檢視。

```rust
// 產生一個空的 VarMap，並使用 VarBuilder 來建立變數，最後使用 save 方法將模型儲存到檔案中。

let varmap = VarMap::new();
let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

let ln1 = linear(IMAGE_DIM, HIDDEN_DIM, vb.pp("ln1"))?;
let ln2 = linear(HIDDEN_DIM, LABELS, vb.pp("ln2"))?;
let embed = embedding(IMAGE_DIM, HIDDEN_DIM, vb.pp("my").pp("embed"))?;
varmap.save(default_model_name)?;

print_linear("ln1", &ln1)?;
print_linear("ln2", &ln2)?;
println!("embed: {:?}", embed.embeddings().to_vec2::<f32>()?);
```

1. 使用 `candle_nn::linear` 來建立兩層 Linear 的神經網路結構，內含 `weight` 與 `bias`。
1. 使用 `candle_nn::embedding` 來建立一層 Embedding 的神經網路結構，輸入維度是 `IMAGE_DIM`，輸出維度是 `HIDDEN_DIM`。
1. 使用 `vb.pp("ln1")` 來指定這層的名稱。`vb.pp` 全名是 `vb.push_prefix`，因為會很常用，Candle 為了簡化，提供 `pp` 的簡寫。
1. 使用 `varmap.save` 來儲存模型的結構與參數到檔案中。

### 載入預訓練的模型

使用 `VarMap::load` 以及 `VarBuilder::from_mmaped_safetensors` 來載入 __safetensors__ 格式的模型檔案。

#### VarMap::load

使用 `VarMap::load` 載入 __safetensors__ 格式的模型檔案後，即可透過 `VarBuilder` 來讀取模型內的變數。

```rust
// 使用 VarMap 載入預訓練的模型，並使用 VarBuilder 讀取模型內的變數。

let mut varmap = VarMap::new();
let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
varmap.load(default_model_name)?;

let ln1 = linear(IMAGE_DIM, HIDDEN_DIM, vb.pp("ln1"))?;
let ln2 = linear(HIDDEN_DIM, LABELS, vb.pp("ln2"))?;
let embed = embedding(IMAGE_DIM, HIDDEN_DIM, vb.pp("my").pp("embed"))?;
print_linear("ln1", &ln1)?;
print_linear("ln2", &ln2)?;
println!("embed: {:?}", embed.embeddings().to_vec2::<f32>()?);
```

#### VarBuilder::from_mmaped_safetensors

因為 `VarBuilder::from_mmaped_safetensors` 是 unsafe call，所以需要使用 unsafe 關鍵字來包住。

```rust
// 直接使用 VarBuilder 來讀取預訓練的模型，並使用 VarBuilder 讀取模型內的變數。

let vb = unsafe {
    VarBuilder::from_mmaped_safetensors(&[default_model_name], DType::F32, &device)?
};

let ln1 = linear(IMAGE_DIM, HIDDEN_DIM, vb.pp("ln1"))?;
let ln2 = linear(HIDDEN_DIM, LABELS, vb.pp("ln2"))?;
let embed = embedding(IMAGE_DIM, HIDDEN_DIM, vb.pp("my").pp("embed"))?;
print_linear("ln1", &ln1)?;
print_linear("ln2", &ln2)?;
println!("embed: {:?}", embed.embeddings().to_vec2::<f32>()?);
```

#### Pytorch 與 Numpy  格式

`VarBuilder` 也有支援 Pytorch 與 Numpy 格式的模型檔案，但目前不支援 `VarMap` 來載入這兩種格式的模型檔案。

- Pytorch 格式： `VarBuilder::from_pth`
- Numpy 的 npz 格式: `VarBuilder::from_npz`
