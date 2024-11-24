# 以 google-bert/bert-base-chinese 為例，展示如何使用 Huggingface 上的預訓練模型

## 1. bert-base-chinese 簡介

簡單來說，此模型是在玩克漏字填空的遊戲，將句字中的 __[MASK]__ 部分填上正確的字。此範例將依[網站](https://huggingface.co/google-bert/bert-base-chinese)上的例句，產生一樣的結果。

執行方式：

```bash
$ cargo run --release  --example bert-base-chinese

Input: 巴黎是[MASK]国的首都。
"法": 0.991
"德": 0.003
"英": 0.002
"美": 0.001
"该": 0.001
Input: 生活的真谛是[MASK]。
"美": 0.341
"爱": 0.229
"乐": 0.033
"人": 0.023
"：": 0.019
```

### 1.1 準備工作

__bert-base-chinese.py__ 是 Pytorch 版本程式，此範例將 __bert-base-chinese.py__，移植到 Candle 上。由於官方上的模型檔案是舊的格式，目前 Candle 不支援，因此需要將模型轉成新格式。__bert-base-chinese.py__ 中有一段程式碼，可以將模型修正成新格式。

```python
model = AutoModelForMaskedLM.from_pretrained("bert-base-chinese")
model.eval()
output_stat_dict = model.state_dict()
for key in output_stat_dict:
    if "beta" in key or "gamma" in key:
        print("warning: old format name:", key)
torch.save(output_stat_dict, "fix-bert-base-chinese.pth")
```

再利用 Candle 功能，讀取 Pytorch 模型檔案後，將每一個權重整理成 `HashMap`，最後再存成 `safetensors` 格式。

```rust
fn conv_pth_to_safetensor() -> Result<()> {
    let pth_vec = candle_core::pickle::read_all("fix-bert-base-chinese.pth")?;
    for item in &pth_vec {
        println!("{:?}", item.0);
    }
    let mut tensor_map = HashMap::new();

    for item in pth_vec {
        tensor_map.insert(item.0, item.1);
    }

    candle_core::safetensors::save(&tensor_map, "fix-bert-base-chinese.safetensors")?;
    Ok(())
}
```

👉 範例程式：[conv_tensor.rs](../../tests/conv_tensor.rs)

我已經將轉換好的模型檔案放在 [kigichang/fix-bert-base-chinese](https://huggingface.co/kigichang/fix-bert-base-chinese) 上，程式將從這邊下載模型檔案。

為什麼不使用[官方的工具](https://huggingface.co/spaces/safetensors/convert)？因為 bert-base-chinese 的模型中，有 [Shared Tensor](https://huggingface.co/docs/safetensors/torch_shared_tensors)，官方的工具只會保留第一個 key 值，其他 key 值會被刪除，造成在載入模型時出現錯誤。

## 2. 程式說明

### 2.1 下載模型

使用 Huggingface 提供的 `hf_hub` 自 Huggingface 下載模型檔案。下載的模型檔案包含三個檔案：`config.json`、`tokenizer.json`、`fix-bert-base-chinese.safetensors`。依我的 Macbook Pro 環境，檔案會放在 `~/.cache/huggingface/hub/models--kigichang--fix-bert-base-chinese` 目錄下。如果已經下載過，則不會再下載。

```rust
let default_model = "kigichang/fix-bert-base-chinese".to_string();
let default_revision = "main".to_string();
let repo = Repo::with_revision(default_model, RepoType::Model, default_revision);

let (config_filename, tokenizer_filename, model_filename) = {
    let api = Api::new()?;
    let api = api.repo(repo);
    let config = api.get("config.json")?;
    let tokenizer = api.get("tokenizer.json")?;
    let model = api.get("fix-bert-base-chinese.safetensors")?;
    (config, tokenizer, model)
};
```

Huggingface 提供 Git 的方式管理模型，因此可以使用 `Repo::with_revision` 來指定模型的版本。目前使用 `main` 作為版本。

### 2.2 載入模型

#### 2.2.1 載入 Tokenizer

使用 Huggingface 提供的 `tokenizers` 載入模型使用的 Tokenizer。Python 版本的 Tokenizer 底層就是使用這個套件。

```rust
let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
```

#### 2.2.2 載入 bert-base-chinese 模型

首先將 JSON 格式的設定檔，載入成 `bert::Config`。

```rust
let config = File::open(config_filename)?;
Ok(serde_json::from_reader(config)?)
```

使用 `VarBuilder` 載入 __safetensors__ 模型檔案。

```rust
let vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[model_filename], DType::F32, &device)? };
```

載入 __safetensors__ 格式，記得要加上 `unsafe` 關鍵字。如要使用 Pytorch 模型，可以參考下面的程式碼。

```rust
let vb = VarBuilder::from_pth(model_filename, DType::F32, &device)?;
```

最後利用 `VarBuilder` 建立我們要使用的 `BertForMaskedLM` 模型。

```rust
let bert = BertForMaskedLM::load(vb, &config)?;
```

### 2.3 推論 (Inference)

首先我們將輸入的句子，產生對應的 ID (`input_ids`)。由於 `BertForMaskedLM` 會需要輸入一個 2D Tensor，因此我們使用 `Tensor::stack` 將 `input_ids` 轉成 2D Tensor。

```rust
let ids = tokenizer.encode(test_str, true).map_err(E::msg)?;
let input_ids = Tensor::stack(&[Tensor::new(ids.get_ids(), &device)?], 0)?;
```

接著產生推論會用到的 `type_ids` 與 `attention_mask`。與 `input_ids` 一樣，我們也使用 `Tensor::stack` 將這兩個 Tensor 轉成 2D Tensor。

```rust
let token_type_ids = Tensor::stack(&[Tensor::new(ids.get_type_ids(), &device)?], 0)?;
let attention_mask = Tensor::stack(&[Tensor::new(ids.get_attention_mask(), &device)?], 0)?;
```

最後使用 `foward` 函數，進行推論。與之前提到的 `Module` 不同，`forward` 函數只是為了對應 Pytorch 的 `forward` 函數。但 Rust 畢竟不是像 Python 一樣的動態語言，因此沒有辨法像 Pytorch 實作 `Module` 一樣的功能。

```rust
let result = bert.forward(&input_ids, &token_type_ids, Some(&attention_mask))?;
```

### 2.4 結果

在結果的張量上，取得 `[MASK]` 的位置的張量結果，並計算 `softmax` 即為每個字的機率。

```rust
let mask_id: u32 = tokenizer.token_to_id("[MASK]").unwrap();
...

let mask_idx = ids.get_ids().iter().position(|&x| x == mask_id).unwrap();
let mask_token_logits = result.i((0, mask_idx, ..))?;
let mask_token_probs = softmax(&mask_token_logits, 0)?;
```

最後取出前 5 個機率最高的字，使用 `tokenizer.id_to_token` 取得對應的字。

```rust
let mut top5_tokens: Vec<(usize, f32)> = mask_token_probs
    .to_vec1::<f32>()?
    .into_iter()
    .enumerate()
    .collect();
top5_tokens.sort_by(|a, b| b.1.total_cmp(&a.1));
let top5_tokens = top5_tokens.into_iter().take(5).collect::<Vec<_>>();

println!("Input: {}", test_str);
for (idx, prob) in top5_tokens {
    println!(
        "{:?}: {:.3}",
        tokenizer.id_to_token(idx as u32).unwrap(),
        prob
    );
}
```

## 3. 復盤

1. 使用 Pytorch 修正模型檔案。
1. 使用 Huggingface 的 `hf_hub` 下載模型檔案。
1. 使用 `VarBuilder` 載入 Pytorch 模型。
1. 使用 `Tokenizer` 載入 Tokenizer。
1. 使用 `BertForMaskedLM` 進行推論。
1. 使用 `softmax` 計算機率。
1. 使用 `tokenizer.id_to_token` 取得字。
