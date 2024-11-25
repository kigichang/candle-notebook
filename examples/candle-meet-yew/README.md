# Candle Meet Yew 如何在瀏覽器上使用預訓練模型

要在瀏覽器上使用預訓練模型，需將程式編譯成 [WebAssembly (Wasm)](https://developer.mozilla.org/en-US/docs/WebAssembly)。WebAssembly 是我很喜歡的功能，我之前有使用過 [Go 的 WebAssembly](https://github.com/kigichang/go_course/tree/main/wasm) 實作類 [Javascript framework](https://github.com/dairaga/js)。但由於 Go 的 WebAssembly 會加入 Go Runtime，導致 wasm 檔案比較大，雖然可以使用 [TinyGo](https://tinygo.org/) 來縮小檔案，但是 TinyGo 會移除一些 Go 的功能，所以在某些情況下，TinyGo 會無法配合使用。

Rust 支援 WASM，加上 Candle 是用 Rust 重新實作 Pytorch 功能，而不是使用 libtorch，所以使用 Candle 可以在瀏覽器上執行預訓練模型。此範例展示使用上一篇 [bert-base-chinese](../bert-base-chinese/README.md) 範例，搭配 [Yew](https://yew.rs/) 來在瀏覽器上使用預訓練模型。由本範例的程式碼可以看出，幾乎不用去更動上一篇 bert-base-chinese 的程式，即可以在瀏覽器上執行。

## 1. 準備工作

### 1.1 編譯 WebAssembly

1. `$ rustup target add wasm32-unknown-unknown`: 讓 Rust 支援 WebAssembly。
1. `$ cargo install --locked trunk`: 使用 trunk 來編譯、打包 WebAssembly，並在本地啟動 Web 伺服器執行。

### 1.2 相依套件

與原先的範例不同，在 `tokenizers` 套件中，需要加入 `unstable_wasm` feature，並加入 `getrandom` 套件，以便在瀏覽器上使用。

```toml
[dependencies]
tokenizers = { version = "0.14.0", default-features = false, features = ["unstable_wasm"] }
getrandom = { version = "0.2", features = ["js"] }
```

👉 [https://github.com/huggingface/candle/issues/1032](https://github.com/huggingface/candle/issues/1032)

### 1.3 預先下載模型

預先從 Huggingface 下載模型，並放在專案目錄下。

```bash
$ cd examples/candle-meet-yew
$ wget https://huggingface.co/kigichang/fix-bert-base-chinese/resolve/main/fix-bert-base-chinese.safetensors
$ wget https://huggingface.co/kigichang/fix-bert-base-chinese/resolve/main/tokenizer.json
$ wget https://huggingface.co/kigichang/fix-bert-base-chinese/resolve/main/config.json
```

## 2. 編譯與執行

### 2.1 編譯

執行以下指令，編譯並啟動 Web 伺服器，此時會自動開啟瀏覽器。

```bash
$ cd examples/candle-meet-yew
$ trunk serve --release --open
```

### 2.2 複製模型檔案至 dist 目錄

編譯後，會在專案目錄下產生 `dist` 目錄，將下載的模型檔案複製到 `dist` 目錄下。

```bash
$ cp fix-bert-base-chinese.safetensors ./dist
$ cp config.json ./dist
$ cp tokenizer.json ./dist
```

### 2.3 執行推論

1. 下載模型：在網頁上，按下 __下載模型__ 按鈕，下載模型。
1. 下載完模型後，預設已經有輸入範例句字，按下 __送出__ 按鈕，稍後下方會顯示結果。

```text
"法": 0.991
"德": 0.003
"英": 0.002
"美": 0.001
"该": 0.001
```

## 3. 程式碼說明

這篇不會說明[Yew](https://yew.rs/)，只會說明如何使用 Candle 在瀏覽器上執行預訓練模型。主要程式碼檔案 [bert_base_chinese.rs](src/bert_base_chinese.rs)。

### 3.1 測試使用 Tensor

利用建立 Tensor，測試是否可以在瀏覽器上執行。

```rust
<h3>{"Test Tensor Only: "}{Tensor::new(0u32, &Device::Cpu).unwrap()}</h3>
```

👉 [show_tensor.rs](src/show_tensor.rs)

### 3.2 下載模型

從網站上下載模型。

```rust
/// 下載 config.json
async fn fetch_config() -> Result<String, JsValue> {
    console::log!("fetch config");
    let window = gloo::utils::window();
    let resp_value = JsFuture::from(window.fetch_with_str("/config.json")).await?;
    let resp: Response = resp_value.dyn_into()?;
    let text = JsFuture::from(resp.text()?).await?;
    Ok(text.as_string().unwrap())
}

/// 下載 tokenizer.json
async fn fetch_tokenizer() -> Result<Vec<u8>, JsValue> {
    console::log!("fetch tokenizer");
    let window = gloo::utils::window();
    let resp_value = JsFuture::from(window.fetch_with_str("/tokenizer.json")).await?;
    let resp: Response = resp_value.dyn_into()?;
    let buf = JsFuture::from(resp.array_buffer()?).await?;

    Ok(js_sys::Uint8Array::new(&buf).to_vec())
}

/// 下載 model 檔
async fn fetch_model() -> Result<Vec<u8>, JsValue> {
    console::log!("fetch model");
    let window = gloo::utils::window();
    let resp_value =
        JsFuture::from(window.fetch_with_str("/fix-bert-base-chinese.safetensors")).await?;
    let resp: Response = resp_value.dyn_into()?;
    let buf = JsFuture::from(resp.array_buffer()?).await?;
    Ok(js_sys::Uint8Array::new(&buf).to_vec())
}
```

### 3.3 載入 Tokenizer

使用 `Tokenizer::from_bytes` 從記憶體中載入 Tokenizer。

```rust
let tokenizer_json = fetch_tokenizer().await.unwrap();
let tokenizer = Tokenizer::from_bytes(&tokenizer_json).unwrap();
```

### 3.4 載入模型

載入 Config。

```rust
let config_json = fetch_config().await.unwrap();
let config = serde_json::from_str::<Config>(&config_json).unwrap();
```

使用 `VarBuilder::from_buffered_safetensors` 從記憶體中載入模型。

```rust
let model_bytes = fetch_model().await.unwrap();
let vb = VarBuilder::from_buffered_safetensors(model_bytes, DType::F32, &Device::Cpu).unwrap();
let model = BertForMaskedLM::load(vb, self.config.as_ref().unwrap()).unwrap();
```

### 3.4 推論

以下為推論的程式，與原本的程式碼幾乎一樣，只是將結果顯示在瀏覽器上。

```rust
/// 推論
fn inference(&self, test_str: &str) {
    let device = &Device::Cpu;
    let tokenizer = self.tokenizer.as_ref().unwrap();
    let mask_id: u32 = tokenizer.token_to_id("[MASK]").unwrap();
    let ids = tokenizer.encode(test_str, true).unwrap();
    let input_ids = Tensor::stack(&[Tensor::new(ids.get_ids(), &device).unwrap()], 0).unwrap();
    let token_type_ids =
        Tensor::stack(&[Tensor::new(ids.get_type_ids(), &device).unwrap()], 0).unwrap();
    let attention_mask = Tensor::stack(
        &[Tensor::new(ids.get_attention_mask(), &device).unwrap()],
        0,
    )
    .unwrap();
    let result = self
        .model
        .as_ref()
        .unwrap()
        .forward(&input_ids, &token_type_ids, Some(&attention_mask))
        .unwrap();

    let mask_idx = ids.get_ids().iter().position(|&x| x == mask_id).unwrap();
    let mask_token_logits = result.i((0, mask_idx, ..)).unwrap();
    let mask_token_probs = candle_nn::ops::softmax(&mask_token_logits, 0).unwrap();
    let mut top5_tokens: Vec<(usize, f32)> = mask_token_probs
        .to_vec1::<f32>()
        .unwrap()
        .into_iter()
        .enumerate()
        .collect();
    top5_tokens.sort_by(|a, b| b.1.total_cmp(&a.1));
    let top5_tokens = top5_tokens.into_iter().take(5).collect::<Vec<_>>();

    clear_status();
    for (idx, prob) in top5_tokens {
        change_status(&format!(
            "{:?}: {:.3}",
            tokenizer.id_to_token(idx as u32).unwrap(),
            prob
        ));
    }
}
```

## 4. 復盤

1. 如何設定編譯 WebAssembly。
1. 如何從記憶體中載入 Tokenizer 與模型。
1. 不需更新原本的程式碼架構，就可以在瀏覽器上執行預訓練模型。
