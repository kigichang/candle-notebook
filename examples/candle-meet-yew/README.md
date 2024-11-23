# WASM

## Preparation

1. `rustup target add wasm32-unknown-unknown`: Add the wasm target to your Rust toolchain.
1. `cargo install --locked trunk`: Install the wasm-bindgen CLI tool.

dependency:

see [https://github.com/huggingface/candle/issues/1032](https://github.com/huggingface/candle/issues/1032)

```toml
[dependencies]
tokenizers = { version = "0.14.0", default-features = false, features = ["unstable_wasm"] }
getrandom = { version = "0.2", features = ["js"] }
```

## Build And Run

```
$ cd examples/candle-meet-yew
$ trunk serve --open
```

## Torch shared tensors

[Torch shared tensors](https://huggingface.co/docs/safetensors/torch_shared_tensors)，如果使用官方轉換工具，會將 shared tensors 只保留用第一個 key 值，其他 key 值會被刪除，造成在載入模型時出現錯誤。

使用以下方法將 pytorch 模型轉換成 safetensors 模型：

```rust
use std::collections::HashMap;
use anyhow::Result;

#[test]
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
