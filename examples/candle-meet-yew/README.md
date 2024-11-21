# WASM

## Preparation

1. `rustup target add wasm32-unknown-unknown`: Add the wasm target to your Rust toolchain.
1. `cargo install --locked trunk`: Install the wasm-bindgen CLI tool.

dependency:

see [https://github.com/huggingface/candle/issues/1032](https://github.com/huggingface/candle/issues/1032)

```
[dependencies]
tokenizers = { version = "0.14.0", default-features = false, features = ["unstable_wasm"] }
getrandom = { version = "0.2", features = ["js"] }
```