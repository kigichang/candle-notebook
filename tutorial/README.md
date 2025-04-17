# Huggingface Candle 簡易版說明

- [1.Introduction](1.Introduction/README.md): 簡介與可能遇到的問題修正
  - :runner: `cargo run --release --bin candle-ex1`
- [2. Tokenizer and Model](2.tokenizer-and-model/README.md): 介紹如何使用 Huggingface 的 `Tokenizer` 以及 Candle 如何建立與載入模型。
  - :runner: `cargo run --release --bin candle-ex2-1`
  - :runner: `cargo run --release --bin candle-ex2-2`
- [3.Candle 硬體加速](3.devices/README.md): 介紹如何使用 CPU 與 GPU 加速。
  
## 工具

- [Netron](https://github.com/lutzroeder/netron): 用來檢視模型結構的工具。
- [MacTop](https://github.com/context-labs/mactop): 在 Mac 上查看 CPU 與 GPU 使用狀況的工具。
- [flamegraph](https://github.com/flamegraph-rs/flamegraph): 查看效能的工具。
