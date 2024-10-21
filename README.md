# candle-notebook

[Huggingface Candle](https://github.com/huggingface/candle) 是由 Huggingface 官方以 [Rust](https://www.rust-lang.org/zh-TW/) 開發的 Machine Learning (ML) framework。與另一個知名的 Rust ML framework [tch-rs](https://github.com/LaurentMazare/tch-rs) 不同的是，Candle 是全用 Rust 開發，而 tch-rs 是用 Rust 封裝 libtorch。Huggingface 目前也邀請 tch-rs 的開發者 [Laurent Mazare](https://github.com/LaurentMazare) 維護 Candle。

由於[官方文件](https://huggingface.github.io/candle/index.html)目前還未完備，所以我參考 [examples](https://github.com/huggingface/candle/tree/main/candle-examples) 與一些 Pytorch 書籍上的範例，利用將 Pytorch 範例移植至 Candle 的方式，來學習 Candle。目前我已經成功移植公司內部一個 Pytorch 的專案。我將會在此分享移植的過程。

## 筆記撰寫方式

因為[官方文件](https://huggingface.github.io/candle/index.html)還在撰寫中，因此我的筆記內容大都會透過程式碼，來說明或解釋 Candle 的使用方式。在這個筆記中，不會再解釋有關 Rust 的知識。

由於 Candle 目前還在開發中，筆記的內容也會因 Candle 的更新，不定期更新。

## 筆記內容 (持續更新中)

1. [matrix-op](examples/matrix-op/README.md): 透過 2D 繪圖，來學習 Candle 基本操作。
1. [mnist-training](examples/mnist-training/README.md): 參考 Candle 的 [Mnist Example](https://github.com/huggingface/candle/blob/main/candle-examples/examples/mnist-training/main.rs)，來了解如何使用 Candle 訓練模型。
1. [crf](examples/crf/README.md): 分享如何將 [Pytorch 版的 Conditional Random Field (CRF)](https://github.com/kmkurn/pytorch-crf) 移植到 Candle 上。
1. [rnn](examples/rnn/README.md): 以 LSTM 為例，分享如何擴充 Candle 不足的功能。
1. [bert-base-chinese](examples/bert-base-chinese/README.md): 使用 [google-bert/bert-base-chinese](https://huggingface.co/google-bert/bert-base-chinese)，學習如何使用 Huggingface 的 Pretrained Model。
1. [CheatSheet](Cheatsheet.md)
