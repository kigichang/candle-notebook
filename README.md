# candle-notebook

[Huggingface Candle](https://github.com/huggingface/candle) 是 Huggingface 官方用 [Rust](https://www.rust-lang.org/zh-TW/) 開發的 Machine Learning (ML) 框架。跟另一個知名的 Rust ML 框架 [tch-rs](https://github.com/LaurentMazare/tch-rs) 不一樣，Candle 完全用 Rust 寫成，而 tch-rs 是用 Rust 封裝 libtorch。現在 Huggingface 也找了 tch-rs 的開發者 [Laurent Mazare](https://github.com/LaurentMazare) 一起維護 Candle。

因為 [Candle 的官方文件](https://huggingface.github.io/candle/index.html) 還不完整，我主要是透過 [examples](https://github.com/huggingface/candle/tree/main/candle-examples) 和一些 Pytorch 書籍的範例，來學習怎麼用 Candle。我已經成功把公司內部的一個 Pytorch 專案移植到 Candle 上了。這些筆記大多透過程式碼來說明怎麼使用 Candle。

由於 Candle 還在開發中，我也會隨著它的更新，持續調整筆記內容。

## 筆記內容 (持續更新中)

1. [matrix-op](examples/matrix-op/README.md): 透過 2D 繪圖，學習 Candle 的基本操作。
1. [tensor-op](examples/tensor-op/README.md): 介紹張量(Tensor)建立與基本運算。
1. *[tensor-adv-op](examples/tensor-adv-op/README.md): 進階張量操作。
1. *[mnist-training](examples/mnist-training/README.md): 參考 Candle 的 [Mnist](https://github.com/huggingface/candle/blob/main/candle-examples/examples/mnist-training/main.rs) 範例，說明如何用 Candle 進行模型訓練。
1. *[crf](examples/crf/README.md): 分享怎麼把 [Pytorch 版的 Conditional Random Field (CRF)](https://github.com/kmkurn/pytorch-crf) 移植到 Candle 上。
1. *[rnn](examples/rnn/README.md): 以 LSTM 為例，講解如何擴充 Candle 的功能。
1. *[bert-base-chinese](examples/bert-base-chinese/README.md): 使用 [google-bert/bert-base-chinese](https://huggingface.co/google-bert/bert-base-chinese)，展示如何使用 Huggingface 的預訓練模型。
1. *[platform](examples/platform/README.md): 在 Apple Silicon、Intel CPU 和 Nvidia GPU 上使用 Candle。
1. *[CheatSheet](Cheatsheet.md) 快速查詢指南。

這份筆記是我邊學邊記錄的結果，未來會根據實際使用經驗和 Candle 的更新繼續補充。希望能幫助到其他對 Candle 有興趣的朋友！
