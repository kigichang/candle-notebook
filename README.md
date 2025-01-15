# candle-notebook

[Huggingface Candle](https://github.com/huggingface/candle) 是 Huggingface 官方用 [Rust](https://www.rust-lang.org/zh-TW/) 開發的 Machine Learning (ML) 框架。跟另一個知名的 Rust ML 框架 [tch-rs](https://github.com/LaurentMazare/tch-rs) 不一樣，Candle 完全用 Rust 寫成，而 tch-rs 是用 Rust 封裝 libtorch。現在 Huggingface 也找了 tch-rs 的開發者 [Laurent Mazare](https://github.com/LaurentMazare) 一起維護 Candle。

因為 [Candle 的官方文件](https://huggingface.github.io/candle/index.html) 還不完整，我主要是透過 [examples](https://github.com/huggingface/candle/tree/main/candle-examples) 和一些 Pytorch 書籍的範例，來學習怎麼用 Candle。我已經成功把公司內部的一個 Pytorch 專案移植到 Candle 上了。這些筆記大多透過程式碼來說明怎麼使用 Candle。

由於 Candle 還在開發中，我也會隨著它的更新，持續調整筆記內容。

## 筆記內容 (持續更新中)

1. [matrix-op](examples/matrix-op/README.md): 透過 2D 繪圖，學習 Candle 的基本操作。
    * 位移、旋轉矩陣。
    * 矩陣乘法。
    * 轉置矩陣。
1. [tensor-op](examples/tensor-op/README.md): 介紹張量(Tensor)建立與基本操作。
    * 張量建立。
    * 張量形狀與維度。
    * 張量形狀相容性。
    * 張量四則運算。
    * 張量矩陣乘法。
1. [tensor-op-2](examples/tensor-op-2/README.md): 其他張量操作。
    * 張量升、降維。
    * 張量內容總和、平均值、最大、最小值。
    * 比較兩個張量關係。
    * 組合多個張量。
    * 張量轉置。
    * 張量索引取值。
1. [mnist-training](examples/mnist-training/README.md): 參考 Candle 的 [Mnist](https://github.com/huggingface/candle/blob/main/candle-examples/examples/mnist-training/main.rs) 範例，說明如何用 Candle 進行模型訓練，以及如何使用硬體加速。
1. [bert-base-chinese](examples/bert-base-chinese/README.md): 使用 [google-bert/bert-base-chinese](https://huggingface.co/google-bert/bert-base-chinese)，展示如何使用 Huggingface 上的預訓練模型。
1. [Candle meet Yew](examples/candle-meet-yew/README.md): 如何在瀏覽器上使用預訓練模型。
1. [Cross Encoder](examples/cross-encoder/README.md): 透過實作 `BertForSequenceClassification` 學習如何擴充 Candle 的功能與可能會遇到的問題。
1. [Sentence Transforms - All-MiniLM-L6-v2](examples/all-MiniLM-L6-v2/README.md): 實作 [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
1. *[BAAI/bge-m3]: 利用剛上 PR 的 xmlroberta，使用[BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)模型。
1. *[Sentence Transforms - Sentence Transformers](examples/sentence-transformers/README.md): 實作 [Alibaba-NLP/gte-multilingual-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-base)
1. *[crf](examples/crf/README.md): 分享怎麼把 [Pytorch 版的 Conditional Random Field (CRF)](https://github.com/kmkurn/pytorch-crf) 移植到 Candle 上。
1. *[rnn](examples/rnn/README.md): 以 LSTM 為例，講解如何擴充 Candle 的功能。
1. *[CheatSheet](cheatsheet.md) 快速查詢指南。

這份筆記是我邊學邊記錄的結果，未來會根據實際使用經驗和 Candle 的更新繼續補充。希望能幫助到其他對 Candle 有興趣的朋友！
