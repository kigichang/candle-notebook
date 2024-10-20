# candle-notebook

[Huggingface Candle](https://github.com/huggingface/candle) 是由 Huggingface 官方開發類似 Pytorch 的 Machine Learning framework。與 Pytorch 不同的是，Candle 是用 Rust 開發；與另一個知名的 Rust ML framework [tch-rs](https://github.com/LaurentMazare/tch-rs) 不同的是，Candle 是全用 Rust 開發，而 tch-rs 是用 Rust 封裝 libtorch。目前 tch-rs 的開發者 [Laurent Mazare](https://github.com/LaurentMazare)，目前也在維護 Candle。

由於官方的文件目前還在撰寫中，所以我參考官方的文件、examples、與一些 Pytorch 書籍上的範例，重新用 Candle 實作一次的方式，來學習 Candle。目前我已經成功將公司內部一個 Pytorch 的專案，成功移植到 Candle 上，我也會在這邊分享這次移植的過程。

## 學習動機

我在接觸 Machine Learning 的時候，有短暫學習過 Tensorflow。最近對於 Deep Learning 感到興趣，前陣子也重新再學習 Rust，因此也藉由學習 Candle，一邊實戰練習 Rust，一邊學習 Deep Learning。

## 筆記撰寫方式

因為官方文件還在撰寫中，因此我的筆記內容大都會透過程式碼，來說明或解釋 Candle 的使用方式。在這個筆記中，不會再解釋有關 Rust 的知識。

由於 Candle 目前還在開發中，因為筆記的內容也會因 Candle 的更新，不定期更新。

## 筆記內容 (持續更新)

1. 透過 2D 繪圖，來學習 Candle 基本矩陣運算。
1. 解釋 Candle 的 Mnist Example，來了解如何使用 Candle 訓練模型。
1. 以 LSTM 為例，分享如何擴充 Candle 不足的功能。
1. 分享如果將 [Pytorch 版的 Conditional Random Field (CRF)](https://github.com/kmkurn/pytorch-crf) 移植到 Candle 上。
1. 使用 [google-bert/bert-base-chinese](https://huggingface.co/google-bert/bert-base-chinese)，學習如何使用 Huggingface 的 Pretrained Model。
