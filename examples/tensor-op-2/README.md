# 張量其他操作

## 一、升維與降維

### 1. 升維 `t.unsqueeze`

在張量中，可以透過 `t.unsqueeze` 函式，指定擴展的維度，產生新的張量。新的張量會比原張量多 1 維，且指定維度的大小為 **1**。例如張量 `t` 是 **3x4** 張量：

* `t.unsqueeze(0)`: 會產生 **1x3x4** 張量。
* `t.unsqueeze(1)`: 會產生 **3x1x4** 張量。
* `t.unsqueeze(2)`: 會產生 **3x4x1** 張量。
* `t.unsqueeze(3)`: 會發生錯誤。

👉 範例程式：[tensor_squeeze.rs](../../tests/tensor_squeeze.rs)

### 2. 降維 `t.squeeze`

在張量中，可以透過 `t.squeeze` 函式，指定要刪除的維度，產生新的張量。新的張量會比原張量少 1 維，且指定維度的大小必須為 **1**。透過源始碼註解說明，為與 Pytorch 行為一致，如果發生錯誤時，會回傳與原張量相同的張量。

👉 範例程式：[tensor_squeeze.rs](../../tests/tensor_squeeze.rs)

## 二、計算張量內容

### 1. 總和 `t.sum`, `t.sum_keepdim` and `t.sum_all`

### 2. 平均值 `t.mean`, `t.mean_keepdim` and `t.mean_all`

### 3. 最大值 `t.max`, `t.max_keepdim` and `t.max`

### 4. 最小值 `t.min`, `t.min_keepdim` and `t.min`

### 5. 最大值與最小值索引 `t.argmax` and `t.argmin`

## 三。兩個張量比較

## maximum and minimum ✅

## broadcat_max and broadcast_min ✅

### eq, ne, lt, le, gt and ge  

### broadcast_eq, broadcast_ne, broadcast_lt, broadcast_le, broadcast_gt and broadcast_ge

## stack and cat ✅

## transpose ✅

## gather

## index and contiguous
