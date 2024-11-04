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

計算張量某個維度的總和或整個張量總和。

1. `t.sum(index)`: 計算張量某個維度的總和。結果的張量維度會比原張量少 1 維。
1. `t.sum_keepdim(index)`: 計算張量某個維度的總和。結果的張量維度會與原張量相同。`t.sum_keepdim` 等同做完 `t.sum` 後，再做 `t.unsqueeze`。
1. `t.sum_all()`: 計算整個張量的總和。結果是一個純量。

整理其數學原理，假設張量 `t` 是 **2x3x4** 張量，則：

* `t.sum(0)`:

    $$
    S_{j,k} = \sum_{i=0}^{2} t_{i,j,k}
    $$

* `t.sum(1)`:

    $$
    S_{i,k} = \sum_{j=0}^{3} t_{i,j,k}
    $$

* `t.sum(2)`:

    $$
    S_{i,j} = \sum_{k=0}^{4} t_{i,j,k}
    $$

👉 範例程式：[tensor_sum.rs](../../tests/tensor_sum.rs)

### 2. 平均值 `t.mean`, `t.mean_keepdim` and `t.mean_all`

類似總和，計算張量某個維度的平均或整個張量平均值。

1. `t.mean(index)`: 計算張量某個維度的平均值。結果的張量維度會比原張量少 1 維。
1. `t.mean_keepdim(index)`: 計算張量某個維度的平均值。結果的張量維度會與原張量相同。`t.mean_keepdim` 等同做完 `t.mean` 後，再做 `t.unsqueeze`。
1. `t.mean_all()`: 計算整個張量的平均。結果是一個純量。

👉 範例程式：[tensor_mean.rs](../../tests/tensor_mean.rs)

### 3. 最大值 `t.max` and `t.max_keepdim`

類似總和，計算張量某個維度的最大值。

1. `t.max(index)`: 計算張量某個維度的最大值。結果的張量維度會比原張量少 1 維。
1. `t.max_keepdim(index)`: 計算張量某個維度的最大值。結果的張量維度會與原張量相同。`t.max_keepdim` 等同做完 `t.max` 後，再做 `t.unsqueeze`。

### 4. 最小值 `t.min` and `t.min_keepdim`

類似總和，計算張量某個維度的最小值。

1. `t.min(index)`: 計算張量某個維度的最小值。結果的張量維度會比原張量少 1 維。
1. `t.min_keepdim(index)`: 計算張量某個維度的最小值。結果的張量維度會與原張量相同。`t.min_keepdim` 等同做完 `t.min` 後，再做 `t.unsqueeze`。

### 5. 最大值與最小值索引 `t.argmax`, `t.argmax_keepdim`, `t.argmin` and `t.argmin_keepdim`

取得張量某個維度的最大值與最小值索引，回傳的張量資料型別是 `u32`。

1. `t.argmax(index)`: 取得張量某個維度的最大值索引。結果的張量維度會比原張量少 1 維。
1. `t.argmax_keepdim(index)`: 取得張量某個維度的最大值索引。結果的張量維度會與原張量相同。`t.argmax_keepdim` 等同做完 `t.argmax` 後，再做 `t.unsqueeze`。
1. `t.argmin(index)`: 取得張量某個維度的最小值索引。結果的張量維度會比原張量少 1 維。
1. `t.argmin_keepdim(index)`: 取得張量某個維度的最小值索引。結果的張量維度會與原張量相同。`t.argmin_keepdim` 等同做完 `t.argmin` 後，再做 `t.unsqueeze`。

## 三、兩個張量比較

### 1. 兩個形狀相同張量，取最大與最小值

#### `t.maximum`

使用 `t.maximum` 比較兩個張量的每個元素，取最大值。由源碼來看，右運算子是 `TensorOrScalar`，可以是：

1. 純量(數值，非純量張量)，資料型別可以與左運算元張量不同。
1. 張量，則兩個張量形狀與型別必須相同。

👉 範例程式：[tensor_maximum.rs](../../tests/tensor_maximum.rs)

#### `t.minimum`

類似 `t.maximun`，`t.minimum` 比較兩個張量的每個元素，取最小值。

👉 範例程式：[tensor_minimum.rs](../../tests/tensor_minimum.rs)

### 2. 比較兩個形狀相同張量關係 `t.eq`, `t.ne`, `t.lt`, `t.le`, `t.gt` and `t.ge`

類似 `t.maximun`，右運算子也是 `TensorOrScalar`，回傳張量的型別是 `u8`，如果兩個張量內的元素，符合比較的關係，則回傳值為 **1**，否則為 **0**。函式說明如下：

1. `eq`: 等於 (equal)
1. `ne`: 不等於 (not equal)
1. `lt`: 小於 (less than)
1. `le`: 小於等於 (less or equal)
1. `gt`: 大於 (greater than)
1. `ge`: 大於等於 (greater or equal)

👉 範例程式：[tensor_compare.rs](../../tests/tensor_compare.rs)

## 3. 不同形狀張量取最大與最小值 `t.broadcast_maximum` and `t.broadcast_minimum`

`t.broadcast_maximum` 和 `t.broadcast_minimum`，都是先用 `broadcast_as` 將兩個張量調整成一致，再使用 `maximum` 和 `minimum` 進行比較。因此左、右兩個運算元的張量形狀必須相容。

👉 範例程式：[tensor_broadcast_max_and_min.rs](../../tests/tensor_broadcast_max_and_min.rs)

### 4. 比較兩個不同形狀張量關係 `t.broadcast_eq`, `t.broadcast_ne`, `t.broadcast_lt`, `t.broadcast_le`, `t.broadcast_gt` and `t.broadcast_ge`

與 `t.broadcast_maximum` 和 `t.broadcast_minimum` 類似，都是先用 `broadcast_as` 將兩個張量調整成一致，再使用對應的函式進行比較。因此左、右兩個運算元的張量形狀必須相容。

函式對應關係表

| broadcast         | 內部使用函式 | 功能    |
|-------------------|------------|--------|
| broadcast_maximum | maximum    | 最大值  |
| broadcast_minimum | minimum    | 最小值  |
| broadcast_eq      | eq         | 等於    |
| broadcast_ne      | ne         | 不等於  |
| broadcast_lt      | lt         | 小於    |
| broadcast_le      | le         | 小於等於 |
| broadcast_gt      | gt         | 大於    |
| broadcast_ge      | ge         | 大於等於 |

👉 範例程式：[tensor_broadcast_compare.rs](../../tests/tensor_broadcast_compare.rs)

## 四、組合多個張量

### 1. 連接多個張量

可以使用 `Tensor::cat` 在指定的維度上，連接多個張量。操作時，每個張量的維度必須相同，且只能指定的維度可以允許不同大小，其餘的維度大小必須相同。結果的張量維度與輸入張量相同。例如：2x3x4 張量與 2x2x4 張量，在第 2 維連接時，會產生 2x5x4 的張量。

👉 範例程式：[tensor_cat.rs](../../tests/tensor_cat.rs)

### 2. 堆疊多個張量

可以使用 `Tensor::stack` 在指定的維度上，堆疊多個張量，操作時，每個張量的維度形狀必須相同。由源碼來看，`Tensor::stack` 等同每個張量先做 `t.unsqueeze`，然後再作`Tensor::cat`，因此結果的張量維度比輸入張量多一維。

👉 範例程式：[tensor_stack.rs](../../tests/tensor_stack.rs)

## 五、張量轉置

可以使用 `t.transpose` 對調兩個指定的維度。以一般 2 維矩陣來說，可以用 `t.transpose(0, 1)` 取得轉置矩陣。而 `t.t()` 是對調最後兩維後的張量。等同 `t.transpose(D::Minus2, D::Minus1)`。

👉 範例程式：[tensor_transpose.rs](../../tests/tensor_transpose.rs)

## gather

## index and contiguous
