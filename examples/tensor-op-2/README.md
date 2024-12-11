# 張量其他操作

## 一、升維與降維

### 1. 升維 `tensor.unsqueeze`

可以用 `tensor.unsqueeze` 函式來增加張量的維度。指定要擴展的維度後，新張量會比原張量多 1 維，而且指定維度的大小會是 1。例如，張量 `t` 是一個 3x4 張量：

* `tensor.unsqueeze(0)`: 會產生 **1x3x4** 張量。
* `tensor.unsqueeze(1)`: 會產生 **3x1x4** 張量。
* `tensor.unsqueeze(2)`: 會產生 **3x4x1** 張量。
* `tensor.unsqueeze(3)`: 則會報錯。

👉 範例程式：[tensor_squeeze.rs](../../tests/tensor_squeeze.rs)

### 2. 降維 `tensor.squeeze`

`tensor.squeeze` 用來移除指定的維度，產生一個比原張量少 1 維的新張量。不過，指定要刪除的維度大小必須是 **1**。為了與 PyTorch 的行為一致，如果操作失敗，則會返回原張量。

👉 範例程式：[tensor_squeeze.rs](../../tests/tensor_squeeze.rs)

## 二、張量內容計算

### 1. 總和 `tensor.sum`, `tensor.sum_keepdim`, `tensor.sum_all`

可以對張量的某一維度或整個張量進行總和計算：

1. `tensor.sum(index)`: 計算某個維度的總和，結果的張量會**少 1 維**。
1. `tensor.sum_keepdim(index)`: 計算某個維度的總和，結果的張量維度保持與原張量相同，等同於 `tensor.sum` 後再做 `tensor.unsqueeze`。
1. `tensor.sum_all()`: 計算整個張量的總和，結果是一個純量。

整理其數學原理，假設張量 `t` 是 **2x3x4** 張量，則

`tensor.sum(0)` 計算公式：

$$
S_{j,k} = \sum_{i=0}^{2} t_{i,j,k}
$$

`tensor.sum(1)` 計算公式：

$$
S_{i,k} = \sum_{j=0}^{3} t_{i,j,k}
$$

`tensor.sum(2)` 計算公式：

$$
S_{i,j} = \sum_{k=0}^{4} t_{i,j,k}
$$

👉 範例程式：[tensor_sum.rs](../../tests/tensor_sum.rs)

### 2. 平均值 `tensor.mean`, `tensor.mean_keepdim`, `tensor.mean_all`

類似總和，可以計算張量某一維度的平均或整個張量的平均值：

1. `tensor.mean(index)`: 計算某維度的平均值，結果張量**少 1 維**。
1. `tensor.mean_keepdim(index)`: 結果維度與原張量相同，相當於 `tensor.mean` 後再 `tensor.unsqueeze`。
1. `tensor.mean_all()`: 計算整個張量的平均，結果為純量。

👉 範例程式：[tensor_mean.rs](../../tests/tensor_mean.rs)

### 3. 最大值與最小值 `tensor.max`, `tensor.max_keepdim`, `tensor.min`, `tensor.min_keepdim`

類似總和，計算張量某個維度或整個張量的最大值。

1. `tensor.max(index)`: 計算張量某個維度的最大值。結果的張量維度會比原張量少 1 維。
1. `tensor.max_keepdim(index)`: 計算張量某個維度的最大值。結果的張量維度會與原張量相同。`tensor.max_keepdim` 等同做完 `tensor.max` 後，再做 `tensor.unsqueeze`。
1. `tensor.max_all()`: 計算整個張量的最大值，結果是一個純量。

👉 範例程式：[tensor_max.rs](../../tests/tensor_max.rs)
👉 範例程式：[tensor_min.rs](../../tests/tensor_min.rs)

### 4. 最大值與最小值索引 `tensor.argmax`, `tensor.argmax_keepdim`, `tensor.argmin`, `tensor.argmin_keepdim`

用於取得張量中某一維度的最大或最小值的索引，結果的張量型別為 u32：

1. `tensor.argmax(index)`: 取得張量某個維度的最大值索引，結果的張量維度比原張量少 1 維。
1. `tensor.argmax_keepdim(index)`: 取得張量某個維度的最大值索引，結果的張量維度和原張量相同。`tensor.argmax_keepdim` 等同做完 `tensor.argmax` 後，再做 `tensor.unsqueeze`。
1. `tensor.argmin(index)`: 取得張量某個維度的最小值索引，結果的張量維度比原張量少 1 維。
1. `tensor.argmin_keepdim(index)`: 取得張量某個維度的最小值索引，結果的張量維度和原張量相同。`tensor.argmin_keepdim` 等同做完 `tensor.argmin` 後，再做 `tensor.unsqueeze`。

## 三、兩個張量比較

### 1. 比較形狀相同的兩個張量，取最大和最小值

#### `tensor.maximum`

使用 `tensor.maximum` 比較兩個張量的每個元素，取最大值。由原始碼來看，右運算子是 `TensorOrScalar`，可以是：

1. 純量(數值，非純量張量)，資料型別可以與左運算元張量不同。
1. 張量，則兩個張量形狀與型別必須相同。

👉 範例程式：[tensor_maximum.rs](../../tests/tensor_maximum.rs)

#### `tensor.minimum`

類似 `tensor.maximun`，`tensor.minimum` 比較兩個張量的每個元素，取最小值。

👉 範例程式：[tensor_minimum.rs](../../tests/tensor_minimum.rs)

### 2. 比較兩個形狀相同張量的關係 `tensor.eq`, `tensor.ne`, `tensor.lt`, `tensor.le`, `tensor.gt`, `tensor.ge`

類似 `tensor.maximun`，右運算子也是 `TensorOrScalar`，回傳張量的型別是 `u8`。若兩個張量的元素符合條件，則回傳值為 **1**，否則為 **0**。函式說明如下：

1. `eq`: 等於 (equal)
1. `ne`: 不等於 (not equal)
1. `lt`: 小於 (less than)
1. `le`: 小於等於 (less or equal)
1. `gt`: 大於 (greater than)
1. `ge`: 大於等於 (greater or equal)

👉 範例程式：[tensor_compare.rs](../../tests/tensor_compare.rs)

## 3. 比較不同形狀的張量，取最大與最小值 `tensor.broadcast_maximum` and `tensor.broadcast_minimum`

`tensor.broadcast_maximum` 和 `tensor.broadcast_minimum`，都是先用 `broadcast_as` 將兩個張量調整成一致，再使用 `maximum` 和 `minimum` 進行比較。因此左、右兩個運算元的張量形狀必須相容。

👉 範例程式：[tensor_broadcast_max_and_min.rs](../../tests/tensor_broadcast_max_and_min.rs)

### 4. 比較兩個不同形狀張量的關係 `tensor.broadcast_eq`, `tensor.broadcast_ne`, `tensor.broadcast_lt`, `tensor.broadcast_le`, `tensor.broadcast_gt` and `tensor.broadcast_ge`

與 `tensor.broadcast_maximum` 和 `tensor.broadcast_minimum` 類似，都是先用 `broadcast_as` 將兩個張量調整成一致，再使用對應的函式進行比較。因此左、右兩個運算元的張量形狀必須相容。

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

`Tensor::cat` 允許在指定維度上連接多個張量。操作時，所有張量在除指定維度外的其他維度必須相同。例如：一個 2x3x4 和 2x2x4 張量在第 2 維度連接後，會產生 2x5x4 張量。

👉 範例程式：[tensor_catensor.rs](../../tests/tensor_catensor.rs)

### 2. 堆疊多個張量

`Tensor::stack` 會在指定的維度上堆疊多個張量，所有張量的形狀必須相同。堆疊後的結果比原張量多一維。由原始碼來看，`Tensor::stack` 等同每個張量先做 `tensor.unsqueeze`，然後再作`Tensor::cat`，因此結果的張量維度比輸入張量多一維。

👉 範例程式：[tensor_stack.rs](../../tests/tensor_stack.rs)

## 五、張量轉置

使用 `tensor.transpose` 可交換兩個指定維度。例如，對於 2 維矩陣，可以使用 `tensor.transpose(0, 1)` 獲得轉置矩陣。 `tensor.t()` 是交換最後兩個維度的張量，相當於 `tensor.transpose(D::Minus2, D::Minus1)`。

👉 範例程式：[tensor_transpose.rs](../../tests/tensor_transpose.rs)

## 六、指定維度，張量索引取值

以下從張量取值時，都需要指定維度。

### 1. 給定索引張量取值 `tensor.gather`

`tensor.gather` 根據索引張量取得指定的元素值。索引張量的資料型別必須是整數型別，如 `u8`, `u32`, `i64` 等，且形狀必須與原張量形狀相同。依 Pytorch [torch.gather](https://pytorch.org/docs/stable/generated/torch.gather.html) 文件說明，其取值的方式如下：

```python
out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2
```

👉 範例程式：[tensor_gather.rs](../../tests/tensor_gather.rs)

### 2. 給定範圍取值 `tensor.narrow`

 `tensor.narrow` 指定維度， `start` 與 `length`，取得指定範圍的張量值。`start` 是起始索引，`length` 是取幾筆資料。

👉 範例程式：[tensor_narrow.rs](../../tests/tensor_narrow.rs)

### 3. 切割張量 `tensor.chunk`

`tensor.chunk` 可在指定的維度上將張量分割成指定數量的子張量。從原始碼可見，`tensor.chunk` 的實作是透過 `tensor.narrow` 來達成分割。

👉 範例程式：[tensor_chunk.rs](../../tests/tensor_chunk.rs)

### 4. 給定索引向量取值 `tensor.index_select`

與 `tensor.gather` 類似，索引張量的資料型別必須是整數型別；但 `tensor.index_select` 的索引張量必須是向量。

👉 範例程式：[tensor_index_selectensor.rs](../../tests/tensor_index_select.rs)

## 七. 張量取值 `tensor.i`

`tensor.i` 是功能最完整的取值函式，可一次指定各維度的取值範圍。根據原始碼分析，`tensor.i` 的索引是由 `TensorIndexer` 定義，其中 `TensorIndexer` 可包含以下幾種類型：

1. `Select(usize)`: 單一 `usize` 數值，用於取得單一維度或張量中的純量值。
1. `Narrow(Bound<usize>, Bound<usize>)`: `Range` 類型，用於取得一段連續範圍的張量值。
1. `IndexSelect(Tensor)`: 使用 `tensor.index_select` 的索引張量，可用於特定或隨機索引來取得張量值。

`tensor.i` 的使用範例如下：

1. `tensor.i(A)?`: 指定第 1 維的取值。
1. `tensor.i((A, B, C))`: 指定每個維度的取值範圍。

根據原始碼顯示，`tensor.i` 的內部實作透過 `tensor.narrow` 和 `tensor.index_select` 來逐維度取值。需要注意的是，索引使用 `Select(usize)` ，則結果的張量會比原張量少一維。

👉 範例程式：[tensor_index.rs](../../tests/tensor_index.rs)

### C contiguous (aka row major) 與 Fortran contiguous (aka column major) 問題 `tensor.is_contiguous`, `tensor.contiguous`

在進行 `tensor.i` 操作後，結果張量的內部記憶體排列可能是 **column-major**。這時可以使用 `tensor.contiguous` 確保記憶體排列。Candle 中的 **contiguous** 是 **row-major**，必須確保記憶體排行方式，否則在某些張量的操作上，會發生錯誤。

[Row- and column-major order](https://en.wikipedia.org/wiki/Row-_and_column-major_order)

## 八、復盤

1. 升降維：`tensor.unsqueeze` 和 `t.squeeze`。
1. 數值計算：總和、平均、最大值、最小值、最大值與最小值索引。
1. `keepdim` 與 `tensor.unsqueeze` 關係。
1. `TensorOrScalar` 可以是純量或張量。
1. 比較：最大值、最小值、相等、不等、小於、小於等於、大於、大於等於。
1. 連接張量 `Tensor::cat` 與堆疊張量 `Tensor::stack`。
1. 轉置 `tensor.transpose`。
1. 索引張量必須是整數型別。
1. `TensorIndexer` 可以是 `usize`, `Range` 與索引張量。
1. 各種取值方式：`tensor.gather`, `tensor.narrow`, `tensor.chunk`, `tensor.index_select` 與 `tensor.i`。
1. `tensor.contiguous` 確認張量記憶體排列。
