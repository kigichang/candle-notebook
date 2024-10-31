# 張量 (Tensor) 的建立與基本運算

這份筆記主要展示如何在 Candle 中建立張量（Tensor）並執行基本的四則運算，以供開發者在需要時參考其用法。

## 一、張量建立

Candle 張量支援的資料型別如下：

| DType | 對應 Rust 型別 |
| --- | --- |
| U8 | u8 |
| U32 | u32 |
| I64 | i64 |
| BF16 | half::bf16 |
| F16 | half::f16 |
| F32 | f32 |
| F64 | f64 |

在建立張量時，請確保 Rust 的資料型別與 Candle 支援的 DType 相對應。

### 1. 使用 `Tensor::new` 建立張量

`Tensor::new()` 根據您提供的資料與裝置建立新張量。

* 第一個參數是 `NdArray`，可以是：
  * 數值 (scalar)
  * Rust `Vec`
  * 1 ~ 4 維陣列切片

👉 範例程式：[tensor_new.rs](../../tests/tensor_new.rs)

### 2. 使用 `Tensor::from_vec` 透過 Rust `Vec` 建立張量

`Tensor::from_vec()` 接收一個 `Vec` 並生成指定維度的張量。

* 第二個參數要能轉換成 `Shape`，可以是：
  * 空元組 ()：產生純量。
  * `usize`：產生向量
  * 數組 (Tuple)：生成 1～6 維的張量。
  * `Vec<usize>`：同樣可生成 1～6 維張量。

👉 範例程式：[tensor_from_vec.rs](../../tests/tensor_from_vec.rs)

### 3. 使用 `Tensor::from_slice` 透過 Slice 建立張量

和 `from_vec` 類似，使用 Slice 建立張量。

👉 範例程式：[tensor_from_slice.rs](../../tests/tensor_from_slice.rs)

### 4. 使用 `Tensor::from_iter` 透過 Iterator 建立張量

`Tensor::from_iter()` 接受實作了 `IntoIterator` 特徵(`trait`)的資料，生成**向量**。

👉 範例程式：[tensor_from_iter.rs](../../tests/tensor_from_iter.rs)

### 5. 使用 `Tensor::arange` 與 `Tensor::arange_step` 建立等差數列張量

* `Tensor::arange`：給定範圍 [start, end) 並以 1 為間隔。
* `Tensor::arange_step`: 可以自定間隔。⚠️ 注意：若使用浮點數，可能會因精度問題導致意外結果。

👉 範例程式：[tensor_arange.rs](../../tests/tensor_arange.rs)

## 二、建立特殊內容張量

### 1. 全是為 0 的張量

* `Tensor::zeros` 給定維度，資料型別與裝置，產生全為 **0** 的張量。`Tensor::zeros` 第二個參數是 `DType`，可以是：`U8, U32, I64, BF16, F16, F32, F64`。
* `t.zeros_like` 從已知的張量 `t` 產生全為 **0** 的張量。並與原張量具有相同維度和裝置。

👉 範例程式：[tensor_zeros.rs](../../tests/tensor_zeros.rs)

### 2. 全是 1 的張量

`Tensor::ones`，`t.ones_like` 與 `t.zeros`，`t.zeros_like` 類似，只是產生全為 **1** 張量。

👉 範例程式：[tensor_ones.rs](../../tests/tensor_ones.rs)

### 3. 指定值的張量

`Tensor::full` 給定一個數值，生成所有元素都為該數值的張量。

👉 範例程式：[tensor_full.rs](../../tests/tensor_full.rs)

### 4. 隨機內容的張量

`Tensor::rand`: 生成在範圍 [lo, up) 內的隨機值。
`t.rand_like`: 基於既有張量 `t`，產生相同維度的隨機張量。

👉 範例程式：[tensor_rand.rs](../../tests/tensor_rand.rs)

### 5. 常態分佈的張量

* `Tensor::randn`: 生成具有指定平均數與標準差的常態分佈張量。
`t.randn_like` 基於既有張量 `t` 產生相同維度的常態分佈張量。

👉 範例程式：[tensor_randn.rs](../../tests/tensor_randn.rs)

## 三、張量的維度操作

### 1. 取得張量維度 `t.rank`

`t.rank()`：取得張量的維度數，例如：

* **0**: 純量。
* **1**: 向量。
* **N > 1**: N 維張量。

### 2. 取得各維度大小 `t.dims()`, `t.dimsN`, `t.dim(index)` and `D::Minus`

* `t.dims()`: 回傳每個維度的大小，例如 **2x3x4** 張量會回傳 `&[2, 3, 4]`。
* `t.dimsN`: 如已知張量維度，可以利用 `t.dimsN` 取得所有維度的大小。如：

    ```rust
    let (d0, d1, d2) = t.dims3()?;
    ```

* `t.dim(index)`: 取得指定維度大小，像是 `t.dim(0)? 會回傳第一個維度的大小。
* `D::Minus`: 倒數第幾個維度
  * `D::Minus1`，是指張量的最後一個維度，如 2x3x4 張量，`t.dim(D::Minus1)?` 取得最後一個維度大小 **4**。
  * `D::Minus2`，是指張量的倒數第二個維度，如 2x3x4 張量，`t.dim(D::Minus2)?` 取得倒數第二個維度大小 **3**。
  * `D::Minus1` 在之後操作中，會經常使用到。

### 3. 由既有張量產生新維度張量 `t.reshape`

`t.reshape`: 利用既有張量，產生新維度張量。新張量的元素數量必須與舊張量相同。如 **24** 個元素的向量，可以產生新的 **2x3x4** 3D 張量。

```rust
// vector with 24 elements
let t = Tensor::arange(0u32, 24u32, &Device::Cpu)?;
// convert to 2x3x4 張量
let t = t.reshape((2, 3, 4))?;
```

👉 範例程式：[tensor_dim.rs](../../tests/tensor_dim.rs)

## 四、取出張量的內容

要取出張量內容，首先需先知道張量的維度，再使用 `t.to_scalar::<DataType>()` 或 `t.to_vecN::<DataType>()` 取出張量內容。

1. 純量取值: `t.to_scalar` 或 `t.to_vec0`。
1. 向量取值: `t.to_vec1`。
1. 2D 張量取值: `t.to_vec2`。
1. 3D 張量取值: `t.to_vec3`。

在取出張量內容時，需留意張量的維度是否與取出的資料型別相符，否則會產生錯誤。

👉 範例程式：[tensor_get_values.rs](../../tests/tensor_get_values.rs)

## 五、透過既有的張量，產生新型別張量 `t.to_dtype`

利用 `t.to_dtype` 可以透過既有的張量，產生新的資料型別張量，如從 `u32` 轉換成 `f32`。新的張量會與原來的張量有相同維度，並在相同裝置上。

👉 範例程式：[tensor_to_dtype.rs](../../tests/tensor_to_dtype.rs)

## 六、張量四則運算 `t.add`, `t.sub`, `t.mul`, and `t.div`

兩個張量四則運算，可以透過 `t.add`, `t.sub`, `t.mul`, and `t.div` 或直接使用 `+`, `-`, `*`, `/` 進行運算。
⚠️ 注意：兩個張量的維度和型別必須相同，不然會出錯。

除法時，如果除數是 **0**，會得到 `NaN`，或者 `Inf` / `-Inf`。

👉 範例程式：[tensor_arithmetic.rs](../../tests/tensor_arithmetic.rs)

## 七、矩陣乘法 `t.matmul`

遵守線性代數矩陣乘法規則，即第一個張量的最後一個維度大小必須與第二個張量的第一個維度大小相同。張量矩陣相乘，可以透過 `t.matmul` 進行。

👉 範例程式：[tensor_matmul.rs](../../tests/tensor_matmul.rs)

## 八、不同維度的張量四則運算 `t.broadcast_add`, `t.broadcast_sub`, `t.broadcast_mul`, and `t.broadcast_div`

如果兩個張量的維度不同時，如要進行四則運算，就需要使用 `broadcast_xxx` 相關函式。由於目前 Candle 目前沒有提供相關說明，因此透過研究源碼，整理運算流程。

### 1. 左、右運算元的張量維度是否相容 `Shape::broadcast_shape_binary_op`

在進行 broadcast 運算，需先確認兩個運算元的張量維度是否相容，透過研究 `Shape::broadcast_shape_binary_op`，整理維度相容的規則如下：

1. 純量張量與任何張量相容。
1. 從右到左比較維度大小，比對兩個張量的每層維度大小，如果兩者維度大小相同，則相容。
1. 承上，如果倒序過程，遇到維度大小為 **1**，則相容。

假設 `t1`, `t2` 是不同維度的張量，

1. `t1` 為純量張量，則與 `t2` 相容。
1. `t1` 是向量，如 `t2` 最後維度的大小與 `t1` 相同，則相容。比如：`t1` 是 **4** 個元素向量，如 `t2` 是 MxNx**4** 張量，則相容；如 `t2` 是 Nx**3** 張量，則不相容。
1. `t1` 是 **1**x4 張量，如 `t2` 是 MxNx**4** 張量，則相容；如 `t2` 是 Nx**3** 張量，則不相容。
1. `t1` 是 **3x4** 張量，如 `t2` 是 Mx**3x4** 張量，則相容，如 `t2` 是 Mx**3x3** 張量，或 Mx**2x4**，則不相容。

### 2. 左、右運算元的張量維度調整成一致 `t.broadcast_as`

在取得相容的維度後，透過 `t.broadcast_as` 函式，將兩個張量的維度調整成一致。

### 3. 使用張量四則運算計算結果

在源碼 **tensor.rs** 中，

```rust
broadcast_binary_op!(broadcast_add, add);
broadcast_binary_op!(broadcast_mul, mul);
broadcast_binary_op!(broadcast_sub, sub);
broadcast_binary_op!(broadcast_div, div);
```

透過 `broadcast_binary_op!` 定義 `broadcast_xxx` 函式，使用對應的四則運算函式進行運算。

👉 範例程式：[tensor_broadcast.rs](../../tests/tensor_broadcast.rs)

## 九、不同維度的張量矩陣乘法 `t.broadcast_matmul`

與其他的 `broadcast_xxx` 四則運算類似，必須先將兩個張量維度調整成相容後，再進行矩陣乘法。依源碼 `Shape::broadcast_shape_matmul` 規則，整理維度相容的規則如下：

1. 兩個張量最後兩個維度大小，必須符合矩陣乘法的規則。
1. 之後由右往左順序的維度大小，必須符合 broadcast 的規則。

👉 範例程式：[tensor_broadcast_matmul.rs](../../tests/tensor_broadcast_matmul.rs)

## 十、復盤

1. 建立張量的方法
   * Shape: 維度設定與操作
   * DType: 指定型別與 `to_dtype` 轉換。
   * `reshape` 產生新維度張量。
1. 張量四則運算與矩陣乘法
   * `add`, `sub`, `mul`, `div`
   * `+`, `-`, `*`, `/`
   * `matmul`
1. 不同維度張量四則運算與矩陣乘法
   * 張量維度相容性判斷
   * 在不同維度下，矩陣乘法的維度相容性判斷。
   * `broadcast_add`, `broadcast_sub`, `broadcast_mul`, `broadcast_div`, `broadcast_matmul`

這就是如何在 Candle 中操作張量的簡單教學！希望對你有幫助！ 🎉

## 十一、下一步

前往 [tensor-adv-op](../tensor-adv-op/README.md) 了解其他張量操作。
