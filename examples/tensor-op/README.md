# 張量 (Tensor) 建立與操作

此範例主要以 Rust Test 展示張量操作，方後日後開發參考用。

## 張量建立與轉型

Candle 張量支援的資料型別 (DType) 有：`U8, U32, I64, BF16, F16, F32, F64`。對應的 Rust 資料型別如下：

| DType | Rust Type |
| --- | --- |
| U8 | u8 |
| U32 | u32 |
| I64 | i64 |
| BF16 | half::bf16 |
| F16 | half::f16 |
| F32 | f32 |
| F64 | f64 |

在建立張量時，需留意給定的 Rust 資料型別是否符合 Candle 張量支援的 DType。

### 建立張量 `Tensor::new`

`Tensor::new()` 依輸入的內容，在指定的裝置上建立一個新的張量。
`new` 第一個參數是 `NdArray`，可以是：

1. scalar
1. Vector
1. 1 ~ 4 維陣列 (array)

see [tensor_new.rs](../../tests/tensor_new.rs)

### 從 Vector 建立張量 `Tensor::from_vec`

`Tensor::from_vec()` 給定 Vector 在指定的裝置上，產生指定維度的張量。
`from_vec` 的第二個參數是 `Into<Shape>`，即可以轉換成 `Shape` 的型別，常用型別有：

1. `()` 空元組，產生純量張量。
1. `usize` 產生向量。
1. tuple，可用來產生 1 ~ 6 維的張量。
1. `Vec<usize>` 產生 1 ~ 6 維的張量。

see [tensor_from_vec.rs](../../tests/tensor_from_vec.rs)

### 從切片 (slice) 建立張量 `Tensor::from_slice`

`Tensor::from_slice` 與 `Tensor::from_vec()` 類似，給定 Slice 在指定的裝置上，產生指定維度的張量。

see [tensor_from_slice.rs](../../tests/tensor_from_slice.rs)

### 從 iterator 建立張量 `Tensor::from_iter`

`Tensor::from_iter` 產生向量(一維張量)。

see [tensor_from_iter.rs](../../tests/tensor_from_iter.rs)

### 從等差數列建立張量 `Tensor::arange` and `Tensor::arange_step`

`Tensor::arange` 給定 [start, end)，即大於等於start 且小於 end 範圍，等差為 **1**，產生向量。
`Tensor::arange_step` 給定 [start, end) 與等差，產生向量。

在使用 `arnge_step` 需留意浮點數的精度問題，可能會產生不預期的結果。

see [tensor_arange.rs](../../tests/tensor_arange.rs)

### 建立內容全是 0 的張量 `Tensor::zeros` and `t.zeros_like`

`Tensor::zeros` 給定維度，資料型別與裝置，產生全為 **0** 的張量。
`Tensor::zeros` 第二個參數是 `DType`，可以是：`U8, U32, I64, BF16, F16, F32, F64`。

`t.zeros_like` 從已知的張量 `t` 產生全為 **0** 的張量。新的張量會與原來的張量有相同維度，並在相同裝置上。

see [tensor_zeros.rs](../../tests/tensor_zeros.rs)

### 建立內容全是 1 的張量 `Tensor::ones` and `Tensor::ones_like`

`Tensor::ones`，`t.ones_like` 與 `t.zeros`，`t.zeros_like` 類似，只是產生全為 **1** 張量。

see [tensor_ones.rs](../../tests/tensor_ones.rs)

### 產生內容全是指定數值的張量 `Tensor::full`

`Tensor::full` 給定數值，維度裝置，產生全為指定值的張量。

see [tensor_full.rs](../../tests/tensor_full.rs)

### 產生隨機內容的張量 `Tensor::rand` and `t.rand_like`

`Tensor::rand` 給定隨機值範圍 [lo, up)，維度與裝置，產生隨機數值的張量。
`t.rand_like` 從已知的 Tensor `t` 產生隨機數值的張量。新的張量會與原來的張量有相同維度，資料型別，並在相同裝置上。

see [tensor_rand.rs](../../tests/tensor_rand.rs)

### 產生常態分佈內容的張量 `Tensor::randn` and `t.randn_like`

`Tensor::randn` 給定平均、標準差，維度與裝置，產生從給定平均、標準差的常態分佈中取樣的張量。
`t.randn_like` 從已知的張量 `t` 產生從常態分佈中取樣的張量。新的張量會與原來的張量有相同維度，資料型別，並在相同裝置上。

see [tensor_randn.rs](../../tests/tensor_randn.rs)

### 張量維度 `t.rank`, `t.dim`, and `t.reshape`

#### 取得張量維度 `t.rank`

由 `t.rank()` 回傳值，可以取得張量的維度：

**0**: 純量
**1**: 向量
**n >= 1**: n 維矩陣

#### 取得每個維度大小 `t.dims` and `t.dimsN`

如要取得每個維度的大小，可以使用 `t.dims()`，回傳 `&[usize]`，即每個維度的大小，如 2x3x4 矩陣，回傳 `[2, 3, 4]`。
如已知張量維度，可以利用 `t.dimsN` 取得所有維度的大小。如已知張量維度為 **3**，則 `let (d0, d1, d2) = t.dims3()?;` 取得所有維度的大小。

#### 取得每個維度大小 `t.dim` 與 `D::Minus`

`Tensor::dim` 取得指定維度的大小，如 2x3x4 矩陣，`t.dim(0)?` 取得第一個維度大小 **2**。

`D::Minus1`，是指張量的最後一個維度，如 2x3x4 矩陣，`t.dim(D::Minus1)?` 取得最後一個維度大小 **4**。
`D::Minus2`，是指張量的倒數第二個維度，如 2x3x4 矩陣，`t.dim(D::Minus2)?` 取得倒數第二個維度大小 **3**。
`D::Minus1` 在之後操作中，會經常使用到。

#### 利用既有張量，產生新維度張量 `t.reshape`

`reshape` 可以透過既有的張量，產生一個指定維度的新張量，新張量的總元素數量必須與舊張量相同。如 **24** 個元素的向量，可以產生新的 **2x3x4** 矩陣。

```rust
// vector with 24 elements
let t = Tensor::arange(0u32, 24u32, &Device::Cpu)?;
// convert to 2x3x4 矩陣
let t = t.reshape((2, 3, 4))?;
```

see [tensor_dim.rs](../../tests/tensor_dim.rs)

### 取出張量內容 `t.to_scalar` and `t.to_vecX`

要取出張量內容，首先需先知道張量的維度，再使用 `to_scalar` 或 `to_vecX` 取出張量內容。

1. scalar: `t.to_scalar`
1. vector: `t.to_vec1`
1. 2維矩陣: `t.to_vec2`
1. 3維矩陣: `t.to_vec3`

在取出張量內容時，需留意張量的維度是否與取出的資料型別相符，否則會產生錯誤。

see [tensor_get_values.rs](../../tests/tensor_get_values.rs)

### 透過既有的張量，產生新型別張量 `t.to_dtype`

利用 `t.to_dtype` 可以透過既有的張量，產生新的資料型別張量，如從 `u32` 轉換成 `f32`。新的張量會與原來的張量有相同維度，並在相同裝置上。

see [tensor_to_dtype.rs](../../tests/tensor_to_dtype.rs)

## arithmetic

1. add
1. sub
1. mul
1. div

## broadcast op

1. `t.broadcase_add`

## sum, mean and keepdim

## max, min and keepdim

## argmax and argmin

## stack and cat

## transpose

## matmul

## index and contiguous
