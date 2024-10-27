# 張量(Tensor)建立與操作

此範例主要以 Rust Test 展示張量操作，方後日後開發參考用。

## 張量建立與轉型

### new

`Tensor::new()` 依輸入的內容，在指定的裝置上建立一個新的張量。
`new` 第一個參數是 `NdArray`，可以是：

1. scalar
1. Vector
1. 1 ~ 4 維陣列 (array)

see [tensor_new.rs](../../tests/tensor_new.rs)

### from_vec

`Tensor::from_vec()` 給定 Vector 在指定的裝置上，產生指定維度的張量。
`from_vec` 的第二個參數是 `Into<Shape>`，即可以轉換成 `Shape` 的型別，常用型別有：

1. `()` 空元組，產生純量張量。
1. `usize` 產生向量。
1. tuple，可用來產生 1 ~ 6 維的張量。
1. `Vec<usize>` 產生 1 ~ 6 維的張量。

see [tensor_from_vec.rs](../../tests/tensor_from_vec.rs)

### from_slice

`Tensor::from_slice` 與 `Tensor::from_vec()` 類似，給定 Slice 在指定的裝置上，產生指定維度的張量。

see [tensor_from_slice.rs](../../tests/tensor_from_slice.rs)

### from_iter

`Tensor::from_iter` 產生向量(一維張量)。

see [tensor_from_iter.rs](../../tests/tensor_from_iter.rs)

### arange and arange_step

`Tensor::arange` 給定 [start, end)，即大於等於start 且小於 end 範圍，等差為 **1**，產生向量。
`Tensor::arange_step` 給定 [start, end) 與等差，產生向量。

在使用 `arnge_step` 需留意浮點數的精度問題，可能會產生不預期的結果。

see [tensor_arange.rs](../../tests/tensor_arange.rs)

### zeros and zeros_like

`Tensor::zeros` 給定維度，資料型別與裝置，產生全為 **0** 的張量。
`Tensor::zeros` 第二個參數是 `DType`，可以是：`U8, U32, I64, BF16, F16, F32, F64`。

`t.zeros_like` 從已知的張量 `t` 產生全為 **0** 的張量。新的張量會與原來的張量有相同維度，並在相同裝置上。

see [tensor_zeros.rs](../../tests/tensor_zeros.rs)

### ones and ones_like

`Tensor::ones`，`t.ones_like` 與 `t.zeros`，`t.zeros_like` 類似，只是產生全為 **1** 張量。

see [tensor_ones.rs](../../tests/tensor_ones.rs)

### full

`Tensor::full` 給定數值，維度裝置，產生全為指定值的張量。

see [tensor_full.rs](../../tests/tensor_full.rs)

### rand and rand_like

`Tensor::rand` 給定隨機值範圍 [lo, up)，維度與裝置，產生隨機數值的張量。
`t.rand_like` 從已知的 Tensor `t` 產生隨機數值的張量。新的張量會與原來的張量有相同維度，資料型別，並在相同裝置上。

### randn and randn_like

`Tensor::randn` 給定平均、標準差，維度與裝置，產生從給定平均、標準差的常態分佈中取樣的張量。
`t.randn_like` 從已知的張量 `t` 產生從常態分佈中取樣的張量。新的張量會與原來的張量有相同維度，資料型別，並在相同裝置上。

### to_dtype

DType: `U8, U32, I64, BF16, F16, F32, F64`

| DType | Rust Scalar Type |
| --- | --- |
| U8 | u8 |
| U32 | u32 |
| I64 | i64 |
| BF16 | half::bf16 |
| F16 | half::f16 |
| F32 | f32 |
| F64 | f64 |

## shape, dim and reshape

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

## view and reshape

## transpose

## matmul

## index and contiguous
