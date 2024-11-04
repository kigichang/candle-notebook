# CheatSheet

## 型別對應

| DType | 對應 Rust 型別 |
|:-----:| ------------- |
| U8    | u8            |
| U32   | u32           |
| I64   | i64           |
| BF16  | half::bf16    |
| F16   | half::f16     |
| F32   | f32           |
| F64   | f64           |

## Functions

|                |  Candle                                                                  | 說明 |
|:--------------:|--------------------------------------------------------------------------|-----|
| **Creation**   | [`Tensor::new`](tests/tensor_new.rs)                                     | 由 `NdArray` 建立新張量。|
|                | [`Tensor::from_vec`](tests/tensor_from_vec.rs)                           | 透過 Rust `Vec` 建立張量。|
|                | [`Tensor::from_slice`](tests/tensor_from_slice.rs)                       | 透過 Slice 建立張量。|
|                | [`Tensor::from_iter`](tests/tensor_from_iter.rs)                         | 透過 Iterator 建立張量。|
|                | [`Tensor::arange` and `Tensor::arange_step`](tests/tensor_from_iter.rs)  | 建立等差數列張量。|
|                | [`Tensor::zeros` and `t.zeros_like`](tests/tensor_zeros.rs)              | 全是為 0 的張量。|
|                | [`Tensor::ones` and `t.ones_like`](tests/tensor_ones.rs)                 | 全是為 1 的張量。|
|                | [`Tensor::full`](tests/tensor_full.rs)                                   | 全是指定值的張量。|
|                | [`Tensor::rand` and `t.rand_like`](tests/tensor_rand.rs)                 | 隨機張量。|
|                | [`Tensor::randn` and `t.randn_like`](tests/tensor_randn.rs)              | 隨機常態分佈張量。|
| **Dimension**  | [`t.rank`](tests/tensor_dim.rs)                                          | 取得張量的維度，如: 3D 張量回傳 **3**。|
|                | [`t.shape`](tests/tensor_dim.rs)                                         | 取得張量的形狀，張量各維度的大小資訊，都是由此提供。|
|                | [`t.dims` and `t.dimsN`](tests/tensor_dim.rs)                            | 取得張量每個維度大小。|
|                | [`t.dim(index)` and `D::Minus`](tests/tensor_dim.rs)                     | 取得指定維度大小。|
|                | [`t.reshape`](tests/tensor_dim.rs)                                       | 由既有張量，產生新維度張量。|
| **Get Valus**  | [`t.to_scalar` and `t.to_vecN`](tests/tensor_get_values.rs)              | 取得張量的內容。|
| **Dtype**      | [`tensor.to_dtype()`](tests/tensor_to_dtype.rs)                          | 由既有張量，產生新資料型別張量。|
| **Arithmetic** | [`&a + &b` or `a.add(&b)`](tests/tensor_arithmetic.rs)                   | 同張量維度四則運算。|
| **Broadcast**  | [`a.broadcast_xxx(&b)`](tests/tensor_broadcast.rs)                       | 不同張量形狀四則運算。|
|                | [`a.broadcast_matmul(&b)`](tests/tensor_broadcast_matmul.rs)             | 不同張量形狀矩陣乘法。|
|                | [`t.broadcast_maximum`](tests/tensor_broadcast_max_and_min.rs)           | 不同形狀張量比較大小，取最大值。 |
|                | [`t.broadcast_minimum`](tests/tensor_broadcast_max_and_min.rs)           | 不同形狀張量比較大小，取最小值。 |
|                | [t.broadcast_eq](tests/tensor_broadcast_compare.rs)                      | 比較兩個不同形狀的張量內容是否相等。|
|                | [t.broadcast_ne](tests/tensor_broadcast_compare.rs)                      | 比較兩個不同形狀的張量內容是否不相等。|
|                | [a.broadcast_lt(&b)](tests/tensor_broadcast_compare.rs)                  | 比較兩個不同形狀的張量內容，a 的是否小於 b。|
|                | [a.broadcast_le(&b)](tests/tensor_broadcast_compare.rs)                  | 比較兩個不同形狀的張量內容，a 的是否小於等於 b。|
|                | [a.broadcast_gt(&b)](tests/tensor_broadcast_compare.rs)                  | 比較兩個不同形狀的張量內容，a 的是否大於 b。|
|                | [a.broadcast_ge(&b)](tests/tensor_broadcast_compare.rs)                  | 比較兩個不同形狀的張量內容，a 的是否大於等於 b。|
| **Operations** | [`a.matmul(&b)?`](tests/tensor_matmul.rs)                                | 張量矩陣乘法。 |
|                | [`t.squeeze`](tests/tensor_squeeze.rs)                                   | 張量降維。 |
|                | [`t.unsqueeze`](tests/tensor_squeeze.rs)                                 | 張量升維。 |
|                | [`t.sum`](tests/tensor_sum.rs)                                           | 計算張量某個維度的總和或整個張量總和。 |
|                | [`t.mean`](tests/tensor_mean.rs)                                         | 計算張量某個維度的平均或整個張量平均。 |
|                | [`t.max`](tests/tensor_max.rs)                                           | 取得張量某個維度的最大值。 |
|                | [`t.min`](tests/tensor_min.rs)                                           | 計算張量某個維度的最小值。 |
|                | [`t.maximum`](tests/tensor_maximum.rs)                                   | 兩個張量比較大小，取最大值。 |
|                | [`t.minimum`](tests/tensor_minimum.rs)                                   | 兩個張量比較大小，取最小值。 |
|                | [t.eq](tests/tensor_broadcast_compare.rs)                                | 比較兩個量內容是否相等。|
|                | [t.ne](tests/tensor_broadcast_compare.rs)                                | 比較兩個張量內容是否不相等。|
|                | [a.lt(&b)](tests/tensor_broadcast_compare.rs)                            | 比較兩個張量內容，a 的是否小於 b。|
|                | [a.le(&b)](tests/tensor_broadcast_compare.rs)                            | 比較兩個張量內容，a 的是否小於等於 b。|
|                | [a.gt(&b)](tests/tensor_broadcast_compare.rs)                            | 比較兩個張量內容，a 的是否大於 b。|
|                | [a.ge(&b)](tests/tensor_broadcast_compare.rs)                            | 比較兩個張量內容，a 的是否大於等於 b。|
| Transpose  | `tensor.t()?` and `t.transpose`                                              | - |
| Indexing   | `tensor.i((.., ..4))?`                                                       | - |
|            | [`t.argmax`](tests/tensor_arg.rs)                                            | - |
|            | [`t.argmin`](tests/tensor_arg.rs)                                            | - |
| Device     | `tensor.to_device(&Device::new_cuda(0)?)?`                                   | - |
| Saving     | `candle::safetensors::save(&HashMap::from([("A", A)]), "model.safetensors")?`| - |
| Loading    | `candle::safetensors::load("model.safetensors", &device)`                    | - |
