# Tensor 建立與操作

此範例主要以 Rust Test 展示 Tensor 操作，方後日後開發參考用。

## create Tensor, to_device and to_dtype

- `Tensor::new()`
- `Tensor::zeros()` and  `t.zero_like()`
- `Tensor::ones()` and `Tensor::ones_like()`
- `Tensor::full()`
- `Tensor::rand()` and `t.rand_like()`
- `Tensor::randn()` and `t.randn_like()`
- `Tensor::from_vec`
- `Tensor::from_iter`
- `Tensor::arange()` and `Tensor::arange_step`
- `t.to_device()`
- `t.to_dtype()`

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

## shape and dim

## view and reshape

## transpose

## matmul

## index and contiguous
