---
markdown:
  path: README.md
export_on_save:
  markdown: true
---
# Tensor 建立與操作

此範例主要以 Rust Test 展示 Tensor 操作，方後日後開發參考用。

## create Tensor, to_device and to_dtype

### new

@import "main.rs" {code_block=true, as=rust, line_begin=6, line_end=40}

### zeros and zeros_like

@import "main.rs" {code_block=true, as=rust, line_begin=41, line_end=96}

### ones and ones_like

@import "main.rs" {code_block=true, as=rust, line_begin=97, line_end=152}

### full

@import "main.rs" {code_block=true, as=rust, line_begin=153, line_end=181}

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
