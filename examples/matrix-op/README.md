# Candle 基本操作

本範例是使用 [ratatui](https://ratatui.rs/) 中的 Canvas 功能，並使用 Candle 基本向量與矩陣運算，來實作 2D 繪圖。

```sh
$ cargo run --example matrix-op
```

並使用方向鍵移動；使用 `a` 與 `s` 鍵來做旋轉。

## 範例說明

### 矩形頂點

範例中，將矩形上的 4 個點，組成一個項點矩陣(4x2)，如下：

$$
\begin{bmatrix}
x_{1} & y_{1} \\
x_{2} & y_{2} \\
x_{3} & y_{3} \\
x_{4} & y_{4}
\end{bmatrix}
$$

### 中心點

在進行旋轉時，會以矩形的中心點為基準，計算旋轉後的新位置。計算中心點的公式如下：

$$
C_{x} = \frac{x_{1} + x_{2} + x_{3} + x_{4}}{4}
$$
$$
C_{y} = \frac{y_{1} + y_{2} + y_{3} + y_{4}}{4}
$$

在做旋轉前，會先將中心點移到原點，得到新的頂點矩陣，公式如下：

$$
\begin{bmatrix}
x_{1} & y_{1} \\
x_{2} & y_{2} \\
x_{3} & y_{3} \\
x_{4} & y_{4}
\end{bmatrix}-
\begin{bmatrix}
C_{x} & C_{y} \\
C_{x} & C_{y} \\
C_{x} & C_{y} \\
C_{x} & C_{y}
\end{bmatrix} =
\begin{bmatrix}
x_{1}^{c} & y_{1}^{c} \\
x_{2}^{c} & y_{2}^{c} \\
x_{3}^{c} & y_{3}^{c} \\
x_{4}^{c} & y_{4}^{c}
\end{bmatrix}
$$

### 旋轉

一般做 2D 旋轉，會使用下列公式：

$$
\begin{bmatrix}
cos\theta  & -sin\theta  \\
sin\theta & cos\theta \\
\end{bmatrix}
\begin{bmatrix}
x \\ y
\end{bmatrix}
$$

在範例中的旋轉計算，是項點矩轉，乘上轉置後的旋轉矩陣，如下：

$$
\begin{bmatrix}
x_{1}^{c} & y_{1}^{c} \\
x_{2}^{c} & y_{2}^{c} \\
x_{3}^{c} & y_{3}^{c} \\
x_{4}^{c} & y_{4}^{c}
\end{bmatrix}
\begin{bmatrix}
cos\theta  & -sin\theta  \\
sin\theta & cos\theta \\
\end{bmatrix}^T =
\begin{bmatrix}
x_{1}^{c} & y_{1}^{c} \\
x_{2}^{c} & y_{2}^{c} \\
x_{3}^{c} & y_{3}^{c} \\
x_{4}^{c} & y_{4}^{c}
\end{bmatrix}
\begin{bmatrix}
cos\theta  & sin\theta  \\
-sin\theta & cos\theta \\
\end{bmatrix} =
\begin{bmatrix}
x_{1}^{r} & y_{1}^{r} \\
x_{2}^{r} & y_{2}^{r} \\
x_{3}^{r} & y_{3}^{r} \\
x_{4}^{r} & y_{4}^{r}
\end{bmatrix}
$$

### 位移

將旋轉後的頂點矩陣，加上指定的 x, y 位移，即可得到位移後的頂點矩陣，如下：

$$
\begin{bmatrix}
x_{1}^{r} & y_{1}^{r} \\
x_{2}^{r} & y_{2}^{r} \\
x_{3}^{r} & y_{3}^{r} \\
x_{4}^{r} & y_{4}^{r}
\end{bmatrix}-
\begin{bmatrix}
d_{x} & d_{y} \\
d_{x} & d_{y} \\
d_{x} & d_{y} \\
d_{x} & d_{y}
\end{bmatrix} =
\begin{bmatrix}
x_{1}^{d} & y_{1}^{d} \\
x_{2}^{d} & y_{2}^{d} \\
x_{3}^{d} & y_{3}^{d} \\
x_{4}^{d} & y_{4}^{d}
\end{bmatrix}
$$

### 回到最初的位置

由於旋轉前，將中心點移至原點，最後需要回到原本的位置，計算如下：

$$
\begin{bmatrix}
x_{1}^{d} & y_{1}^{d} \\
x_{2}^{d} & y_{2}^{d} \\
x_{3}^{d} & y_{3}^{d} \\
x_{4}^{d} & y_{4}^{d}
\end{bmatrix}+
\begin{bmatrix}
C_{x} & C_{y} \\
C_{x} & C_{y} \\
C_{x} & C_{y} \\
C_{x} & C_{y}
\end{bmatrix}=
\begin{bmatrix}
x_{1}^{'} & y_{1}^{'} \\
x_{2}^{'} & y_{2}^{'} \\
x_{3}^{'} & y_{3}^{'} \\
x_{4}^{'} & y_{4}^{'}
\end{bmatrix}
$$

## 範例中使用到的 Candle 函式

### Tensor 簡介

與 Pytorch 相同，Candle 操作的基本單位是 Tensor，Tensor 內的資料，可以是：

1. scalar: 純量，簡單來說，就是單一數值。
1. vector: 向量，也就是一維陣列。
1. matrix: 2 維或 3 維矩陣。

Candle 支援的資料型別有：

- f32
- f64
- i64
- u8
- u32
- bf16
- f16

但不同的平台 (CPU, GPU) 有不同的支援度，在練習的時候，可以預設使用 **f32**。

### 建立 Tensor

```rust
Tensor::new(&[[1f32, 2.], [3., 4.]], &Device::Cpu)?
```

### 運算

### 中心點計算

```rust
let centroid = self.points.mean(0)?
```

### 中心點移至原點

```rust
let points = self.points.broadcast_sub(&centroid)?;
```

### 旋轉計算

```rust
let (sin, cos) = (angle.sin(), angle.cos());
let rotation = Tensor::from_vec(&[[cos, sin], [-sin, cos]])?;
let points = points.matmul(&rotation.t()?)?;
```

### 位移計算

```rust
let displacement = Tensor::from_vec(&[dx, dy], &Device::Cpu)?;
let points = points.broadcast_add(&displacement)?;
```

### 回到原本位置

```rust
let points = points.broadcast_add(&centroid)?;
```

## 復盤

1. `Tensor::new` 建立 Tensor
1. `t.broadcast_add` and `t.broadcast_sub`
1. `t.matmul` and `t.t()` to transpose
1. `t.mean` to calculate mean
