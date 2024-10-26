# 如何用 Candle 來做 2D 繪圖

這篇範例介紹如何使用 Candle 提供的向量和矩陣運算，搭配 [ratatui](https://ratatui.rs/) 的 Canvas 功能，實作簡單的 2D 繪圖。

```sh
$ cargo run --example matrix-op
```

運行程式後，可以使用 **方向鍵** 移動圖形，按 **a** 和 **s** 旋轉圖形。

## 範例解說

### 矩形頂點

這個範例將矩形的 4 個頂點組成一個 4x2 矩陣，表示如下：

$$
\begin{bmatrix}
x_{1} & y_{1} \\
x_{2} & y_{2} \\
x_{3} & y_{3} \\
x_{4} & y_{4}
\end{bmatrix}
$$

### 中心點計算

旋轉圖形時，我們會以矩形的中心點為基準來計算新位置。中心點的公式如下：

$$
C_{x} = \frac{x_{1} + x_{2} + x_{3} + x_{4}}{4},
C_{y} = \frac{y_{1} + y_{2} + y_{3} + y_{4}}{4}
$$

在旋轉之前，我們要先將矩形的中心點移到原點。移動後的頂點矩陣如下：

$$
新的頂點矩陣 \begin{bmatrix}
x_{1}^{c} & y_{1}^{c} \\
x_{2}^{c} & y_{2}^{c} \\
x_{3}^{c} & y_{3}^{c} \\
x_{4}^{c} & y_{4}^{c}
\end{bmatrix}=
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
\end{bmatrix}
$$

### 旋轉

進行 2D 旋轉時，我們會用以下的旋轉矩陣：

$$
\begin{bmatrix}
x \\ y
\end{bmatrix}
\begin{bmatrix}
cos\theta  & -sin\theta  \\
sin\theta & cos\theta \\
\end{bmatrix}^T

$$

在範例中，我們使用頂點矩陣與轉置後的旋轉矩陣相乘：

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

接著，將旋轉後的頂點加上指定的 x 和 y 位移，得到位移後的新頂點：

$$
\begin{bmatrix}
x_{1}^{r} & y_{1}^{r} \\
x_{2}^{r} & y_{2}^{r} \\
x_{3}^{r} & y_{3}^{r} \\
x_{4}^{r} & y_{4}^{r}
\end{bmatrix}+
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

### 回到原位置

最後，我們再將原本移到原點的中心點加回來：

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

## Candle 的函式使用

### Tensor 簡介

和 PyTorch 一樣，Candle 的基本運算單位是 Tensor，它可以儲存：

1. scalar: 純量，即單一數值。
1. vector: 向量，即一維陣列。
1. matrix: 矩陣 (二維或三維陣列)。

Candle 支援的資料型別有：
`f32, f64, i64, u8, u32, bf16, f16`

不同平台 (CPU/GPU) 的支援度不一樣，平常可以預設使用 **f32**。

#### 建立 Tensor

```rust
Tensor::new(&[[1f32, 2.], [3., 4.]], &Device::Cpu)?
```

這段程式碼會在 **CPU** 上建立一個 Tensor。第二個參數是指定要使用 **CPU** 還是 **GPU**。

### 運算範例

#### 計算中心點

要計算矩形的中心點，我們可以用 `tensor.mean()`，並指定參數為 **0** (即第一維度)：

```rust
let centroid = self.points.mean(0)?
```

這相當於計算：

$$
C_{x} = \frac{x_{1} + x_{2} + x_{3} + x_{4}}{4},
C_{y} = \frac{y_{1} + y_{2} + y_{3} + y_{4}}{4}
$$

#### 中心點移至原點

由於頂點是 **4x2** 矩陣，而中心點是 **2 維**向量，要做運算前，需要使用 `broadcast` 相關函式：

```rust
let points = self.points.broadcast_sub(&centroid)?;
```

#### 旋轉矩陣計算

使用 `tensor.t()` 將旋轉矩陣轉置，再進行矩陣相乘 `t.matmul()`：

```rust
let (s, c) = theta.sin_cos();
let rotation = Tensor::new(&[[c, -s], [s, c]], &Device::Cpu?)
let points = points.matmul(&rotation.t()?)?;
```

#### 位移計算

由於位移 `displacement` 是向量，我們使用 `broadcast_add` 進行運算：

```rust
let displacement = Tensor::new(&[dx, dy], &Device::Cpu)?;
let points = points.broadcast_add(&displacement)?;
```

#### 回到原位置

回到原位置同樣使用 `broadcast_add`：

```rust
let points = points.broadcast_add(&centroid)?;
```

## 復盤

1. 用 `Tensor::new` 建立 Tensor。
1. 用 `t.mean()` 計算平均值。
1. 用 `t.broadcast_add()` 和 `t.broadcast_sub()` 做矩陣與向量運算。
1. 用 `t.t()` 進行矩陣轉置。
1. 用 `t.matmul()` 做矩陣乘法。

## Next

前往 [tensor-op](../tensor-op/README.md) 了解更多 tensor 操作。
