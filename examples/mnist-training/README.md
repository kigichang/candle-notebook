# Candle 模型訓練與 CPU/GPU 加速

這個範例展示如何使用 Candle 訓練一個手寫辨識模型，以及 CPU/GPU 加速。

在 Candle 的函式庫內，有關深度學習的函式，多定義 `candle_core` 與 `candle_nn`；另外有 `candle-transformers` 類比 HuggingFace 官方的 Transformers 套件。

## 1. MNIST

MNIST 是深度學習入門經典的例子，數據集包含 0 到 9 的手寫數字圖片，每張圖片是 28x28 影像。本範例是修改 [官方的 MNIST 範例](https://github.com/huggingface/candle/tree/main/candle-examples/examples/mnist-training)，部分改用 `Modlule`, `ModuleT`, 及 `Sequential` 來實作。

Pytorch 或 Tensorflow 為了方便開發者去堆疊多個神經網路，都有提供 `Sequential` 功能。Candle 也有類似的功能 `Sequential`，但礙於 Rust 不是 Script 程式語言，無法做到像 Pytorch 這樣方便。其實只要將每個神經層依序呼叫就可以達到 `Sequential` 的效果。

如果你對深度學習不是那麼熟悉，可以參考[陳縕儂老師的深度學習之應用](https://www.youtube.com/playlist?list=PLOAQYZPRn2V6kQfY453CIakozNGR_gQdA)。

### 1.1 如何執行

執行或編譯程式時，記得加上 --release，不然速度會很慢。

```bash
$ cargo run --release --example mnist-training 
```

這個官方的範例，提供了 **Linear**, **MLP**, 與 **CNN** 三種訓練方式，預設是 **Linear**。其他訓練方式如下：

- Linear:

```bash
$ cargo run --release --example mnist-training -- linear
```

- MLP:

```bash
$ cargo run --release --example mnist-training -- mlp
```

- CNN

```bash
$ cargo run --release --example mnist-training -- cnn
```

## 2. 程式碼解析

### 2.1 載入並查看資料集

使用 Candle 的 `candle_datasets` 模組，載入 MNIST 資料集。

```rust
let m = if let Some(directory) = args.local_mnist {
    candle_datasets::vision::mnist::load_dir(directory)?
} else {
    candle_datasets::vision::mnist::load()?
};
println!("train-images: {:?}", m.train_images.shape());
println!("train-labels: {:?}", m.train_labels.shape());
println!("test-images: {:?}", m.test_images.shape());
println!("test-labels: {:?}", m.test_labels.shape());
```

### 2.2 建立模型

使用 `VarMap` 與 `VarBuilder` 來建立模型。

```rust
let mut varmap = VarMap::new();
let vs = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
let model = M::new(vs.clone())?;
```

以 CNN 為例，可以使用 `varbuilder.pp()` 來為每一層神經網路命名。每一層神經網路的參數會被加上前綴，例如 `c1`, `c2`, `fc1`, `fc2`，並儲存在 `VarMap` 內。

```rust
impl Model for ConvNet {
    fn new(vs: VarBuilder) -> Result<Self> {
        let conv1 = candle_nn::conv2d(1, 32, 5, Default::default(), vs.pp("c1"))?;
        let conv2 = candle_nn::conv2d(32, 64, 5, Default::default(), vs.pp("c2"))?;
        let fc1 = candle_nn::linear(1024, 1024, vs.pp("fc1"))?;
        let fc2 = candle_nn::linear(1024, LABELS, vs.pp("fc2"))?;
        let dropout = candle_nn::Dropout::new(0.5);
        Ok(Self {
            conv1,
            conv2,
            fc1,
            fc2,
            dropout,
        })
    }
}
```

### 2.3 載入先前訓練的模型

使用 `varmap.load()` 載入先前訓練的模型。

```rust
if let Some(load) = &args.load {
    println!("loading weights from {load}");
    varmap.load(load)?
}
```

### 2.4 定義 Optimizer

Candle 提供了 `SGD`, `AdamW` 兩種 Optimizer。

- SGD:

```rust
let mut sgd = candle_nn::SGD::new(varmap.all_vars(), args.learning_rate)?;
```

- AdamW:

```rust
let adamw_params = candle_nn::ParamsAdamW {
    lr: args.learning_rate,
    ..Default::default()
};
```

### 2.5 訓練與測試模型

範例中的 `training_loop` 與 `training_loop_cnn` 不同點在於 `training_loop_cnn` 使用 Batch 的方式訓練。

- training_loop_cnn:

    ```rust
    for epoch in 1..args.epochs {
        let mut sum_loss = 0f32;
        batch_idxs.shuffle(&mut thread_rng());
        for batch_idx in batch_idxs.iter() {
            let train_images = train_images.narrow(0, batch_idx * BSIZE, BSIZE)?;
            let train_labels = train_labels.narrow(0, batch_idx * BSIZE, BSIZE)?;
            let logits = model.forward_t(&train_images, true)?;
            let log_sm = ops::log_softmax(&logits, D::Minus1)?;
            let loss = loss::nll(&log_sm, &train_labels)?;
            opt.backward_step(&loss)?;
            sum_loss += loss.to_vec0::<f32>()?;
        }
        let avg_loss = sum_loss / n_batches as f32;

        let test_logits = model.forward_t(&test_images, false)?;
        let sum_ok = test_logits
            .argmax(D::Minus1)?
            .eq(&test_labels)?
            .to_dtype(DType::F32)?
            .sum_all()?
            .to_scalar::<f32>()?;
        let test_accuracy = sum_ok / test_labels.dims1()? as f32;
        println!(
            "{epoch:4} train loss {:8.5} test acc: {:5.2}%",
            avg_loss,
            100. * test_accuracy
        );
    }
    ```

### 2.6 儲存模型

使用 `varmap.save()` 儲存 **safetensors** 格式的模型。目前官方推薦使用 **safetensors** 格式，而不是 Pytorch 格式。

```rust
if let Some(save) = &args.save {
    println!("saving trained weights in {save}");
    varmap.save(save)?
}
```

### 2.7 Module and MuduleT

Candle 提供類似 Pytorch 的 `Module` 功能，但由於 Rust 不是 Script 程式語言，無法像 Pytorch 一樣方便。

要使用 `Sequential` 來堆疊多個神經層，就必須實作 `Module`。`ModuleT` 與 `Module` 的差別在於 `ModuleT` 多了一個 `bool` 參數，用來區分是否為訓練模式。這個參數在此範例中，決定是否要執行 `Dropout`。在推論過程中，是不需要執行 `Dropout`。

透過以下的源始碼，可以得知只要實作 `Module` 就自動實作 `ModuleT`。

```rust

// A simple trait defining a module with forward method using a single argument.
pub trait Module {
    fn forward(&self, xs: &Tensor) -> Result<Tensor>;
}

// A trait defining a module with forward method using a single tensor argument and a flag to
// separate the training and evaluation behaviors.
pub trait ModuleT {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Result<Tensor>;
}

impl<M: Module> ModuleT for M {
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Result<Tensor> {
        self.forward(xs)
    }
}
```

### 2.8 Sequential

以 MLP 為例，`candle_cnn::Linear` 有實作了 `Module`；如果函式是 `fn xxx(xs: &Tensor) -> Result<Tensor>`，可以直接使用 `add` 加入 `Sequential`。

```rust
fn relu(xs: &Tensor) -> Result<Tensor> {
    xs.relu()
}

impl Model for Mlp {
    fn new(vs: VarBuilder) -> Result<Self> {
        let ln1 = candle_nn::linear(IMAGE_DIM, 100, vs.pp("ln1"))?;
        let ln2 = candle_nn::linear(100, LABELS, vs.pp("ln2"))?;
        //Ok(Self { ln1, ln2 })
        let seq = sequential::seq().add(ln1).add(relu).add(ln2);
        Ok(Self { seq })
    }
}
```

`fn xxx(xs: &Tensor) -> Result<Tensor>` 可以直接使用 `add` 加入 `Sequential` 的原因是，Candle 自動幫這類型的函式實作了 `Module`。

```rust
impl<T: Fn(&Tensor) -> Result<Tensor>> Module for T {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self(xs)
    }
}
```

### 2.9 實作細節

實作時，要留意以下實作的順序，否則訓練時會失敗：

1. 使用 `VarMap` 與 `VarBuilder` 先建立模型。
1. 設定 Optimizer。
1. 載入先前訓練的模型。

一定要先建立好模型，才能設定 Optimizer 與載入模型。

## 3. CPU/GPU 加速

範例程式預設嘗試使用 GPU 加速。我目前在 Intel + RTX4090 + Ubuntu 與 Apple M1 Pro 上環境開發與測試。

在撰寫程式時，如要使用 CPU 加速，請在程式碼上，加入：

```rust
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;
```

- `intel_mkl_src`: 應用在 Intel CPU。
- `accelerate_src`: 應用在 Apple M 系列晶片的 CPU 加速。

使用 GPU 硬體加速程式碼如下：

- `cuda_is_available`: 檢查是否有 CUDA 加速。
- `metal_is_available`: 檢查是否有 Metal 加速。
- `Device::new_cuda(0)`: 使用 CUDA 加速。
- `Device::new_metal(0)`: 使用 Metal 加速。

```rust
use candle_core::utils::{cuda_is_available, metal_is_available};
use candle_core::{DType, Device, Result, Shape, Tensor, WithDType};

// from: https://github.com/huggingface/candle/blob/main/candle-examples/src/lib.rs
pub fn device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            println!(
                "Running on CPU, to run on GPU(metal), build this example with `--features metal`"
            );
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            println!("Running on CPU, to run on GPU, build this example with `--features cuda`");
        }
        Ok(Device::Cpu)
    }
}
```

### 3.1 Mac OS 環境

Mac OS 不用再安裝額外的套件，即可使用硬體加速，也因此推薦 Mac 上練習。在執行或編譯時，加入 `--features "accelerate,metal"`，指定使用 `accelerate` 與 `metal`。

```bash
$ cargo run --release --example mnist-training --features "accelerate,metal"
```

### 3.2 Intel + nVidia RTX4090 + Ubuntu 環境

我另一組實驗的環境是 **Ubuntu**，安裝了 Intel-MKL 與 CUDA。MKL 開發套件不要安裝最新 24.x.x。目前我使用的版本是 23.1.0。

CUDA 安裝的版本是：12.5。Driver 版本是：555.42.02。

在執行或編譯時，加入 `--features "mkl,cuda"`，指定使用 `mkl` 與 `cuda`。

```bash
$ cargo run --release --example mnist-training --features "mkl,cuda"
```

## 4. 復盤

1. `VarMap` 載入與儲存 **safetensors** 格式的模型。
1. `Module` 與 `Sequential` 對比 Pytorch 的 `Module` 與 `Sequential`，但不像 Pytorch 那麼方便。
1. 訓練程式實作順序: 先建立模型，再設定 Optimizer 及載入模型。
1. CPU/GPU 加速: Mac OS 使用 `accelerate` 與 `metal`，Intel CPU + nVidia GPU 使用 `mkl` 與 `cuda`。
