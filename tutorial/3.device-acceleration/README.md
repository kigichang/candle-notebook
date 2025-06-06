  
# Candle 硬體加速
  
目前我的工作環境有 Macbook Pro / Apple M1 Pro 與 Intel i5 / Ubuntu 20.04 / NVIDIA RTX 4090。
  
在 Mac OS 開發上比較單純，只要安裝 XCode command line tools 就可以了。Apple 加速支援有：
  
- [Accelerate](https://developer.apple.com/documentation/accelerate ): 提供 BLAS 與 LAPACK 加速。使用 [accelerate-src](https://github.com/blas-lapack-rs/accelerate-src )
- [Metal](https://developer.apple.com/metal/ ): 提供 GPU 加速。
  
Intel 加速有 [Intel Math Kernel Library (oneMKL)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html#gs.l4dkt9 )。因為 [intel-mkl-src](https://github.com/rust-math/intel-mkl-src ) 的版本太久沒更新了，之前測試 2024 版本會編譯失敗，目前我的環境是 2023.1.0 是可以正常編譯。
  
Candle 支援 Nvidia Cuda，需要安裝 Cuda Toolkit，建議是 12.0 以上的版本。注意搭配的 GCC 版本。
  
如果搞不定開發環境的話，可以使用 Docker 的方式來建立開發環境，先安裝好 nvidia-container-toolkit，再依照使用的 Cuda 版本，下載對應的 Nivida 開發 Image，比如我使用 Cuda 12.5 版本，就下載 `docker pull nvidia/cuda:12.5.1-cudnn-runtime-ubuntu20.04`。大致流程如下：
  
1. 下載 Image: `docker pull nvidia/cuda:12.5.1-cudnn-runtime-ubuntu20.04`
1. 啟動並保留環境: `docker run -d --name my_dev -v PROJECT_DIR:/workspace:z cuda:12.5.1-cudnn-runtime-ubuntu20.04 tail -f /dev/null`
1. 進入容器: `docker exec -it my_dev bash`
1. 安裝 curl，等會安裝 Rust 會用到: `apt update && apt install -y curl`
1. 安裝 Rust: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
1. 將 Rust 程式加入環境變數 `<img src="https://latex.codecogs.com/gif.latex?PATH`:%20`source"/>HOME/.cargo/env`
1. 進入專案目錄: `cd /workspace`
1. 編譯: `cargo build --release --features cuda`
  
## 專案環境設定
  
```toml
[package]
name = "candle-ex3"
version = "0.1.0"
edition = "2024"
  
[dependencies]
candle-core = "0.8.4"
candle-nn = "0.8.4"
accelerate-src = { version = "0.3.2", optional = true }
bindgen_cuda = { version = "0.1.5", optional = true }
intel-mkl-src = { version = "0.8.1", optional = true }
anyhow = "1.0.97"
clap = { version = "4.5.36", features = ["derive"] }
  
[features]
default = []
accelerate = ["dep:accelerate-src", "candle-core/accelerate", "candle-nn/accelerate"]
cuda = ["dep:bindgen_cuda", "candle-core/cuda", "candle-nn/cuda"]
mkl = ["dep:intel-mkl-src", "candle-core/mkl", "candle-nn/mkl"]
metal = ["candle-core/metal", "candle-nn/metal"]
  
```  
  
在專案環境設定上，可以仿照官方的做法，在 `[dependencies]` 區段加入：
  
```toml
accelerate-src = { version = "0.3.2", optional = true }
bindgen_cuda = { version = "0.1.5", optional = true }
intel-mkl-src = { version = "0.8.1", optional = true }
```
  
在 `[features]` 區段加入：
  
```toml
[features]
default = []
accelerate = ["dep:accelerate-src", "candle-core/accelerate", "candle-nn/accelerate"]
cuda = ["dep:bindgen_cuda", "candle-core/cuda", "candle-nn/cuda"]
mkl = ["dep:intel-mkl-src", "candle-core/mkl", "candle-nn/mkl"]
metal = ["candle-core/metal", "candle-nn/metal"]
```
  
在編譯時，指定要編譯的特性，例如：
  
```bash
cargo build --release --features accelerate # accelerate 加速
cargo build --release --features mkl # mkl 加速
cargo build --release --features metal # metal 加速
cargo build --release --features cuda # cuda 加速
```
  
p.s. 如果要 benchmark 或 release 正式版本，記得要加 `--release`，因為 __debug__ 模式的執行速度會很慢很慢。通常我是都直接使用 __release__ 模式來編譯。
  
## 完整程式碼
  
```rust
use anyhow::Result;
use candle_core::utils;
use candle_core::{Device, Tensor};
use candle_nn::{linear, Module, VarBuilder};
use clap::Parser;
  
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;
  
#[cfg(feature = "accelerate")]
extern crate accelerate_src;
  
// from: https://github.com/huggingface/candle/blob/main/candle-examples/src/lib.rs
pub fn device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if utils::cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if utils::metal_is_available() {
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
  
// This function prints the device environment information
// such as number of threads, CUDA availability, MKL availability, etc.
// It uses the `utils` module from the `candle_core` crate to get this information.
fn device_environment(device: &Device) {
    println!("cuda_is_available? {}", utils::cuda_is_available());
    println!("metal_is_available? {}", utils::metal_is_available());
    println!("has_mkl? {}", utils::has_mkl());
    println!("has_accelerate? {}", utils::has_accelerate());
    println!("with_avx? {}", utils::with_avx());
    println!("with_f16c? {}", utils::with_f16c());
    println!("with_neon? {}", utils::with_neon());
    println!("with_simd128? {}", utils::with_simd128());
    println!("num of threads: {}", utils::get_num_threads());
  
    println!("device: {:?}", device);
    println!("\tis cuda? {}", device.is_cuda());
    println!("\tis metal? {}", device.is_metal());
    println!("\tis cpu? {}", device.is_cpu());
    println!("\tlocation {:?}", device.location());
    println!("\tsupport bf16? {}", device.supports_bf16());
}
  
#[derive(Parser)]
struct Args {
    #[clap(long, default_value_t = false)]
    cpu: bool,
  
    #[clap(long, default_value_t = 500)]
    loops: usize,
}
  
fn main() -> Result<()> {
    let args = Args::parse();
  
    let device = device(args.cpu)?;
    device_environment(&device);
  
    let vb = VarBuilder::zeros(candle_core::DType::F32, &device);
    let ln = linear(1024, 768, vb.pp("ln1"))?;
    let input = Tensor::rand(-1.0f32, 1.0f32, (64, 1024), &device)?;
  
    let tm = std::time::Instant::now();
    for _i in 0..args.loops {
        let _result = ln.forward(&input)?;
        // _result.to_vec2::<f32>()?; // 將 tensor 資料，轉到 RAM 上，會花費時間。
    }
    println!("Time taken: {:?}", tm.elapsed());
  
    let weights = Tensor::rand(-1.0f32, 1.0f32, (768, 1024), &device)?;
    let bias = Tensor::rand(-1.0f32, 1.0f32, 768, &device)?;
  
    let tm = std::time::Instant::now();
    for _i in 0..args.loops {
        let _result = input.matmul(&weights.t()?)?;
        let _result = _result.broadcast_add(&bias)?;
    }
    println!("Time taken: {:?}", tm.elapsed());
  
    Ok(())
}
  
```  
  
### 避免 Intel MKL 或 Accelerate 編譯失敗
  
依官方的做法，在 `main.rs` 上，加以下的程式碼，避免 Intel MKL 或 Accelerate 編譯失敗。
  
```rust
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;
  
#[cfg(feature = "accelerate")]
extern crate accelerate_src;
```
  
:eyes: 參考官方的[Common Errors](https://github.com/huggingface/candle?tab=readme-ov-file#common-errors )
  
### 偵測硬體支援
  
在 `candle_core::utils` 有提供一些函式可以偵測硬體的支援情況，基本上最常用的是偵測是否有支援 CUDA 與 Metal。
  
- `utils::cuda_is_available()`: 是否支援 CUDA
- `utils::metal_is_available()`: 是否支援 Metal
  
```rust
// This function prints the device environment information
// such as number of threads, CUDA availability, MKL availability, etc.
// It uses the `utils` module from the `candle_core` crate to get this information.
fn device_environment(device: &Device) {
    println!("cuda_is_available? {}", utils::cuda_is_available());
    println!("metal_is_available? {}", utils::metal_is_available());
    println!("has_mkl? {}", utils::has_mkl());
    println!("has_accelerate? {}", utils::has_accelerate());
    println!("with_avx? {}", utils::with_avx());
    println!("with_f16c? {}", utils::with_f16c());
    println!("with_neon? {}", utils::with_neon());
    println!("with_simd128? {}", utils::with_simd128());
    println!("num of threads: {}", utils::get_num_threads());
  
    println!("device: {:?}", device);
    println!("\tis cuda? {}", device.is_cuda());
    println!("\tis metal? {}", device.is_metal());
    println!("\tis cpu? {}", device.is_cpu());
    println!("\tlocation {:?}", device.location());
    println!("\tsupport bf16? {}", device.supports_bf16());
}
```
  
Thread 的部分，Canle 使用 Rayon 套件，Thread 數預設會使用 CPU 的核心數量，這個可以透過 `RAYON_NUM_THREADS` 環境變數來設定。
  
### 使用 CPU 或 GPU
  
指定使用 CPU 或 GPU 的方式是使用 `candle_core::Device` 來指定。
  
- CPU: `Device::Cpu`
- CUDA: `Device::new_cuda(0)`，0 是 GPU 的 index，也就是第一張 GPU。
- Metal: `Device::new_metal(0)`，0 是 GPU 的 index，通常是 0。
  
以下是官方提供的範例，目前我都是直接引用。
  
```rust
// from: https://github.com/huggingface/candle/blob/main/candle-examples/src/lib.rs
pub fn device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if utils::cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if utils::metal_is_available() {
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
  
在取得 device 後，在產生 `VarBuilder` 或 `Tensor` 時，帶入 `device` 即可，__Tensor__ 就會載入到 VRAM 或共用的 RAM 上，如：
  
```rust
let vb = VarBuilder::zeros(candle_core::DType::F32, &device);
let ln = linear(1024, 768, vb.pp("ln1"))?;
let input = Tensor::rand(-1.0f32, 1.0f32, (64, 1024), &device)?;
  
let weights = Tensor::rand(-1.0f32, 1.0f32, (768, 1024), &device)?;
let bias = Tensor::rand(-1.0f32, 1.0f32, 768, &device)?;
```
  
需要留意的是 `to_vec` 的操作，如 `to_vec2`，會將 `Tensor` 轉移到 CPU 使用的 RAM 上，會花費一些時間，通常都是拿最後的結果使用，不要在運算過程中使用，以免影響效能。
  
```rust
for _i in 0..args.loops {
    let _result = ln.forward(&input)?;
    // _result.to_vec2::<f32>()?; // 將 tensor 資料，轉到 RAM 上，會花費時間。
}
```
  