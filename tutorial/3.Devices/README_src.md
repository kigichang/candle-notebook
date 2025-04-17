---
markdown:
  image_dir: assets
  path: README.md
  ignore_from_front_matter: true
  absolute_image_path: false

export_on_save:
  markdown: true
---
# Candle 硬體加速

目前我的工作環境有 Macbook Pro / Apple M1 Pro 與 Intel i5 / Ubuntu 20.04 / NVIDIA RTX 4090。本範例皆在以上的環境下進行。

在 Mac OS 開發上比較單純，只要安裝 XCode command line tools 就可以了。Apple 加速支援有：

- [Accelerate](https://developer.apple.com/documentation/accelerate): 提供 BLAS 與 LAPACK 加速。:eyes: [accelerate-src](https://github.com/blas-lapack-rs/accelerate-src)
- [Metal](https://developer.apple.com/metal/): 提供 GPU 加速。

Intel 加速有 [Intel Math Kernel Library (oneMKL)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html#gs.l4dkt9)。:eyes: [intel-mkl-src](https://github.com/rust-math/intel-mkl-src)。因為 intel-mkl-src 的版本太久沒更新了，之前測試 2024 版本會編譯失敗，目前我的環境是 2023.1.0 是可以正常編譯。

Candle 支援 Nvidia Cuda，需要安裝 Cuda Toolkit，建議是 12.0 以上的版本。注意搭配的 GCC 版本。

## 專案環境設定

@import "Cargo.toml" {as=toml}

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

@import "src/main.rs" {as=rust}

### 避免 Intel MKL 或 Accelerate 編譯失敗

依官方的做法，在 `main.rs` 上，加以下的程式碼，避免 Intel MKL 或 Accelerate 編譯失敗。

```rust
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;
```

:eyes: 參考官方的[Common Errors](https://github.com/huggingface/candle?tab=readme-ov-file#common-errors)

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

在取得 device 後，在產生 `VarBuilder` 或 `Tensor` 時，帶入 `device` 即可。__Tensor__ 就會載入到 VRAM 或共用的 RAM 上。需要留意的是 `to_vec` 的操作，如 `to_vec2`，會將 `Tensor` 轉移到 CPU 使用的 RAM 上，會花費一些時間，通常都是拿最後的結果使用，不要在運算過程中使用，以免影響效能。
