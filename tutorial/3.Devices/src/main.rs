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
