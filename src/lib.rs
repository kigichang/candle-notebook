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

/// 將純量轉換為張量。
pub fn scalar_to_tensor<T: WithDType, S: Into<Shape>>(
    v: T,
    shape: S,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    Tensor::new(v, &Device::Cpu)?
        .broadcast_as(shape)?
        .to_dtype(dtype)?
        .to_device(device)
}

/// 將兩個一維向量的每個元素進行比較，取最大值。
pub fn maximum_vec1<T: PartialOrd + Copy>(v1: &[T], v2: &[T]) -> Vec<T> {
    let mut v = vec![];
    for (a, b) in v1.iter().zip(v2.iter()) {
        v.push(if a > b { *a } else { *b })
    }
    return v;
}

/// 將兩個二維向量的每個元素進行比較，取最大值。
pub fn maximum_vec2<T: PartialOrd + Copy>(v1: &[Vec<T>], v2: &[Vec<T>]) -> Vec<Vec<T>> {
    let mut v = vec![];
    for (a, b) in v1.iter().zip(v2.iter()) {
        v.push(maximum_vec1(a, b));
    }
    return v;
}

/// 將兩個三維向量的每個元素進行比較，取最大值。
pub fn maximum_vec3<T: PartialOrd + Copy>(
    v1: &[Vec<Vec<T>>],
    v2: &[Vec<Vec<T>>],
) -> Vec<Vec<Vec<T>>> {
    let mut v = vec![];
    for (a, b) in v1.iter().zip(v2.iter()) {
        v.push(maximum_vec2(a, b));
    }
    return v;
}

/// 將兩個一維向量的每個元素進行比較，取最小值。
pub fn minimum_vec1<T: PartialOrd + Copy>(v1: &[T], v2: &[T]) -> Vec<T> {
    let mut v = vec![];
    for (a, b) in v1.iter().zip(v2.iter()) {
        v.push(if a < b { *a } else { *b })
    }
    return v;
}

/// 將兩個二維向量的每個元素進行比較，取最小值。
pub fn minimum_vec2<T: PartialOrd + Copy>(v1: &[Vec<T>], v2: &[Vec<T>]) -> Vec<Vec<T>> {
    let mut v = vec![];
    for (a, b) in v1.iter().zip(v2.iter()) {
        v.push(minimum_vec1(a, b));
    }
    return v;
}

/// 將兩個三維向量的每個元素進行比較，取最小值。
pub fn minimum_vec3<T: PartialOrd + Copy>(
    v1: &[Vec<Vec<T>>],
    v2: &[Vec<Vec<T>>],
) -> Vec<Vec<Vec<T>>> {
    let mut v = vec![];
    for (a, b) in v1.iter().zip(v2.iter()) {
        v.push(minimum_vec2(a, b));
    }
    return v;
}
