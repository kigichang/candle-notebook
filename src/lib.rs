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

macro_rules! cmp_op  {
    ($fn: ident, $op: tt, $l: tt, $r: tt, $t: expr, $f: expr, $result: ty) => {
        pub (crate) fn $fn<T: PartialEq + PartialOrd + Copy>($l: &T, $r: &T) -> $result {
            if $l $op $r { $t } else { $f }
        }
    };
}

macro_rules! vec_n_op {
    ($fn: ident, $inner_fn: ident, $in: ty, $result: ty) => {
        pub fn $fn<T: PartialOrd + PartialEq + Copy>(v1: &[$in], v2: &[$in]) -> Vec<$result> {
            let mut v = vec![];
            for (a, b) in v1.iter().zip(v2.iter()) {
                v.push($inner_fn(a, b));
            }
            return v;
        }
    };
}

cmp_op!(max, >, a, b, *a, *b, T);

// 將兩個一維向量的每個元素進行比較，取最大值。
vec_n_op!(maximum_vec1, max, T, T);

// 將兩個二維向量的每個元素進行比較，取最大值。
vec_n_op!(maximum_vec2, maximum_vec1, Vec<T>, Vec<T>);

// 將兩個三維向量的每個元素進行比較，取最大值。
vec_n_op!(maximum_vec3, maximum_vec2, Vec<Vec<T>>, Vec<Vec<T>>);

cmp_op!(min, <, a, b, *a, *b, T);

// 將兩個一維向量的每個元素進行比較，取最小值。
vec_n_op!(minimum_vec1, min, T, T);

// 將兩個二維向量的每個元素進行比較，取最小值。
vec_n_op!(minimum_vec2, minimum_vec1, Vec<T>, Vec<T>);

// 將兩個三維向量的每個元素進行比較，取最小值。
vec_n_op!(minimum_vec3, minimum_vec2, Vec<Vec<T>>, Vec<Vec<T>>);

cmp_op!(eq, ==, a, b, 1, 0, u8);

// 用於比較兩個一維向量的每個元素是否相等，相等為 1，不相等為 0。
vec_n_op!(eq_vec1, eq, T, u8);

// 用於比較兩個二維向量的每個元素是否相等，相等為 1，不相等為 0。
vec_n_op!(eq_vec2, eq_vec1, Vec<T>, Vec<u8>);

// 用於比較兩個三維向量的每個元素是否相等，相等為 1，不相等為 0
vec_n_op!(eq_vec3, eq_vec2, Vec<Vec<T>>, Vec<Vec<u8>>);

cmp_op!(lt, <, a, b, 1, 0, u8);
vec_n_op!(lt_vec1, lt, T, u8);
vec_n_op!(lt_vec2, lt_vec1, Vec<T>, Vec<u8>);
vec_n_op!(lt_vec3, lt_vec2, Vec<Vec<T>>, Vec<Vec<u8>>);

cmp_op!(gt, >, a, b, 1, 0, u8);
vec_n_op!(gt_vec1, gt, T, u8);
vec_n_op!(gt_vec2, gt_vec1, Vec<T>, Vec<u8>);
vec_n_op!(gt_vec3, gt_vec2, Vec<Vec<T>>, Vec<Vec<u8>>);

cmp_op!(le, <=, a, b, 1, 0, u8);
vec_n_op!(le_vec1, le, T, u8);
vec_n_op!(le_vec2, le_vec1, Vec<T>, Vec<u8>);
vec_n_op!(le_vec3, le_vec2, Vec<Vec<T>>, Vec<Vec<u8>>);

cmp_op!(ge, >=, a, b, 1, 0, u8);
vec_n_op!(ge_vec1, ge, T, u8);
vec_n_op!(ge_vec2, ge_vec1, Vec<T>, Vec<u8>);
vec_n_op!(ge_vec3, ge_vec2, Vec<Vec<T>>, Vec<Vec<u8>>);
