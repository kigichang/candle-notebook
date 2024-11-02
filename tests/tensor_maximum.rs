use candle_core::{DType, Device, Result, Shape, Tensor, WithDType};

/// 使用 Tensor::maximum 比較張量，比較兩個張量的每個元素，取最大值。
/// 由源碼來看：
/// 如果右運算元是純量(數值)，資料型別可以與左運算元張量不同。
/// 如果右運算元是張量，則兩個張量維度與型別必須相同。

fn maximum_vec1<T: PartialOrd + Copy>(v1: &[T], v2: &[T]) -> Vec<T> {
    let mut v = vec![];
    for (a, b) in v1.iter().zip(v2.iter()) {
        v.push(if a > b { *a } else { *b })
    }
    return v;
}

fn maximum_vec2<T: PartialOrd + Copy>(v1: &[Vec<T>], v2: &[Vec<T>]) -> Vec<Vec<T>> {
    let mut v = vec![];
    for (a, b) in v1.iter().zip(v2.iter()) {
        v.push(maximum_vec1(a, b));
    }
    return v;
}

fn maximum_vec3<T: PartialOrd + Copy>(v1: &[Vec<Vec<T>>], v2: &[Vec<Vec<T>>]) -> Vec<Vec<Vec<T>>> {
    let mut v = vec![];
    for (a, b) in v1.iter().zip(v2.iter()) {
        v.push(maximum_vec2(a, b));
    }
    return v;
}

fn scalar_to_tensor<T: WithDType, S: Into<Shape>>(
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

#[test]
fn maximum_vector() -> Result<()> {
    let t1 = Tensor::rand(-1.0f32, 1.0, 3, &Device::Cpu)?;
    let v1 = t1.to_vec1::<f32>()?;
    let t2 = Tensor::rand(-1.0f32, 1.0, 3, &Device::Cpu)?;
    let v2 = t2.to_vec1::<f32>()?;

    let max = t1.maximum(&t2)?;
    assert_eq!(max.to_vec1::<f32>()?, maximum_vec1(&v1, &v2));

    // 與數值比較
    let max = t1.maximum(0u32)?;
    assert_eq!(max.to_vec1::<f32>()?, maximum_vec1(&v1, &[0.0f32; 3]));
    let zero = scalar_to_tensor(0u32, 3, DType::F32, t1.device())?;
    assert_eq!(zero.dims(), vec![3]);
    assert_eq!(zero.dtype(), DType::F32);
    let max_with_zero = t1.maximum(&zero)?;
    assert_eq!(max_with_zero.to_vec1::<f32>()?, max.to_vec1::<f32>()?);

    // 與純量張量比較
    let t2 = Tensor::rand(-1.0f32, 1.0f32, (), &Device::Cpu)?;
    let max = t1.maximum(&t2);
    assert!(max.is_err());

    // 與不同長度向量比較
    let t2 = Tensor::rand(-1.0f32, 1.0f32, 2, &Device::Cpu)?;
    let max = t1.maximum(&t2);
    assert!(max.is_err());

    // 與不同型別張量比較
    let t2 = Tensor::rand(-1.0f64, 1.0f64, 3, &Device::Cpu)?;
    let max = t1.maximum(&t2);
    assert!(max.is_err());

    Ok(())
}

#[test]
fn maximum() -> Result<()> {
    let t1 = Tensor::rand(-1.0f32, 1.0, (2, 3, 4), &Device::Cpu)?;
    let v1 = t1.to_vec3::<f32>()?;
    let t2 = Tensor::rand(-1.0f32, 1.0, (2, 3, 4), &Device::Cpu)?;
    let v2 = t2.to_vec3::<f32>()?;

    let max = t1.maximum(&t2)?;
    assert_eq!(max.to_vec3::<f32>()?, maximum_vec3(&v1, &v2));

    let max = t1.maximum(0u32)?;
    let zero = scalar_to_tensor(0u32, (2, 3, 4), DType::F32, t1.device())?;
    let max_with_zero = t1.maximum(&zero)?;
    assert_eq!(max_with_zero.to_vec3::<f32>()?, max.to_vec3::<f32>()?);

    Ok(())
}
