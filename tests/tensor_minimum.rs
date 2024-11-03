use candle_core::{DType, Device, Result, Tensor};
use candle_notebook::*;
use half::f16;

/// 使用 Tensor::minimum 比較張量，比較兩個張量的每個元素，取最大值。
/// 由源碼來看：
/// 如果右運算元是純量(數值)，資料型別可以與左運算元張量不同。
/// 如果右運算元是張量，則兩個張量形狀與型別必須相同。

#[test]
fn minimum_vector() -> Result<()> {
    let t1 = Tensor::rand(-1.0f32, 1.0, 3, &Device::Cpu)?;
    let v1 = t1.to_vec1::<f32>()?;
    let t2 = Tensor::rand(-1.0f32, 1.0, 3, &Device::Cpu)?;
    let v2 = t2.to_vec1::<f32>()?;

    assert_eq!(t1.shape(), t2.shape());
    let min = t1.minimum(&t2)?;
    assert_eq!(min.to_vec1::<f32>()?, minimum_vec1(&v1, &v2));

    // 與數值比較
    let min = t1.minimum(0u32)?;
    assert_eq!(min.to_vec1::<f32>()?, minimum_vec1(&v1, &[0.0f32; 3]));
    let zero = scalar_to_tensor(f16::from_f32(0f32), 3, DType::F32, t1.device())?;
    assert_eq!(zero.dims(), vec![3]);
    assert_eq!(zero.dtype(), DType::F32);
    let min_with_zero = t1.minimum(&zero)?;
    assert_eq!(min_with_zero.to_vec1::<f32>()?, min.to_vec1::<f32>()?);

    // 與純量張量比較
    let t2 = Tensor::rand(-1.0f32, 1.0f32, (), &Device::Cpu)?;
    let min = t1.minimum(&t2);
    assert!(min.is_err());

    // 與不同長度向量比較
    let t2 = Tensor::rand(-1.0f32, 1.0f32, 2, &Device::Cpu)?;
    let min = t1.minimum(&t2);
    assert!(min.is_err());

    // 與不同型別張量比較
    let t2 = Tensor::rand(-1.0f64, 1.0f64, 3, &Device::Cpu)?;
    let min = t1.minimum(&t2);
    assert!(min.is_err());

    Ok(())
}

#[test]
fn minimum() -> Result<()> {
    let t1 = Tensor::rand(-1.0f32, 1.0, (2, 3, 4), &Device::Cpu)?;
    let v1 = t1.to_vec3::<f32>()?;
    let t2 = Tensor::rand(-1.0f32, 1.0, (2, 3, 4), &Device::Cpu)?;
    let v2 = t2.to_vec3::<f32>()?;

    let min = t1.minimum(&t2)?;
    assert_eq!(min.to_vec3::<f32>()?, minimum_vec3(&v1, &v2));

    let min = t1.minimum(0u32)?;
    let zero = scalar_to_tensor(0u32, (2, 3, 4), DType::F32, t1.device())?;
    let min_with_zero = t1.minimum(&zero)?;
    assert_eq!(min_with_zero.to_vec3::<f32>()?, min.to_vec3::<f32>()?);

    Ok(())
}
