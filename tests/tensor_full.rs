use candle_core::{DType, Device, Result, Tensor};

/// 建立全為指定值的張量
#[test]
fn full() -> Result<()> {
    // 純量
    let t = Tensor::full(1.0f32, (), &Device::Cpu)?;
    assert_eq!(t.to_scalar::<f32>()?, 1.0);

    // 向量
    let t = Tensor::full(1.0f32, 2, &Device::Cpu)?;
    assert_eq!(t.to_vec1::<f32>()?, &[1.; 2]);
    let z = Tensor::ones(2, DType::F32, &Device::Cpu)?;
    assert_eq!(t.shape(), z.shape());
    assert_eq!(t.to_vec1::<f32>()?, z.to_vec1::<f32>()?);

    // 2D 張量 (2x2)
    let t = Tensor::full(0.0f32, (2, 2), &Device::Cpu)?;
    assert_eq!(t.to_vec2::<f32>()?, &[[0.; 2], [0.; 2]]);
    let z = Tensor::zeros((2, 2), DType::F32, &Device::Cpu)?;
    assert_eq!(t.shape(), z.shape());
    assert_eq!(t.to_vec2::<f32>()?, z.to_vec2::<f32>()?);

    // 3D 張量 (2x2x2)
    let t = Tensor::full(-1.0f32, (2, 2, 2), &Device::Cpu)?;
    assert_eq!(
        t.to_vec3::<f32>()?,
        &[[[-1.; 2], [-1.; 2]], [[-1.; 2], [-1.; 2]]]
    );

    Ok(())
}
