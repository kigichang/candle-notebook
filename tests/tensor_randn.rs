use candle_core::{DType, Device, Result, Tensor};

/// 產生常態分佈的隨機數張量
#[test]
fn randn() -> Result<()> {
    // 常態分佈 N(0, 1)

    // 純量
    let t = Tensor::randn(0.0f32, 1.0, (), &Device::Cpu)?;
    assert_eq!(t.shape(), &().into());

    // 向量
    let t = Tensor::randn(0.0f32, 1.0, 2, &Device::Cpu)?;
    assert_eq!(t.dims(), &[2]);

    // 2D 張量 (2x2)
    let t = Tensor::randn(0.0f32, 1.0, (2, 2), &Device::Cpu)?;
    assert_eq!(t.dims(), &[2, 2]);

    // 3D 張量 (2x2x2)
    let t = Tensor::randn(0.0f32, 1.0, (2, 2, 2), &Device::Cpu)?;
    assert_eq!(t.dims(), &[2, 2, 2]);

    // 由己知張量產生隨機數張量
    let t = Tensor::ones((2, 3, 4), DType::F32, &Device::Cpu)?;
    let z = t.randn_like(0.0f64, 1.0)?;
    assert_eq!(z.dims(), &[2, 3, 4]);
    assert_eq!(z.shape(), t.shape());
    assert_eq!(z.dtype(), t.dtype());
    Ok(())
}
