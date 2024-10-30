use candle_core::{DType, Device, Result, Tensor};

/// 產生隨機數的張量
#[test]
fn rand() -> Result<()> {
    // 純量
    let t = Tensor::rand(0.0f32, 1.0, (), &Device::Cpu)?;
    assert_eq!(t.rank(), 0);
    assert_eq!(t.shape(), &().into());

    // 向量
    let t = Tensor::rand(0.0f32, 1.0, 2, &Device::Cpu)?;
    assert_eq!(t.rank(), 1);
    assert_eq!(t.dims(), &[2]);

    // 2D 張量 (2x2)
    let t = Tensor::rand(0.0f32, 1.0, (2, 2), &Device::Cpu)?;
    assert_eq!(t.rank(), 2);
    assert_eq!(t.dims(), &[2, 2]);

    // 3D 張量 (2x2x2)
    let t = Tensor::rand(0.0f32, 1.0, (2, 2, 2), &Device::Cpu)?;
    assert_eq!(t.rank(), 3);
    assert_eq!(t.dims(), &[2, 2, 2]);

    // 由己知張量產生隨機數張量
    let t = Tensor::ones((2, 2), DType::F32, &Device::Cpu)?;
    let z = t.rand_like(0.0f64, 1.0)?;
    assert_eq!(z.shape(), t.shape());
    assert_eq!(z.dtype(), t.dtype());

    Ok(())
}
