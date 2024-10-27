use candle_core::{DType, Device, Result, Tensor};

#[test]
fn randn() -> Result<()> {
    // 常態分佈 N(0, 1)

    // scalar
    let t = Tensor::randn(0.0f32, 1.0, (), &Device::Cpu)?;
    println!("{:?}", t.to_scalar::<f32>()?);

    // vector
    let t = Tensor::randn(0.0f32, 1.0, 2, &Device::Cpu)?;
    println!("{:?}", t.to_vec1::<f32>()?);

    // 2x2 矩陣
    let t = Tensor::randn(0.0f32, 1.0, (2, 2), &Device::Cpu)?;
    println!("{:?}", t.to_vec2::<f32>()?);

    // 2x2x2 矩陣
    let t = Tensor::randn(0.0f32, 1.0, (2, 2, 2), &Device::Cpu)?;
    println!("{:?}", t.to_vec3::<f32>()?);

    let t = Tensor::ones((2, 3, 4), DType::F32, &Device::Cpu)?;
    let z = t.randn_like(0.0f64, 1.0)?;
    println!("{:?}", z.to_vec3::<f32>()?);
    assert_eq!(z.shape(), t.shape());
    assert_eq!(z.dtype(), t.dtype());
    Ok(())
}
