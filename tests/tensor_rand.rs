use candle_core::{DType, Device, Result, Tensor};

#[test]
fn rand() -> Result<()> {
    // open range `[lo, up)` (excluding `up`)

    // scalar
    let t = Tensor::rand(0.0f32, 1.0, (), &Device::Cpu)?;
    println!("{:?}", t.to_scalar::<f32>()?);
    assert_eq!(t.rank(), 0);
    assert_eq!(t.shape(), &().into());

    // vector
    let t = Tensor::rand(0.0f32, 1.0, 2, &Device::Cpu)?;
    println!("{:?}", t.to_vec1::<f32>()?);
    assert_eq!(t.rank(), 1);

    // 2x2 矩陣
    let t = Tensor::rand(0.0f32, 1.0, (2, 2), &Device::Cpu)?;
    println!("{:?}", t.to_vec2::<f32>()?);
    assert_eq!(t.rank(), 2);

    // 2x2x2 矩陣
    let t = Tensor::rand(0.0f32, 1.0, (2, 2, 2), &Device::Cpu)?;
    println!("{:?}", t.to_vec3::<f32>()?);
    assert_eq!(t.rank(), 3);

    // rand_like
    let t = Tensor::ones((2, 2), DType::F32, &Device::Cpu)?;
    let z = t.rand_like(0.0f64, 1.0)?;
    println!("{:?}", z.to_vec2::<f32>()?);
    assert_eq!(z.shape(), t.shape());
    assert_eq!(z.dtype(), t.dtype());

    Ok(())
}
