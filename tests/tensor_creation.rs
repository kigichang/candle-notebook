use candle_core::{DType, Device, Result, Tensor};

#[test]
fn to_dtype() -> Result<()> {
    let t = Tensor::new(1.0f32, &Device::Cpu)?;
    let z = t.to_dtype(DType::F64)?;
    println!("{:?}", z.to_scalar::<f64>()?);

    // BF16 test
    let t = Tensor::new(1.0f32, &Device::Cpu)?;
    let z = t.to_dtype(DType::BF16)?;
    println!("{:?}", z.to_scalar::<half::bf16>()?);

    // f16 test
    let t = Tensor::new(1.0f32, &Device::Cpu)?;
    let z = t.to_dtype(DType::F16)?;
    println!("{:?}", z.to_scalar::<half::f16>()?);

    Ok(())
}

#[test]
fn reshape() -> Result<()> {
    let t = Tensor::from_iter(0..24u32, &Device::Cpu)?
        .to_dtype(DType::F32)?
        .reshape((2, 3, 4))?;
    assert_eq!(t.dims(), &[2, 3, 4]);
    Ok(())
}
