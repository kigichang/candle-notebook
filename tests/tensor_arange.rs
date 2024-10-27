use candle_core::{Device, Result, Tensor};

#[test]
fn arange() -> Result<()> {
    // vector only

    let t = Tensor::arange(0.0f32, 4.0, &Device::Cpu)?;
    assert_eq!(t.to_vec1::<f32>()?, vec![0.0, 1.0, 2.0, 3.0]);

    let t = Tensor::arange_step(0u32, 12, 3, &Device::Cpu)?;
    assert_eq!(t.to_vec1::<u32>()?, vec![0, 3, 6, 9]);

    let t = Tensor::arange_step(0.0f32, 3.0, 0.5, &Device::Cpu)?;
    assert_eq!(t.to_vec1::<f32>()?, vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5]);

    // be careful of floating point precision
    let t = Tensor::arange_step(0.0f32, 3.0, 0.3, &Device::Cpu)?;
    println!("{:?}", t.to_vec1::<f32>()?);

    Ok(())
}
