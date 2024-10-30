use candle_core::{DType, Device, Result, Tensor};

#[test]
fn broadcast_matmul() -> Result<()> {
    let a = Tensor::arange(0u32, 24, &Device::Cpu)?
        .reshape(&[2, 3, 4])?
        .to_dtype(DType::F32)?; // 2x3x4
    let b = Tensor::arange(0u32, 8, &Device::Cpu)?
        .reshape(&[4, 2])?
        .to_dtype(DType::F32)?; // 4x2

    let c = a.broadcast_matmul(&b)?; // 2x3x4 * 4x2 = 2x3x2
    let c = c.to_vec3::<f32>()?;

    let a1 = Tensor::arange(0u32, 12, &Device::Cpu)?
        .reshape(&[1, 3, 4])?
        .to_dtype(DType::F32)?;

    let c1 = a1.broadcast_matmul(&b)?;
    let c1 = c1.to_vec3::<f32>()?;

    let a2 = Tensor::arange(12u32, 24, &Device::Cpu)?
        .reshape(&[1, 3, 4])?
        .to_dtype(DType::F32)?;
    let c2 = a2.broadcast_matmul(&b)?;
    let c2 = c2.to_vec3::<f32>()?;

    assert_eq!(c[0], c1[0]);
    assert_eq!(c[1], c2[0]);

    // let c3 = Tensor::stack(&[&c1, &c2], 1)?;
    // println!("{:?}", c3.shape());
    // let c3 = c3.squeeze(0)?;
    // println!("{:?}", c3.shape());
    // println!("{:?}", c3.to_vec3::<f32>()?);

    let a = Tensor::arange(0u32, 24, &Device::Cpu)?
        .reshape(&[2, 1, 3, 4])?
        .to_dtype(DType::F32)?; // 2x1x3x4

    let b = Tensor::arange(0u32, 16, &Device::Cpu)?
        .reshape(&[2, 4, 2])?
        .to_dtype(DType::F32)?; // 2x4x2

    let c = a.broadcast_matmul(&b)?; // 2x1x3x4 * 2x4x2 = 2x2x3x2
    assert_eq!(c.dims(), &[2, 2, 3, 2]);

    Ok(())
}
