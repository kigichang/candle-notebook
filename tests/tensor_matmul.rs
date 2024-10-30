use candle_core::{DType, Device, Result, Tensor};

#[test]
fn matmul() -> Result<()> {
    // 2D 張量 (2x3)
    let a = Tensor::from_iter(0u32..6, &Device::Cpu)?
        .reshape((2, 3))?
        .to_dtype(DType::F32)?;

    // 2D 張量 (3x2)
    let b = Tensor::from_iter(0u32..6, &Device::Cpu)?
        .reshape((3, 2))?
        .to_dtype(DType::F32)?;

    let c = a.matmul(&b)?; // 2x3 * 3x2 = 2x2
    assert_eq!(
        c.to_vec2::<f32>()?,
        vec![vec![10.0, 13.0], vec![28.0, 40.0]]
    );

    // 長度為 2 的向量
    let b = Tensor::new(&[1.0f32; 2], &Device::Cpu)?;
    let c = b.matmul(&a); // shape mismatch
    assert!(c.is_err());

    // 2D 張量 (1x2)
    let b = Tensor::new(&[[1.0f32; 2]], &Device::Cpu)?;
    let c = b.matmul(&a)?; // 1x2 * 2x3 = 1x3
    assert_eq!(c.to_vec2::<f32>()?, vec![vec![3.0, 5.0, 7.0]]);
    Ok(())
}
