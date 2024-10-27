use candle_core::{DType, Device, Result, Tensor};

#[test]
fn full() -> Result<()> {
    // scalar
    let t = Tensor::full(1.0f32, (), &Device::Cpu)?;
    assert_eq!(t.to_scalar::<f32>()?, 1.0);

    // vector
    let t = Tensor::full(1.0f32, 2, &Device::Cpu)?;
    assert_eq!(t.to_vec1::<f32>()?, [1., 1.]);
    let z = Tensor::ones(2, DType::F32, &Device::Cpu)?;
    assert_eq!(t.shape(), z.shape());
    assert_eq!(t.to_vec1::<f32>()?, z.to_vec1::<f32>()?);

    // 2x2 矩陣
    let t = Tensor::full(0.0f32, (2, 2), &Device::Cpu)?;
    assert_eq!(t.to_vec2::<f32>()?, vec![vec![0., 0.], vec![0., 0.]]);
    let z = Tensor::zeros((2, 2), DType::F32, &Device::Cpu)?;
    assert_eq!(t.shape(), z.shape());
    assert_eq!(t.to_vec2::<f32>()?, z.to_vec2::<f32>()?);

    // 2x2x2 矩陣
    let t = Tensor::full(-1.0f32, (2, 2, 2), &Device::Cpu)?;
    assert_eq!(
        t.to_vec3::<f32>()?,
        vec![
            vec![vec![-1., -1.], vec![-1., -1.]],
            vec![vec![-1., -1.], vec![-1., -1.]]
        ]
    );

    Ok(())
}
