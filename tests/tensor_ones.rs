use candle_core::{DType, Device, Result, Tensor};

#[test]
fn ones() -> Result<()> {
    // scalar
    let t = Tensor::ones((), DType::F32, &Device::Cpu)?;
    assert_eq!(t.to_scalar::<f32>()?, 1.0);

    // vector
    let t = Tensor::ones(3, DType::F32, &Device::Cpu)?;
    assert_eq!(t.to_vec1::<f32>()?, vec![1., 1., 1.]);

    // 2x2 矩陣
    let t = Tensor::ones((2, 2), DType::F32, &Device::Cpu)?;
    assert_eq!(t.to_vec2::<f32>()?, vec![vec![1., 1.], vec![1., 1.]]);

    // 2x2x2 矩陣
    let t = Tensor::ones((2, 2, 2), DType::F32, &Device::Cpu)?;
    assert_eq!(
        t.to_vec3::<f32>()?,
        vec![
            vec![vec![1., 1.], vec![1., 1.]],
            vec![vec![1., 1.], vec![1., 1.]]
        ]
    );

    // scalar
    let t = Tensor::new(0.0f32, &Device::Cpu)?;
    let z = t.ones_like()?;
    assert_eq!(z.to_scalar::<f32>()?, 1.);
    assert_eq!(z.shape(), t.shape());

    // vector
    let t = Tensor::new(&[1.0f32, 2., 3., 4.], &Device::Cpu)?;
    let z = t.ones_like()?;
    assert_eq!(z.to_vec1::<f32>()?, vec![1., 1., 1., 1.]);
    assert_eq!(z.shape(), t.shape());

    // 2x2 矩陣
    let t = Tensor::new(&[[1.0f32, 2.], [3., 4.]], &Device::Cpu)?;
    let z = t.ones_like()?;
    assert_eq!(z.to_vec2::<f32>()?, vec![vec![1., 1.], vec![1., 1.]]);
    assert_eq!(z.shape(), t.shape());

    // 2x2x2 矩陣
    let t = Tensor::new(
        &[[[1.0f32, 2.], [3., 4.]], [[5., 6.], [7., 8.]]],
        &Device::Cpu,
    )?;
    let z = t.ones_like()?;
    assert_eq!(
        z.to_vec3::<f32>()?,
        vec![
            vec![vec![1., 1.], vec![1., 1.]],
            vec![vec![1., 1.], vec![1., 1.]]
        ]
    );
    assert_eq!(z.shape(), t.shape());

    Ok(())
}
