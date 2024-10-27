use candle_core::{DType, Device, Result, Tensor};

#[test]
fn zeros() -> Result<()> {
    // scalar
    let t = Tensor::zeros((), DType::F32, &Device::Cpu)?;
    assert_eq!(t.to_scalar::<f32>()?, 0.0);

    // vector
    let t = Tensor::zeros(3, DType::F32, &Device::Cpu)?;
    assert_eq!(t.to_vec1::<f32>()?, vec![0.0, 0.0, 0.0]);

    // 2x2 矩陣
    let t = Tensor::zeros((2, 2), DType::F32, &Device::Cpu)?;
    assert_eq!(t.to_vec2::<f32>()?, vec![vec![0.0, 0.0], vec![0.0, 0.0]]);

    // 2x2x2 矩陣
    let t = Tensor::zeros((2, 2, 2), DType::F32, &Device::Cpu)?;
    assert_eq!(
        t.to_vec3::<f32>()?,
        vec![
            vec![vec![0.0, 0.0], vec![0.0, 0.0]],
            vec![vec![0.0, 0.0], vec![0.0, 0.0]]
        ]
    );

    // scalar
    let t = Tensor::new(1.0f32, &Device::Cpu)?;
    let z = t.zeros_like()?;
    assert_eq!(z.to_scalar::<f32>()?, 0.0);
    assert_eq!(z.shape(), t.shape());

    // vector
    let t = Tensor::new(&[1.0f32, 2., 3., 4.], &Device::Cpu)?;
    let z = t.zeros_like()?;
    assert_eq!(z.to_vec1::<f32>()?, vec![0.0, 0.0, 0.0, 0.0]);
    assert_eq!(z.shape(), t.shape());

    // 2x2 矩陣
    let t = Tensor::new(&[[1.0f32, 2.], [3., 4.]], &Device::Cpu)?;
    let z = t.zeros_like()?;
    assert_eq!(z.to_vec2::<f32>()?, vec![vec![0.0, 0.0], vec![0.0, 0.0]]);
    assert_eq!(z.shape(), t.shape());

    // 2x2x2 矩陣
    let t = Tensor::new(
        &[[[1.0f32, 2.], [3., 4.]], [[5., 6.], [7., 8.]]],
        &Device::Cpu,
    )?;
    let z = t.zeros_like()?;
    assert_eq!(
        z.to_vec3::<f32>()?,
        vec![
            vec![vec![0.0, 0.0], vec![0.0, 0.0]],
            vec![vec![0.0, 0.0], vec![0.0, 0.0]]
        ]
    );
    assert_eq!(z.shape(), t.shape());

    Ok(())
}
