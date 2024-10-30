use candle_core::{DType, Device, Result, Tensor};

/// 建立全為 1 的張量
#[test]
fn ones() -> Result<()> {
    // 純量
    let t = Tensor::ones((), DType::F32, &Device::Cpu)?;
    assert_eq!(t.to_scalar::<f32>()?, 1.0);

    // 向量
    let t = Tensor::ones(3, DType::F32, &Device::Cpu)?;
    assert_eq!(t.to_vec1::<f32>()?, vec![1., 1., 1.]);

    // 2D 張量 (2x2)
    let t = Tensor::ones((2, 2), DType::F32, &Device::Cpu)?;
    assert_eq!(t.to_vec2::<f32>()?, vec![vec![1., 1.], vec![1., 1.]]);

    // 3D 張量 (2x2x2)
    let t = Tensor::ones((2, 2, 2), DType::F32, &Device::Cpu)?;
    assert_eq!(
        t.to_vec3::<f32>()?,
        vec![
            vec![vec![1., 1.], vec![1., 1.]],
            vec![vec![1., 1.], vec![1., 1.]]
        ]
    );

    // 已知純量
    let t = Tensor::new(0.0f32, &Device::Cpu)?;
    let z = t.ones_like()?;
    assert_eq!(z.to_scalar::<f32>()?, 1.);
    assert_eq!(z.shape(), t.shape());

    // 已知向量
    let t = Tensor::new(&[1.0f32, 2., 3., 4.], &Device::Cpu)?;
    let z = t.ones_like()?;
    assert_eq!(z.to_vec1::<f32>()?, vec![1., 1., 1., 1.]);
    assert_eq!(z.shape(), t.shape());

    // 已知 2D 張量 (2x2)
    let t = Tensor::new(&[[1.0f32, 2.], [3., 4.]], &Device::Cpu)?;
    let z = t.ones_like()?;
    assert_eq!(z.to_vec2::<f32>()?, vec![vec![1., 1.], vec![1., 1.]]);
    assert_eq!(z.shape(), t.shape());

    // 已知 3D 張量 (2x2x2)
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
