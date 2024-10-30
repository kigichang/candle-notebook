use candle_core::{DType, Device, Result, Tensor};

/// 建立全為 0 的張量
#[test]
fn zeros() -> Result<()> {
    // 純量
    let t = Tensor::zeros((), DType::F32, &Device::Cpu)?;
    assert_eq!(t.to_scalar::<f32>()?, 0.0);

    // 向量
    let t = Tensor::zeros(3, DType::F32, &Device::Cpu)?;
    assert_eq!(t.to_vec1::<f32>()?, vec![0.0, 0.0, 0.0]);

    // 2D 張量 (2x2)
    let t = Tensor::zeros((2, 2), DType::F32, &Device::Cpu)?;
    assert_eq!(t.to_vec2::<f32>()?, vec![vec![0.0, 0.0], vec![0.0, 0.0]]);

    // 3D 張量 (2x2x2)
    let t = Tensor::zeros((2, 2, 2), DType::F32, &Device::Cpu)?;
    assert_eq!(
        t.to_vec3::<f32>()?,
        vec![
            vec![vec![0.0, 0.0], vec![0.0, 0.0]],
            vec![vec![0.0, 0.0], vec![0.0, 0.0]]
        ]
    );

    // 已知純量
    let t = Tensor::new(1.0f32, &Device::Cpu)?;
    let z = t.zeros_like()?;
    assert_eq!(z.to_scalar::<f32>()?, 0.0);
    assert_eq!(z.dims0()?, ());
    assert_eq!(z.shape(), t.shape());

    // 已知向量
    let t = Tensor::new(&[1.0f32, 2., 3., 4.], &Device::Cpu)?;
    let z = t.zeros_like()?;
    assert_eq!(z.to_vec1::<f32>()?, vec![0.0, 0.0, 0.0, 0.0]);
    assert_eq!(z.dims1()?, 4);
    assert_eq!(z.shape(), t.shape());

    // 已知 2D 張量 (2x2)
    let t = Tensor::new(&[[1.0f32, 2.], [3., 4.]], &Device::Cpu)?;
    let z = t.zeros_like()?;
    assert_eq!(z.to_vec2::<f32>()?, vec![vec![0.0, 0.0], vec![0.0, 0.0]]);
    assert_eq!(z.dims2()?, (2, 2));
    assert_eq!(z.shape(), t.shape());

    // 已知 3D 張量 (2x2x2)
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
    assert_eq!(z.dims3()?, (2, 2, 2));
    assert_eq!(z.shape(), t.shape());

    Ok(())
}
