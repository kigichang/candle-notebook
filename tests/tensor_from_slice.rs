use candle_core::{Device, Result, Tensor};

/// 使用 Tensor::from_slice 建立張量
#[test]
fn from_slice() -> Result<()> {
    // 純量張量
    let t = Tensor::from_slice(&[10.0f32], (), &Device::Cpu)?;
    assert_eq!(t.to_scalar::<f32>()?, 10.0);

    // 向量
    let t = Tensor::from_slice(&[1.0f32, 2., 3., 4.], 4, &Device::Cpu)?;
    assert_eq!(t.to_vec1::<f32>()?, vec![1.0, 2.0, 3.0, 4.0]);
    let v = Tensor::from_slice(&vec![1.0f32, 2., 3., 4.], 4, &Device::Cpu)?;
    assert_eq!(v.to_vec1::<f32>()?, t.to_vec1::<f32>()?);

    // 2D 張量 (2x2)
    let t = Tensor::from_slice(&[1.0f32, 2., 3., 4.], (2, 2), &Device::Cpu)?;
    assert_eq!(t.to_vec2::<f32>()?, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    let v = Tensor::from_slice(&vec![1.0f32, 2., 3., 4.], (2, 2), &Device::Cpu)?;
    assert_eq!(v.to_vec2::<f32>()?, t.to_vec2::<f32>()?);

    // 3D 張量 (2x2x2)
    let t = Tensor::from_slice(
        &[1.0f32, 2., 3., 4., 5., 6., 7., 8.],
        (2, 2, 2),
        &Device::Cpu,
    )?;
    assert_eq!(
        t.to_vec3::<f32>()?,
        vec![
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec![vec![5.0, 6.0], vec![7.0, 8.0]]
        ]
    );
    let v = Tensor::from_slice(
        &vec![1.0f32, 2., 3., 4., 5., 6., 7., 8.],
        (2, 2, 2),
        &Device::Cpu,
    )?;
    assert_eq!(v.to_vec3::<f32>()?, t.to_vec3::<f32>()?);

    Ok(())
}
