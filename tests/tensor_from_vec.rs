use candle_core::{Device, Result, Tensor};

/// 使用 Tensor::from_vec 建立張量
#[test]
fn from_vec() -> Result<()> {
    // 純量張量
    let t = Tensor::from_vec(vec![1.0f32], (), &Device::Cpu)?;
    assert_eq!(t.dims0()?, ());
    assert_eq!(t.to_scalar::<f32>()?, 1.0);

    // 向量
    let t = Tensor::from_vec(vec![1.0f32, 2., 3., 4.], 4, &Device::Cpu)?;
    assert_eq!(t.dims1()?, 4);
    assert_eq!(t.to_vec1::<f32>()?, vec![1.0, 2.0, 3.0, 4.0]);

    // &[N] Shape
    let z = Tensor::from_vec(vec![1.0f32, 2., 3., 4.], &[4], &Device::Cpu)?;
    assert_eq!(z.dims(), t.dims());
    assert_eq!(z.to_vec1::<f32>()?, t.to_vec1::<f32>()?);

    // (N,) Shape
    let z = Tensor::from_vec(vec![1.0f32, 2., 3., 4.], (4,), &Device::Cpu)?;
    assert_eq!(z.dims(), t.dims());
    assert_eq!(z.to_vec1::<f32>()?, t.to_vec1::<f32>()?);

    // vec![N] Shape
    let z = Tensor::from_vec(vec![1.0f32, 2., 3., 4.], vec![4], &Device::Cpu)?;
    assert_eq!(z.dims(), t.dims());
    assert_eq!(z.to_vec1::<f32>()?, t.to_vec1::<f32>()?);

    // 2D 張量 (2x2)
    let t = Tensor::from_vec(vec![1.0f32, 2., 3., 4.], (2, 2), &Device::Cpu)?;
    assert_eq!(t.dims2()?, (2, 2));
    assert_eq!(t.to_vec2::<f32>()?, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);

    // &[N; S] Shape
    let z = Tensor::from_vec(vec![1.0f32, 2., 3., 4.], &[2; 2], &Device::Cpu)?;
    assert_eq!(z.dims(), t.dims());
    assert_eq!(z.to_vec2::<f32>()?, t.to_vec2::<f32>()?);

    // vec![N1, N2, ...] Shape
    let z = Tensor::from_vec(vec![1.0f32, 2., 3., 4.], vec![2, 2], &Device::Cpu)?;
    assert_eq!(z.dims(), t.dims());
    assert_eq!(z.to_vec2::<f32>()?, t.to_vec2::<f32>()?);

    // 3D 張量 (2x2x2)
    let t = Tensor::from_vec(
        vec![1.0f32, 2., 3., 4., 5., 6., 7., 8.],
        (2, 2, 2),
        &Device::Cpu,
    )?;
    assert_eq!(t.dims3()?, (2, 2, 2));
    let z = Tensor::new(
        vec![
            vec![vec![1.0f32, 2.0], vec![3.0, 4.0]],
            vec![vec![5.0, 6.0], vec![7.0, 8.0]],
        ],
        &Device::Cpu,
    )?;
    assert_eq!(t.to_vec3::<f32>()?, z.to_vec3::<f32>()?);

    // 6D 張量 (1x1x1x1x1x6)
    let t = Tensor::from_vec(
        vec![1.0f32, 2., 3., 4., 5., 6.],
        (1, 1, 1, 1, 1, 6),
        &Device::Cpu,
    )?;
    assert_eq!(t.dims(), &[1, 1, 1, 1, 1, 6]);

    // &[N1, N2, ...] Shape
    let t = Tensor::from_vec(
        vec![1.0f32, 2., 3., 4., 5., 6.],
        &[1, 1, 1, 1, 1, 6],
        &Device::Cpu,
    )?;
    assert_eq!(t.dims(), &[1, 1, 1, 1, 1, 6]);

    // vec![N1, N2, ...] Shape
    let t = Tensor::from_vec(
        vec![1.0f32, 2., 3., 4., 5., 6.],
        vec![1, 1, 1, 1, 1, 6],
        &Device::Cpu,
    )?;
    assert_eq!(t.dims(), &[1, 1, 1, 1, 1, 6]);

    Ok(())
}
