use candle_core::{Device, Result, Tensor};

#[test]
fn from_vec() -> Result<()> {
    // scalar
    let t = Tensor::from_vec(vec![1.0f32], (), &Device::Cpu)?;
    assert_eq!(t.to_scalar::<f32>()?, 1.0);

    // vector
    let t = Tensor::from_vec(vec![1.0f32, 2., 3., 4.], 4, &Device::Cpu)?;
    assert_eq!(t.to_vec1::<f32>()?, vec![1.0, 2.0, 3.0, 4.0]);

    let z = Tensor::from_vec(vec![1.0f32, 2., 3., 4.], &[4], &Device::Cpu)?;
    assert_eq!(z.dims(), t.dims());
    assert_eq!(z.to_vec1::<f32>()?, t.to_vec1::<f32>()?);

    let z = Tensor::from_vec(vec![1.0f32, 2., 3., 4.], (4,), &Device::Cpu)?;
    assert_eq!(z.dims(), t.dims());
    assert_eq!(z.to_vec1::<f32>()?, t.to_vec1::<f32>()?);

    let z = Tensor::from_vec(vec![1.0f32, 2., 3., 4.], vec![4], &Device::Cpu)?;
    assert_eq!(z.dims(), t.dims());
    assert_eq!(z.to_vec1::<f32>()?, t.to_vec1::<f32>()?);

    // 2x2 矩陣
    let t = Tensor::from_vec(vec![1.0f32, 2., 3., 4.], (2, 2), &Device::Cpu)?;
    assert_eq!(t.to_vec2::<f32>()?, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);

    let z = Tensor::from_vec(vec![1.0f32, 2., 3., 4.], &[2; 2], &Device::Cpu)?;
    assert_eq!(z.dims(), t.dims());
    assert_eq!(z.to_vec2::<f32>()?, t.to_vec2::<f32>()?);

    let z = Tensor::from_vec(vec![1.0f32, 2., 3., 4.], vec![2, 2], &Device::Cpu)?;
    assert_eq!(z.dims(), t.dims());
    assert_eq!(z.to_vec2::<f32>()?, t.to_vec2::<f32>()?);

    // 2x2x2 矩陣
    let t = Tensor::from_vec(
        vec![1.0f32, 2., 3., 4., 5., 6., 7., 8.],
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

    // 1x1x1x1x1x6 矩陣
    let t = Tensor::from_vec(
        vec![1.0f32, 2., 3., 4., 5., 6.],
        (1, 1, 1, 1, 1, 6),
        &Device::Cpu,
    )?;
    println!("{:?}", t.shape());

    let t = Tensor::from_vec(
        vec![1.0f32, 2., 3., 4., 5., 6.],
        &[1, 1, 1, 1, 1, 6],
        &Device::Cpu,
    )?;
    println!("{:?}", t.shape());

    let t = Tensor::from_vec(
        vec![1.0f32, 2., 3., 4., 5., 6.],
        vec![1, 1, 1, 1, 1, 6],
        &Device::Cpu,
    )?;
    println!("{:?}", t.shape());

    Ok(())
}
