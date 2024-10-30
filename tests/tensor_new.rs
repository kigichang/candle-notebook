use candle_core::{Device, Result, Tensor};

/// 使用 Tensor::new 建立張量
#[test]
fn new() -> Result<()> {
    // 純量張量
    let t = Tensor::new(1.0f32, &Device::Cpu)?;
    assert_eq!(t.dims0()?, ());
    assert_eq!(t.to_scalar::<f32>()?, 1.0);
    assert_eq!(t.to_vec0::<f32>()?, 1.0);

    // 向量
    let t = Tensor::new(&[1.0f32, 2., 3., 4.], &Device::Cpu)?;
    assert_eq!(t.dims1()?, 4);
    assert_eq!(t.to_vec1::<f32>()?, vec![1.0, 2.0, 3.0, 4.0]);
    let z = Tensor::new(vec![1.0f32, 2., 3., 4.], &Device::Cpu)?;
    assert_eq!(z.to_vec1::<f32>()?, t.to_vec1::<f32>()?);

    // 2D 張量 (2x2)
    let t = Tensor::new(&[[1.0f32, 2.], [3., 4.]], &Device::Cpu)?;
    assert_eq!(t.dims2()?, (2, 2));
    assert_eq!(t.to_vec2::<f32>()?, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);

    // 3D 張量 (2x2x2)
    let t = Tensor::new(
        &[[[1.0f32, 2.], [3., 4.]], [[5., 6.], [7., 8.]]],
        &Device::Cpu,
    )?;
    assert_eq!(t.dims3()?, (2, 2, 2));
    assert_eq!(
        t.to_vec3::<f32>()?,
        vec![
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec![vec![5.0, 6.0], vec![7.0, 8.0]]
        ]
    );

    // 4D 張量 (2x2x2x2)
    let t = Tensor::new(
        &[
            [[[1.0f32, 2.], [3., 4.]], [[5., 6.], [7., 8.]]],
            [[[-1., -2.], [-3., 4.]], [[-5., 6.], [-7., -8.]]],
        ],
        &Device::Cpu,
    )?;
    assert_eq!(t.dims4()?, (2, 2, 2, 2));

    Ok(())
}
