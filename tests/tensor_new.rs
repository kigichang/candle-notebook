use candle_core::{Device, Result, Tensor};

#[test]
fn new() -> Result<()> {
    // scalar
    let t = Tensor::new(1.0f32, &Device::Cpu)?;
    println!("{:?}", t.shape());
    assert_eq!(t.to_scalar::<f32>()?, 1.0);
    assert_eq!(t.to_vec0::<f32>()?, 1.0);

    // vector
    let t = Tensor::new(&[1.0f32, 2., 3., 4.], &Device::Cpu)?;
    println!("{:?}", t.shape());
    assert_eq!(t.to_vec1::<f32>()?, vec![1.0, 2.0, 3.0, 4.0]);

    let z = Tensor::new(vec![1.0f32, 2., 3., 4.], &Device::Cpu)?;
    assert_eq!(z.to_vec1::<f32>()?, t.to_vec1::<f32>()?);

    // 2x2 矩陣
    let t = Tensor::new(&[[1.0f32, 2.], [3., 4.]], &Device::Cpu)?;
    println!("{:?}", t.shape());
    assert_eq!(t.to_vec2::<f32>()?, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);

    // 2x2x2 矩陣
    let t = Tensor::new(
        &[[[1.0f32, 2.], [3., 4.]], [[5., 6.], [7., 8.]]],
        &Device::Cpu,
    )?;
    println!("{:?}", t.shape());
    assert_eq!(
        t.to_vec3::<f32>()?,
        vec![
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec![vec![5.0, 6.0], vec![7.0, 8.0]]
        ]
    );

    // 2x2x2x2 矩陣
    let t = Tensor::new(
        &[
            [[[1.0f32, 2.], [3., 4.]], [[5., 6.], [7., 8.]]],
            [[[-1., -2.], [-3., 4.]], [[-5., 6.], [-7., -8.]]],
        ],
        &Device::Cpu,
    )?;
    println!("{:?}", t.shape());

    Ok(())
}
