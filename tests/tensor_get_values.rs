use candle_core::{Device, Result, Tensor};

#[test]
fn get_values() -> Result<()> {
    let t = Tensor::new(1.0f32, &Device::Cpu)?;
    // scalar
    assert_eq!(t.to_scalar::<f32>()?, 1.0);
    assert_eq!(t.to_vec0::<f32>()?, 1.0);
    assert!(t.to_vec1::<f32>().is_err());
    assert!(t.to_vec2::<f32>().is_err());
    assert!(t.to_vec3::<f32>().is_err());

    // vector
    let t = Tensor::new(&[1.0f32, 2., 3., 4.], &Device::Cpu)?;
    assert!(t.to_scalar::<f32>().is_err());
    assert!(t.to_vec0::<f32>().is_err());
    assert_eq!(t.to_vec1::<f32>()?, vec![1.0, 2.0, 3.0, 4.0]);
    assert!(t.to_vec2::<f32>().is_err());
    assert!(t.to_vec3::<f32>().is_err());

    // 2x2 矩陣
    let t = Tensor::new(&[[1.0f32, 2.], [3., 4.]], &Device::Cpu)?;
    assert!(t.to_scalar::<f32>().is_err());
    assert!(t.to_vec0::<f32>().is_err());
    assert!(t.to_vec1::<f32>().is_err());
    assert_eq!(t.to_vec2::<f32>()?, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    assert!(t.to_vec3::<f32>().is_err());

    // 2x2x2 矩陣
    let t = Tensor::new(
        &[[[1.0f32, 2.], [3., 4.]], [[5., 6.], [7., 8.]]],
        &Device::Cpu,
    )?;
    assert!(t.to_scalar::<f32>().is_err());
    assert!(t.to_vec0::<f32>().is_err());
    assert!(t.to_vec1::<f32>().is_err());
    assert!(t.to_vec2::<f32>().is_err());
    assert_eq!(
        t.to_vec3::<f32>()?,
        vec![
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec![vec![5.0, 6.0], vec![7.0, 8.0]]
        ]
    );

    Ok(())
}
