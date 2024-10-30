use candle_core::{Device, Result, Tensor};

/// 取出張量內容
#[test]
fn get_values() -> Result<()> {
    // 純量
    let t = Tensor::new(1.0f32, &Device::Cpu)?;
    assert_eq!(t.to_scalar::<f32>()?, 1.0);
    assert_eq!(t.to_vec0::<f32>()?, 1.0);
    assert!(t.to_vec1::<f32>().is_err());
    assert!(t.to_vec2::<f32>().is_err());
    assert!(t.to_vec3::<f32>().is_err());

    // 向量
    let t = Tensor::new(&[1.0f32, 2., 3., 4.], &Device::Cpu)?;
    assert!(t.to_scalar::<f32>().is_err());
    assert!(t.to_vec0::<f32>().is_err());
    assert_eq!(t.to_vec1::<f32>()?, &[1.0, 2.0, 3.0, 4.0]);
    assert!(t.to_vec2::<f32>().is_err());
    assert!(t.to_vec3::<f32>().is_err());

    // 2D 張量 (2x2)
    let t = Tensor::new(&[[1.0f32, 2.], [3., 4.]], &Device::Cpu)?;
    assert!(t.to_scalar::<f32>().is_err());
    assert!(t.to_vec0::<f32>().is_err());
    assert!(t.to_vec1::<f32>().is_err());
    assert_eq!(t.to_vec2::<f32>()?, &[[1.0, 2.0], [3.0, 4.0]]);
    assert!(t.to_vec3::<f32>().is_err());

    // 3D 張量 (2x2x2)
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
        &[[[1.0f32, 2.], [3., 4.]], [[5., 6.], [7., 8.]]]
    );

    Ok(())
}
