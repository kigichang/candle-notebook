use candle_core::{DType, Device, Result, Tensor};

/// 相同維度張量加法
#[test]
fn add() -> Result<()> {
    let a = Tensor::new(&[1.0f32, 2., 3., 4.], &Device::Cpu)?;
    let b = Tensor::new(&[5.0f32, 6., 7., 8.], &Device::Cpu)?;
    let c = a.add(&b)?;
    assert_eq!(c.to_vec1::<f32>()?, vec![6.0, 8.0, 10.0, 12.0]);

    let s = Tensor::new(10f32, &Device::Cpu)?;
    assert!(a.add(&s).is_err());

    let a = Tensor::from_iter(0..24u32, &Device::Cpu)?.reshape((2, 3, 4))?;
    let b = Tensor::from_iter(0..24u32, &Device::Cpu)?.reshape((2, 3, 4))?;
    let c = a.add(&b)?;
    let ans = Tensor::from_iter((0..24u32).map(|v| v * 2), &Device::Cpu)?.reshape((2, 3, 4))?;
    assert_eq!(c.to_vec3::<u32>()?, ans.to_vec3::<u32>()?);

    let z = Tensor::zeros(24, DType::U32, &Device::Cpu)?;
    assert!(c.add(&z).is_err());

    let c = (&a + &b)?;
    assert_eq!(
        c.to_vec3::<u32>()?,
        vec![
            vec![vec![0, 2, 4, 6], vec![8, 10, 12, 14], vec![16, 18, 20, 22]],
            vec![
                vec![24, 26, 28, 30],
                vec![32, 34, 36, 38],
                vec![40, 42, 44, 46]
            ]
        ]
    );

    let c = &a + &z;
    assert!(c.is_err());

    let a1 = Tensor::new(1.0f32, &Device::Cpu)?;
    let b1 = Tensor::new(2.0f64, &Device::Cpu)?;

    let c = &a1 + &b1;
    assert!(c.is_err());

    Ok(())
}

/// 相同維度張量減法
#[test]
fn sub() -> Result<()> {
    let a = Tensor::new(&[1.0f32, 2., 3., 4.], &Device::Cpu)?;
    let b = Tensor::new(&[5.0f32, 6., 7., 8.], &Device::Cpu)?;

    let c = a.sub(&b)?;
    assert_eq!(c.to_vec1::<f32>()?, vec![-4.0, -4.0, -4.0, -4.0]);

    let c = (&a - &b)?;
    assert_eq!(c.to_vec1::<f32>()?, vec![-4.0, -4.0, -4.0, -4.0]);

    let b = Tensor::new(10f32, &Device::Cpu)?;
    assert!(a.sub(&b).is_err());
    assert!((&a - &b).is_err());

    let b = Tensor::new(10u32, &Device::Cpu)?;
    assert!(a.sub(&b).is_err());
    Ok(())
}

/// 相同維度張量乘法
#[test]
fn mul() -> Result<()> {
    let a = Tensor::new(&[1.0f32, 2., 3., 4.], &Device::Cpu)?;
    let b = Tensor::new(&[5.0f32, 6., 7., 8.], &Device::Cpu)?;

    let c = a.mul(&b)?;
    assert_eq!(c.to_vec1::<f32>()?, vec![5.0, 12.0, 21.0, 32.0]);

    let c = (&a * &b)?;
    assert_eq!(c.to_vec1::<f32>()?, vec![5.0, 12.0, 21.0, 32.0]);

    let b = Tensor::new(10f32, &Device::Cpu)?;
    assert!(a.mul(&b).is_err());
    assert!((&a * &b).is_err());

    let b = Tensor::new(10u32, &Device::Cpu)?;
    assert!(a.mul(&b).is_err());
    Ok(())
}

/// 相同維度張量除法
#[test]
fn div() -> Result<()> {
    let a = Tensor::new(&[1.0f32, 2., 3., 4.], &Device::Cpu)?;
    let b = Tensor::new(&[5.0f32, 6., 7., 8.], &Device::Cpu)?;

    let c = a.div(&b)?;
    assert_eq!(c.to_vec1::<f32>()?, vec![0.2, 0.33333334, 0.42857143, 0.5]);

    let c = (&a / &b)?;
    assert_eq!(c.to_vec1::<f32>()?, vec![0.2, 0.33333334, 0.42857143, 0.5]);

    let b = Tensor::new(10f32, &Device::Cpu)?;
    assert!(a.div(&b).is_err());
    assert!((&a / &b).is_err());

    let b = Tensor::new(10u32, &Device::Cpu)?;
    assert!(a.div(&b).is_err());

    let b = a.zeros_like()?;
    let c = a.div(&b); // n / 0 = Inf
    assert!(c.is_ok()); // tensor with Inf is valid
    assert_eq!(c?.to_vec1::<f32>()?, &[f32::INFINITY; 4]);

    let a = a.zeros_like()?;
    let b = a.zeros_like()?;
    let c = a.div(&b); // 0 / 0 = NaN
    assert!(c.is_ok()); // tensor with NaN is valid
    Ok(())
}
